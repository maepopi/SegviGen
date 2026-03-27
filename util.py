"""util.py — Utility functions for the SegviGen app.

Contains two subsystems that are NOT part of the segvigen package:

Split
-----
Post-process a segmented GLB by splitting it into per-part sub-meshes based
on the texture-encoded color palette.  Main entry-point:
``split_glb_by_texture_palette_rgb()``.

Guidance Map
------------
Generate a flat-color 2D segmentation map (PNG) using Gemini VLM + image
generation.  Main entry-point: ``run_pixmesh()``.
"""

from __future__ import annotations

import base64
import concurrent.futures
import copy
import json
import math
import os
import struct
import tempfile
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import trimesh
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ═════════════════════════════════════════════════════════════════════════════════
#  SPLIT — Split a segmented GLB into per-part sub-meshes
# ═════════════════════════════════════════════════════════════════════════════════

CHUNK_TYPE_JSON = 0x4E4F534A  # b'JSON'
CHUNK_TYPE_BIN = 0x004E4942   # b'BIN\0'


def _default_out_path(in_path: str) -> str:
    root, ext = os.path.splitext(in_path)
    if ext.lower() not in [".glb", ".gltf"]:
        ext = ".glb"
    return f"{root}_seg.glb"


def _quantize_rgb(rgb: np.ndarray, step: int) -> np.ndarray:
    if step is None or step <= 0:
        return rgb
    q = (rgb.astype(np.int32) + step // 2) // step * step
    return np.clip(q, 0, 255).astype(np.uint8)


def _load_glb_json_and_bin(glb_path: str) -> Tuple[dict, bytes]:
    data = open(glb_path, "rb").read()
    if len(data) < 12:
        raise RuntimeError("Invalid GLB: too small")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise RuntimeError("Not a GLB file (missing glTF header)")
    offset = 12
    gltf_json = None
    bin_chunk = None
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset: offset + chunk_len]
        offset += chunk_len
        if chunk_type == CHUNK_TYPE_JSON:
            gltf_json = chunk_data.decode("utf-8", errors="replace")
        elif chunk_type == CHUNK_TYPE_BIN:
            bin_chunk = chunk_data
    if gltf_json is None:
        raise RuntimeError("GLB missing JSON chunk")
    if bin_chunk is None:
        raise RuntimeError("GLB missing BIN chunk")
    return json.loads(gltf_json), bin_chunk


def _extract_basecolor_texture_image(glb_path: str, debug_print: bool = False) -> np.ndarray:
    gltf, bin_chunk = _load_glb_json_and_bin(glb_path)
    materials = gltf.get("materials", [])
    textures = gltf.get("textures", [])
    images = gltf.get("images", [])
    buffer_views = gltf.get("bufferViews", [])
    if not materials:
        raise RuntimeError("No materials in GLB")
    pbr = materials[0].get("pbrMetallicRoughness", {})
    base_tex_index = pbr.get("baseColorTexture", {}).get("index", None)
    if base_tex_index is None:
        raise RuntimeError("Material has no baseColorTexture")
    if base_tex_index >= len(textures):
        raise RuntimeError("baseColorTexture index out of range")
    tex = textures[base_tex_index]
    img_index = tex.get("source", None)
    if img_index is None or img_index >= len(images):
        raise RuntimeError("Texture has no valid image source")
    img_info = images[img_index]
    bv_index = img_info.get("bufferView", None)
    mime = img_info.get("mimeType", None)
    if bv_index is None:
        uri = img_info.get("uri", None)
        raise RuntimeError(f"Image is not embedded (bufferView missing). uri={uri}")
    if bv_index >= len(buffer_views):
        raise RuntimeError("image.bufferView out of range")
    bv = buffer_views[bv_index]
    bo = int(bv.get("byteOffset", 0))
    bl = int(bv.get("byteLength", 0))
    img_bytes = bin_chunk[bo: bo + bl]
    if debug_print:
        print(
            f"[Texture] baseColorTextureIndex={base_tex_index}, imageIndex={img_index}, "
            f"bufferView={bv_index}, mime={mime}, bytes={len(img_bytes)}"
        )
    pil = Image.open(trimesh.util.wrap_as_stream(img_bytes)).convert("RGBA")
    return np.array(pil, dtype=np.uint8)


def _merge_palette_rgb(
    palette_rgb: np.ndarray, counts: np.ndarray, merge_dist: float, debug_print: bool = False
) -> np.ndarray:
    if palette_rgb is None or len(palette_rgb) == 0:
        return palette_rgb
    if merge_dist is None or merge_dist <= 0:
        return palette_rgb
    rgb = palette_rgb.astype(np.float32)
    counts = counts.astype(np.int64)
    order = np.argsort(-counts)
    centers = []
    center_w = []
    thr2 = float(merge_dist) * float(merge_dist)
    for idx in order:
        x = rgb[idx]
        w = int(counts[idx])
        if not centers:
            centers.append(x.copy())
            center_w.append(w)
            continue
        C = np.stack(centers, axis=0)
        d2 = np.sum((C - x[None, :]) ** 2, axis=1)
        k = int(np.argmin(d2))
        if float(d2[k]) <= thr2:
            cw = center_w[k]
            centers[k] = (centers[k] * cw + x * w) / (cw + w)
            center_w[k] = cw + w
        else:
            centers.append(x.copy())
            center_w.append(w)
    merged = np.clip(np.rint(np.stack(centers, axis=0)), 0, 255).astype(np.uint8)
    if debug_print:
        print(f"[PaletteMerge] before={len(palette_rgb)} after={len(merged)} merge_dist={merge_dist}")
    return merged


def _build_palette_rgb(
    tex_rgba: np.ndarray,
    color_quant_step: int,
    palette_sample_pixels: int,
    palette_min_pixels: int,
    palette_max_colors: int,
    palette_merge_dist: int,
    debug_print: bool = False,
) -> np.ndarray:
    rgb = tex_rgba[:, :, :3].reshape(-1, 3)
    n = rgb.shape[0]
    if n > palette_sample_pixels:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=palette_sample_pixels, replace=False)
        rgb = rgb[idx]
    rgb = _quantize_rgb(rgb, color_quant_step)
    uniq, counts = np.unique(rgb, axis=0, return_counts=True)
    order = np.argsort(-counts)
    uniq = uniq[order]
    counts = counts[order]
    keep = counts >= palette_min_pixels
    uniq = uniq[keep]
    counts = counts[keep]
    if len(uniq) > palette_max_colors:
        uniq = uniq[:palette_max_colors]
        counts = counts[:palette_max_colors]
    if debug_print:
        print(
            f"[Palette] quant_step={color_quant_step} palette_size(before_merge)={len(uniq)} "
            f"min_pixels={palette_min_pixels}"
        )
    uniq = _merge_palette_rgb(uniq.astype(np.uint8), counts, palette_merge_dist, debug_print)
    if debug_print:
        print(f"[Palette] palette_size(after_merge)={len(uniq)}")
    return uniq.astype(np.uint8)


def _unwrap_uv3_for_seam(uv3: np.ndarray) -> np.ndarray:
    out = uv3.copy()
    for d in range(2):
        v = out[:, :, d]
        vmin = v.min(axis=1)
        vmax = v.max(axis=1)
        seam = (vmax - vmin) > 0.5
        if np.any(seam):
            vv = v[seam]
            vv = np.where(vv < 0.5, vv + 1.0, vv)
            out[seam, :, d] = vv
    return out


def _barycentric_samples(uv3: np.ndarray, samples_per_face: int) -> np.ndarray:
    uv3 = _unwrap_uv3_for_seam(uv3)
    if samples_per_face == 1:
        w = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        uvs = uv3[:, 0, :] * w[0] + uv3[:, 1, :] * w[1] + uv3[:, 2, :] * w[2]
        return uvs[:, None, :]
    ws = np.array(
        [[1/3, 1/3, 1/3], [0.80, 0.10, 0.10], [0.10, 0.80, 0.10], [0.10, 0.10, 0.80]],
        dtype=np.float32,
    )
    uvs = (
        uv3[:, None, 0, :] * ws[None, :, 0, None]
        + uv3[:, None, 1, :] * ws[None, :, 1, None]
        + uv3[:, None, 2, :] * ws[None, :, 2, None]
    )
    return uvs


def _sample_texture_nearest_rgb(
    tex_rgba: np.ndarray, uv: np.ndarray, flip_v: bool, uv_wrap_repeat: bool
) -> np.ndarray:
    h, w = tex_rgba.shape[0], tex_rgba.shape[1]
    if uv_wrap_repeat:
        uv = np.mod(uv, 1.0)
    else:
        uv = np.clip(uv, 0.0, 1.0)
    u = uv[:, 0]
    v = 1.0 - uv[:, 1] if flip_v else uv[:, 1]
    x = np.rint(u * (w - 1)).astype(np.int32)
    y = np.rint(v * (h - 1)).astype(np.int32)
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return tex_rgba[y, x, :3].astype(np.uint8)


def _map_to_palette_rgb(
    colors_rgb: np.ndarray, palette_rgb: np.ndarray, chunk: int = 20000
) -> Tuple[np.ndarray, np.ndarray]:
    if palette_rgb is None or len(palette_rgb) == 0:
        uniq, inv = np.unique(colors_rgb, axis=0, return_inverse=True)
        return inv.astype(np.int32), uniq.astype(np.uint8)
    c = colors_rgb.astype(np.float32)
    p = palette_rgb.astype(np.float32)
    out = np.empty((c.shape[0],), dtype=np.int32)
    for i in range(0, c.shape[0], chunk):
        cc = c[i: i + chunk]
        d2 = ((cc[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
        out[i: i + chunk] = np.argmin(d2, axis=1).astype(np.int32)
    return out, palette_rgb


def _face_labels_from_texture_rgb(
    mesh: trimesh.Trimesh,
    tex_rgba: np.ndarray,
    palette_rgb: np.ndarray,
    color_quant_step: int,
    samples_per_face: int,
    flip_v: bool,
    uv_wrap_repeat: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        return None
    uv = np.asarray(uv, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] != len(mesh.vertices):
        return None
    faces = mesh.faces
    uv3 = uv[faces]
    uvs = _barycentric_samples(uv3, samples_per_face)
    F, S = uvs.shape[0], uvs.shape[1]
    flat_uv = uvs.reshape(-1, 2)
    sampled_rgb = _sample_texture_nearest_rgb(tex_rgba, flat_uv, flip_v, uv_wrap_repeat)
    sampled_rgb = _quantize_rgb(sampled_rgb, color_quant_step)
    sample_label, used_palette = _map_to_palette_rgb(sampled_rgb, palette_rgb)
    sample_label = sample_label.reshape(F, S)
    if S == 1:
        return sample_label[:, 0].astype(np.int32), used_palette
    l0, l1, l2, l3 = sample_label[:, 0], sample_label[:, 1], sample_label[:, 2], sample_label[:, 3]
    c0 = 1 + (l0 == l1) + (l0 == l2) + (l0 == l3)
    c1 = 1 + (l1 == l0) + (l1 == l2) + (l1 == l3)
    c2 = 1 + (l2 == l0) + (l2 == l1) + (l2 == l3)
    c3 = 1 + (l3 == l0) + (l3 == l1) + (l3 == l2)
    counts = np.stack([c0, c1, c2, c3], axis=1)
    vals = np.stack([l0, l1, l2, l3], axis=1)
    best = vals[np.arange(F), np.argmax(counts, axis=1)]
    return best.astype(np.int32), used_palette


def _get_physical_face_adjacency(mesh: trimesh.Trimesh) -> np.ndarray:
    v_rounded = np.round(mesh.vertices, decimals=3)
    v_unique, inv_indices = np.unique(v_rounded, axis=0, return_inverse=True)
    physical_faces = inv_indices[mesh.faces]
    tmp_mesh = trimesh.Trimesh(vertices=v_unique, faces=physical_faces, process=False)
    return tmp_mesh.face_adjacency


def smooth_face_labels_by_topology(
    mesh: trimesh.Trimesh,
    face_label: np.ndarray,
    small_component_min_faces: int = 50,
    small_component_action: str = "reassign",
    postprocess_iters: int = 3,
    debug_print: bool = False,
) -> np.ndarray:
    labels = face_label.copy()
    edges = _get_physical_face_adjacency(mesh)
    F = len(mesh.faces)

    # Phase 1: same-color connected component smoothing
    for iteration in range(postprocess_iters):
        same_label = labels[edges[:, 0]] == labels[edges[:, 1]]
        sub_edges = edges[same_label]
        if len(sub_edges) > 0:
            data = np.ones(len(sub_edges), dtype=bool)
            graph = coo_matrix((data, (sub_edges[:, 0], sub_edges[:, 1])), shape=(F, F))
            graph = graph.maximum(graph.T)
            n_components, comp_labels = connected_components(graph, directed=False)
        else:
            n_components = F
            comp_labels = np.arange(F)

        comp_sizes = np.bincount(comp_labels, minlength=n_components)
        small_comps = np.where(comp_sizes < small_component_min_faces)[0]
        if len(small_comps) == 0:
            break

        is_small = np.isin(comp_labels, small_comps)
        mask0 = is_small[edges[:, 0]]
        mask1 = is_small[edges[:, 1]]
        boundary_edges_0 = edges[mask0 & ~mask1]
        boundary_edges_1 = edges[mask1 & ~mask0]
        b_inner = np.concatenate([boundary_edges_0[:, 0], boundary_edges_1[:, 1]])
        b_outer = np.concatenate([boundary_edges_0[:, 1], boundary_edges_1[:, 0]])

        if len(b_inner) == 0:
            break

        if small_component_action == "drop":
            labels[is_small] = -1
        else:
            outer_labels = labels[b_outer]
            inner_comps = comp_labels[b_inner]
            for cid in np.unique(inner_comps):
                cid_mask = inner_comps == cid
                surrounding_labels = outer_labels[cid_mask]
                if len(surrounding_labels) > 0:
                    best_label = np.bincount(surrounding_labels).argmax()
                    labels[comp_labels == cid] = best_label

    # Phase 2: full physical adjacency for remaining small components
    same_label = labels[edges[:, 0]] == labels[edges[:, 1]]
    sub_edges = edges[same_label]
    if len(sub_edges) > 0:
        data = np.ones(len(sub_edges), dtype=bool)
        graph = coo_matrix((data, (sub_edges[:, 0], sub_edges[:, 1])), shape=(F, F))
        graph = graph.maximum(graph.T)
        n_components, comp_labels = connected_components(graph, directed=False)
    else:
        n_components = F
        comp_labels = np.arange(F)

    comp_sizes = np.bincount(comp_labels, minlength=n_components)
    small_comps_set = set(np.where(comp_sizes < small_component_min_faces)[0])

    if small_comps_set and small_component_action == "reassign":
        adj = defaultdict(set)
        for e0, e1 in edges:
            adj[int(e0)].add(int(e1))
            adj[int(e1)].add(int(e0))

        for _ in range(3):
            changed = False
            small_comps_now = set(int(c) for c in range(n_components) if comp_sizes[c] < small_component_min_faces and c in small_comps_set)
            if not small_comps_now:
                break
            for cid in small_comps_now:
                cid_faces = np.where(comp_labels == cid)[0]
                neighbor_labels = []
                for fi in cid_faces:
                    for nf in adj[int(fi)]:
                        if comp_labels[nf] != cid:
                            neighbor_labels.append(labels[nf])
                if len(neighbor_labels) > 0:
                    best_label = int(np.bincount(neighbor_labels).argmax())
                    labels[cid_faces] = best_label
                    changed = True
            if not changed:
                break
            same_label = labels[edges[:, 0]] == labels[edges[:, 1]]
            sub_edges = edges[same_label]
            if len(sub_edges) > 0:
                data = np.ones(len(sub_edges), dtype=bool)
                graph = coo_matrix((data, (sub_edges[:, 0], sub_edges[:, 1])), shape=(F, F))
                graph = graph.maximum(graph.T)
                n_components, comp_labels = connected_components(graph, directed=False)
            else:
                n_components = F
                comp_labels = np.arange(F)
            comp_sizes = np.bincount(comp_labels, minlength=n_components)
            small_comps_set = set(np.where(comp_sizes < small_component_min_faces)[0])

    # Phase 3: orphan faces by centroid distance
    same_label = labels[edges[:, 0]] == labels[edges[:, 1]]
    sub_edges = edges[same_label]
    if len(sub_edges) > 0:
        data = np.ones(len(sub_edges), dtype=bool)
        graph = coo_matrix((data, (sub_edges[:, 0], sub_edges[:, 1])), shape=(F, F))
        graph = graph.maximum(graph.T)
        _, comp_labels = connected_components(graph, directed=False)
    else:
        comp_labels = np.arange(F)
    comp_sizes = np.bincount(comp_labels)
    orphan_comps = set(np.where(comp_sizes < small_component_min_faces)[0])

    if orphan_comps:
        orphan_mask = np.array([comp_labels[i] in orphan_comps for i in range(F)])
        non_orphan_mask = ~orphan_mask
        if non_orphan_mask.any() and orphan_mask.any():
            centroids = mesh.triangles_center
            orphan_indices = np.where(orphan_mask)[0]
            non_orphan_indices = np.where(non_orphan_mask)[0]
            non_orphan_centroids = centroids[non_orphan_indices]
            for oi in orphan_indices:
                dists = np.linalg.norm(non_orphan_centroids - centroids[oi], axis=1)
                nearest = non_orphan_indices[np.argmin(dists)]
                labels[oi] = labels[nearest]
            if debug_print:
                print(f"  [Phase3] Assigned {int(orphan_mask.sum())} orphan faces by centroid proximity")

    return labels


def split_glb_by_texture_palette_rgb(
    in_glb_path: str,
    out_glb_path: Optional[str] = None,
    min_faces_per_part: int = 1,
    bake_transforms: bool = True,
    color_quant_step: int = 16,
    palette_sample_pixels: int = 2_000_000,
    palette_min_pixels: int = 500,
    palette_max_colors: int = 256,
    palette_merge_dist: int = 32,
    samples_per_face: int = 4,
    flip_v: bool = True,
    uv_wrap_repeat: bool = True,
    transition_conf_thresh: float = 1.0,
    transition_prop_iters: int = 6,
    transition_neighbor_min: int = 1,
    small_component_action: str = "reassign",
    small_component_min_faces: int = 50,
    postprocess_iters: int = 3,
    debug_print: bool = True,
) -> str:
    if out_glb_path is None:
        out_glb_path = _default_out_path(in_glb_path)

    tex_rgba = _extract_basecolor_texture_image(in_glb_path, debug_print=debug_print)
    palette_rgb = _build_palette_rgb(
        tex_rgba,
        color_quant_step=color_quant_step,
        palette_sample_pixels=palette_sample_pixels,
        palette_min_pixels=palette_min_pixels,
        palette_max_colors=palette_max_colors,
        palette_merge_dist=palette_merge_dist,
        debug_print=debug_print,
    )

    scene = trimesh.load(in_glb_path, force="scene", process=False)
    out_scene = trimesh.Scene()
    part_count = 0
    base = os.path.splitext(os.path.basename(in_glb_path))[0]

    for node_name in scene.graph.nodes_geometry:
        geom_name = scene.graph[node_name][1]
        if geom_name is None:
            continue
        geom = scene.geometry.get(geom_name, None)
        if geom is None or not isinstance(geom, trimesh.Trimesh):
            continue

        mesh = geom.copy()
        if bake_transforms:
            T, _ = scene.graph.get(node_name)
            if T is not None:
                mesh.apply_transform(T)

        res = _face_labels_from_texture_rgb(
            mesh, tex_rgba, palette_rgb,
            color_quant_step=color_quant_step,
            samples_per_face=samples_per_face,
            flip_v=flip_v,
            uv_wrap_repeat=uv_wrap_repeat,
        )
        if res is None:
            if debug_print:
                print(f"[{node_name}] no uv / cannot sample -> keep orig")
            out_scene.add_geometry(mesh, geom_name=f"{base}__{node_name}__orig")
            continue

        face_label, label_rgb = res

        face_label = smooth_face_labels_by_topology(
            mesh, face_label,
            small_component_min_faces=small_component_min_faces,
            small_component_action=small_component_action,
            postprocess_iters=postprocess_iters,
            debug_print=debug_print,
        )

        if debug_print:
            uniq_labels, cnts = np.unique(face_label, return_counts=True)
            order = np.argsort(-cnts)
            print(
                f"[{node_name}] faces={len(mesh.faces)} labels_used={len(uniq_labels)} palette_size={len(label_rgb)}"
            )
            for i in order[:10]:
                lab = int(uniq_labels[i])
                r, g, b = ([int(x) for x in label_rgb[lab]] if 0 <= lab < len(label_rgb) else (0, 0, 0))
                print(f"  label={lab} rgb=({r},{g},{b}) faces={int(cnts[i])}")

        groups = defaultdict(list)
        for fi, lab in enumerate(face_label):
            if int(lab) >= 0:
                groups[int(lab)].append(fi)

        for lab, face_ids in groups.items():
            if len(face_ids) < min_faces_per_part:
                continue
            sub = mesh.submesh([np.array(face_ids, dtype=np.int64)], append=True, repair=False)
            if sub is None:
                continue
            if isinstance(sub, (list, tuple)):
                if not sub:
                    continue
                sub = sub[0]
            if 0 <= lab < len(label_rgb):
                r, g, b = [int(x) for x in label_rgb[lab]]
                part_name = f"{base}__{node_name}__label_{lab}__rgb_{r}_{g}_{b}"
            else:
                part_name = f"{base}__{node_name}__label_{lab}"
            out_scene.add_geometry(sub, geom_name=part_name)
            part_count += 1

    if part_count == 0:
        if debug_print:
            print("[INFO] part_count==0, fallback to original scene export.")
        out_scene = scene

    out_scene.export(out_glb_path)
    return out_glb_path


# ═════════════════════════════════════════════════════════════════════════════════
#  GUIDANCE MAP — Generate flat-color 2D segmentation maps via Gemini VLM
# ═════════════════════════════════════════════════════════════════════════════════

# ── Kelly 22-color palette ─────────────────────────────────────────────────
_KELLY_PALETTE: List[str] = [
    "#FFB300", "#803E75", "#FF6800", "#A6BDD7", "#C10020",
    "#CEA262", "#817066", "#007D34", "#F6768E", "#00538A",
    "#FF7A5C", "#53377A", "#FF8E00", "#B32851", "#F4C800",
    "#7F180D", "#93AA00", "#593315", "#F13A13", "#232C16",
    "#0000FF", "#00FF00",
]

# Canonical view names in display order
CANONICAL_VIEW_NAMES: List[str] = ["front", "back", "left", "right", "top", "bottom"]

# POV occlusion hints
_POV_OWN: Dict[str, set] = {
    "front":  {"front"},
    "back":   {"back"},
    "left":   {"left"},
    "right":  {"right"},
    "top":    {"top", "upper", "roof"},
    "bottom": {"bottom", "base", "floor", "foot", "feet", "lower"},
}
_POV_OPP: Dict[str, set] = {
    "front":  {"back"},
    "back":   {"front"},
    "left":   {"right"},
    "right":  {"left"},
    "top":    {"bottom", "base", "floor", "foot", "feet", "lower"},
    "bottom": {"top", "upper", "roof"},
}


# ── Camera utilities ───────────────────────────────────────────────────────

def _look_at_matrix(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    world_up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> List[List[float]]:
    eye    = np.array(eye,      dtype=float)
    target = np.array(target,   dtype=float)
    wup    = np.array(world_up, dtype=float)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    if abs(np.dot(forward, wup)) > 0.999:
        wup = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, wup)
    right /= np.linalg.norm(right)

    up   = np.cross(right, forward)
    back = -forward

    return [
        [right[0], up[0], back[0], eye[0]],
        [right[1], up[1], back[1], eye[1]],
        [right[2], up[2], back[2], eye[2]],
        [0.0,      0.0,   0.0,    1.0   ],
    ]


_CAM_DIST = 2.0

def _canonical_cameras() -> Dict[str, List[List[float]]]:
    D = _CAM_DIST
    return {
        "front":  _look_at_matrix([ 0,  -D,   0], world_up=(0, 0, 1)),
        "back":   _look_at_matrix([ 0,   D,   0], world_up=(0, 0, 1)),
        "left":   _look_at_matrix([-D,   0,   0], world_up=(0, 0, 1)),
        "right":  _look_at_matrix([ D,   0,   0], world_up=(0, 0, 1)),
        "top":    _look_at_matrix([ 0,   0,   D], world_up=(0, 1, 0)),
        "bottom": _look_at_matrix([ 0,   0,  -D], world_up=(0, 1, 0)),
    }


# ── Rendering ─────────────────────────────────────────────────────────────

def _render_views_bpy(
    glb_path: str,
    view_cameras: Dict[str, List[List[float]]],
    resolution: int = 512,
) -> Dict[str, Image.Image]:
    import bpy
    from data_toolkit.bpy_render import BpyRenderer

    renderer = BpyRenderer(resolution=resolution, engine="CYCLES")
    renderer.init_render_settings()
    renderer.init_scene()
    renderer.load_object(glb_path)
    renderer.normalize_scene()
    cam = renderer.init_camera()
    renderer.init_lighting()
    cam.data.lens = 16 / math.tan(0.698 / 2)

    results: Dict[str, Image.Image] = {}
    for view_name, matrix in view_cameras.items():
        renderer.set_camera_from_matrix(cam, matrix)
        bpy.context.view_layer.update()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        results[view_name] = Image.open(out_path).convert("RGB")
        print(f"  rendered {view_name} → {out_path}")

    return results


def _render_main_view(
    glb_path: str,
    transforms_path: str,
    resolution: int = 512,
) -> Image.Image:
    with open(transforms_path) as f:
        t = json.load(f)[0]
    matrix = t["transform_matrix"]
    results = _render_views_bpy(glb_path, {"main": matrix}, resolution)
    return results["main"]


# ── Grid assembly ──────────────────────────────────────────────────────────

def _assemble_grid(
    images: Dict[str, Image.Image],
    view_order: List[str],
    cols: int,
    tile_size: int = 512,
    add_labels: bool = True,
) -> Image.Image:
    present = [v for v in view_order if v in images]
    rows = math.ceil(len(present) / cols)
    grid = Image.new("RGB", (cols * tile_size, rows * tile_size), (255, 255, 255))

    for idx, name in enumerate(present):
        tile = images[name].resize((tile_size, tile_size), Image.LANCZOS)
        row, col = divmod(idx, cols)
        x, y = col * tile_size, row * tile_size
        grid.paste(tile, (x, y))

        if add_labels:
            draw = ImageDraw.Draw(grid)
            label = name.upper()
            draw.rectangle([x + 2, y + 2, x + len(label) * 7 + 6, y + 16], fill=(0, 0, 0))
            draw.text((x + 4, y + 3), label, fill=(255, 255, 255))

    return grid


# ── POV visibility ─────────────────────────────────────────────────────────

def _compute_pov_visibility(
    color_table: Dict[str, str],
) -> Dict[str, Dict[str, List[str]]]:
    result: Dict[str, Dict[str, List[str]]] = {
        v: {"visible": [], "occluded": []} for v in CANONICAL_VIEW_NAMES
    }
    for part_name in color_table:
        words = set(part_name.lower().split())
        for view in CANONICAL_VIEW_NAMES:
            if words & _POV_OPP[view]:
                result[view]["occluded"].append(part_name)
            else:
                result[view]["visible"].append(part_name)
    return result


# ── Shared utilities ───────────────────────────────────────────────────────

def _assign_palette(
    description: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    updated = copy.deepcopy(description)
    parts_ordered: List[str] = []
    for obj in updated.get("objects", []):
        for group in obj.get("assembly_tree", []):
            for part in group.get("parts", []):
                name = part.get("name", "").strip()
                if name and name not in parts_ordered:
                    parts_ordered.append(name)

    color_table: Dict[str, str] = {
        name: _KELLY_PALETTE[i % len(_KELLY_PALETTE)]
        for i, name in enumerate(parts_ordered)
    }
    for obj in updated.get("objects", []):
        for group in obj.get("assembly_tree", []):
            for part in group.get("parts", []):
                name = part.get("name", "").strip()
                if name in color_table:
                    part["assigned_color_hex"] = color_table[name]

    return updated, color_table


def _img_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ── Gemini API calls ───────────────────────────────────────────────────────

def _gemini_describe(
    image: Image.Image,
    api_key: str,
    model: str = "gemini-2.5-flash",
    is_grid: bool = False,
) -> Dict[str, Any]:
    view_context = (
        "a grid showing the object from multiple canonical angles "
        "(front, back, left, right, top, bottom)"
        if is_grid else
        "an isometric view of the object"
    )
    system_prompt = (
        "### ROLE\n"
        "Senior 3D Product Analyst and Mechanical Design Expert.\n\n"
        "### OBJECTIVES\n"
        "1. VISUALLY INSPECT all views to identify every distinct component.\n"
        "2. FAVOUR DETAIL: list every part you can visually distinguish.\n"
        "3. GENERATE the most complete assembly tree possible.\n\n"
        "### OUTPUT FORMAT\n"
        "Return a SINGLE valid JSON object. No markdown fences, no extra text.\n"
    )
    user_prompt = (
        f"You will receive {view_context}.\n\n"
        "Decompose the object into its constituent parts and return this JSON:\n"
        "{\n"
        '  "scene_description": "<max 5 words>",\n'
        '  "language": "en",\n'
        '  "objects": [{\n'
        '    "category": "<Object Name>",\n'
        '    "assembly_tree": [{\n'
        '      "group_name": "<Functional Group>",\n'
        '      "parts": [\n'
        '        {"name": "<Part Name>", "base_color_hex": "<HEX>", "material": "<material>"}\n'
        "      ]\n"
        "    }]\n"
        "  }]\n"
        "}\n\n"
        "Rules:\n"
        "1. List every visually-distinct part as a SEPARATE entry.\n"
        "   Only merge when two regions are truly INDISTINGUISHABLE.\n"
        "2. REPEATED INSTANCES get unique positional names "
        "   (e.g. 'Leg Front Left', 'Leg Front Right', 'Leg Back Left', 'Leg Back Right').\n"
        "   NEVER group multiple physical instances under one name.\n"
        "3. Short names only (max 3 words, no color adjectives).\n"
        "4. Do NOT hallucinate parts not visible in the image."
    )
    img_b64 = _img_to_b64(image)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                {"text": user_prompt},
            ]
        }],
        "generationConfig": {"maxOutputTokens": 8000, "temperature": 0.1},
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    j0, j1 = text.find("{"), text.rfind("}") + 1
    if j0 < 0 or j1 <= j0:
        raise ValueError(f"No JSON object found in VLM response:\n{text[:400]}")
    return json.loads(text[j0:j1])


def _gemini_generate_segmentation(
    image: Image.Image,
    description: Dict[str, Any],
    color_table: Dict[str, str],
    api_key: str,
    model: str = "gemini-3-pro-image-preview",
    image_size: Tuple[int, int] = (512, 512),
    bg_color_hex: str = "#ffffff",
    view_name: str = "main",
    pov_visibility: Optional[Dict[str, List[str]]] = None,
) -> Image.Image:
    W, H = image_size
    n_colors = len(color_table)

    if pov_visibility and view_name in pov_visibility:
        vis  = pov_visibility[view_name]["visible"]
        occ  = pov_visibility[view_name]["occluded"]
        visible_ct = {n: c for n, c in color_table.items() if n in vis} or color_table
    else:
        vis, occ = list(color_table.keys()), []
        visible_ct = color_table

    n_visible = len(visible_ct)
    color_table_str = "\n".join(f"  {n} → {c}" for n, c in visible_ct.items())
    vis_str = ", ".join(f"{p} ({visible_ct[p]})" for p in vis if p in visible_ct) or "—"
    occ_str = ", ".join(occ) or "—"
    json_str = json.dumps(description, indent=2, ensure_ascii=False)

    prompt = (
        "You are an expert 3D Segmentation Colorist performing a STRICT, PURE RECOLORING task.\n\n"
        f"## VIEW: {view_name.upper()} — Image size: {W}×{H} px\n\n"
        f"Parts visible from this angle: {vis_str}\n"
        f"Parts occluded (DO NOT PAINT THESE): {occ_str}\n\n"
        "## JSON ASSEMBLY TREE (reference only)\n"
        f"```json\n{json_str}\n```\n\n"
        f"## COLOR TABLE — ONLY these {n_visible} hex codes are allowed\n"
        f"{color_table_str}\n\n"
        f"Hard limit: at most {n_visible} distinct part colors in the output.\n\n"
        "## RULES\n"
        f"1. Output must be exactly {W}×{H} px — no cropping, no padding.\n"
        f"2. Background ({bg_color_hex}) is NOT a part — leave all background pixels untouched.\n"
        "3. Flood-fill each enclosed region with its single flat hex color.\n"
        "   ONE region = ONE color. No gradients, no outlines, no contour lines.\n"
        "4. Object silhouette must be pixel-accurate to the input.\n"
        "5. Do NOT invent new part boundaries or sub-regions.\n"
        f"6. Use ONLY the {n_visible} hex codes listed above — no other colors.\n"
        f"7. If a part is not visible in the {view_name.upper()} view, do NOT paint it."
    )

    img_b64 = _img_to_b64(image)
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
            ]
        }],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()

    parts = resp.json()["candidates"][0]["content"]["parts"]
    for part in parts:
        if "inlineData" in part:
            img_data = base64.b64decode(part["inlineData"]["data"])
            img = Image.open(BytesIO(img_data)).convert("RGB")
            if img.size != (W, H):
                img = img.resize((W, H), Image.LANCZOS)
            return img
    raise ValueError(
        "Gemini image generation returned no image. "
        "Check that the model supports image output and the API key is valid.\n"
        f"Raw response: {resp.text[:400]}"
    )


# ── Public entry-point: Pixmesh 2D render ─────────────────────────────────

def run_pixmesh(
    glb_path: str,
    transforms_path: str,
    gemini_api_key: str,
    analyze_model: str = "gemini-2.5-flash",
    generate_model: str = "gemini-3-pro-image-preview",
    mode: str = "single",
    grid_views: Tuple[str, ...] = ("front", "back", "left", "right"),
    grid_cols: int = 2,
    resolution: int = 512,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[str, Dict[str, Any]]:
    """Pixmesh pipeline.  Returns ``(guidance_map_path, assembly_tree_dict)``."""
    if not gemini_api_key or not gemini_api_key.strip():
        raise ValueError("A Gemini API key is required for the Pixmesh method.")

    bg_hex = "#%02x%02x%02x" % bg_color

    # ── Single-view mode ──────────────────────────────────────────────────
    if mode == "single":
        print("Pixmesh single [1/3]: rendering main view …")
        rendered = _render_main_view(glb_path, transforms_path, resolution=resolution)

        print("Pixmesh single [2/3]: describing mesh …")
        description = _gemini_describe(rendered, gemini_api_key, model=analyze_model)
        print(f"  → {description.get('scene_description', '?')}")

        updated_desc, color_table = _assign_palette(description)
        print(f"  → {len(color_table)} parts: {', '.join(color_table.keys())}")

        print("Pixmesh single [3/3]: generating segmentation …")
        seg_image = _gemini_generate_segmentation(
            rendered, updated_desc, color_table,
            api_key=gemini_api_key, model=generate_model,
            image_size=(resolution, resolution), bg_color_hex=bg_hex,
            view_name="main",
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        seg_image.save(out_path)
        print(f"Pixmesh single: done → {out_path}")
        return out_path, updated_desc

    # ── Grid mode ─────────────────────────────────────────────────────────
    cameras = _canonical_cameras()
    selected = {v: cameras[v] for v in grid_views if v in cameras}
    if not selected:
        raise ValueError(f"No valid views in grid_views={grid_views}.")

    with open(transforms_path) as _f:
        _t = json.load(_f)[0]
    cameras["main"] = _t["transform_matrix"]

    _DESCRIBE_VIEWS = ("front", "back", "left", "right")
    all_views_needed = dict.fromkeys(list(_DESCRIBE_VIEWS) + ["main"])
    all_cameras = {v: cameras[v] for v in all_views_needed if v in cameras}

    print(f"Pixmesh grid [1/3]: rendering {len(all_cameras)} views …")
    all_images = _render_views_bpy(glb_path, all_cameras, resolution=resolution)

    describe_views_present = [v for v in _DESCRIBE_VIEWS if v in all_images]
    describe_grid = _assemble_grid(
        all_images, describe_views_present, cols=2,
        tile_size=resolution, add_labels=True,
    )

    print("Pixmesh grid [2/3]: describing mesh from 4-view grid …")
    description = _gemini_describe(
        describe_grid, gemini_api_key, model=analyze_model, is_grid=True,
    )
    print(f"  → {description.get('scene_description', '?')}")

    updated_desc, color_table = _assign_palette(description)
    print(f"  → {len(color_table)} parts: {', '.join(color_table.keys())}")

    gen_view = "main"
    pov_visibility = _compute_pov_visibility(color_table)

    print(f"Pixmesh grid [3/3]: generating segmentation from '{gen_view}' view …")
    seg_image = _gemini_generate_segmentation(
        all_images[gen_view], updated_desc, color_table,
        api_key=gemini_api_key, model=generate_model,
        image_size=(resolution, resolution), bg_color_hex=bg_hex,
        view_name=gen_view, pov_visibility=pov_visibility,
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        out_path = f.name
    seg_image.save(out_path)
    print(f"Pixmesh grid: done → {out_path}")
    return out_path, updated_desc
