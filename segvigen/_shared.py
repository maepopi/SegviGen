"""segvigen._shared — Shared model loading, voxel I/O, image preprocessing,
and GLB export utilities used by all three segmentation modules.

Not part of the public API; import from the top-level package instead.
"""

from __future__ import annotations

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import tempfile
from collections import OrderedDict
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import trimesh
import o_voxel
from torch.nn import functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import trellis2.modules.sparse as sp
from trellis2 import models
from trellis2.modules.utils import manual_cast
from trellis2.representations import MeshWithVoxel
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor


# ─── Global model cache ────────────────────────────────────────────────────────

_loaded_models: Dict[str, Any] = {}


# ─── VRAM helpers ──────────────────────────────────────────────────────────────

def _to_cuda(model):
    return model.cuda()


def _offload(model):
    model.cpu()
    torch.cuda.empty_cache()
    return model


# ─── Scene / texture preprocessing ────────────────────────────────────────────

def make_texture_square_pow2(img: Image.Image, target_size: Optional[int] = None) -> Image.Image:
    w, h = img.size
    max_side = max(w, h)
    pow2 = 1
    while pow2 < max_side:
        pow2 *= 2
    if target_size is not None:
        pow2 = target_size
    pow2 = min(pow2, 2048)
    return img.resize((pow2, pow2), Image.BILINEAR)


def preprocess_scene_textures(asset: trimesh.Scene) -> trimesh.Scene:
    if not isinstance(asset, trimesh.Scene):
        return asset
    TEX_KEYS = ["baseColorTexture", "normalTexture", "metallicRoughnessTexture",
                "emissiveTexture", "occlusionTexture"]
    for geom in asset.geometry.values():
        visual = getattr(geom, "visual", None)
        mat = getattr(visual, "material", None)
        if mat is None:
            continue
        for key in TEX_KEYS:
            if not hasattr(mat, key):
                continue
            tex = getattr(mat, key)
            if tex is None:
                continue
            if isinstance(tex, Image.Image):
                setattr(mat, key, make_texture_square_pow2(tex))
            elif hasattr(tex, "image") and tex.image is not None:
                img = tex.image
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                tex.image = make_texture_square_pow2(img)
        if hasattr(mat, "image") and mat.image is not None:
            img = mat.image
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            mat.image = make_texture_square_pow2(img)
    return asset


def _ensure_texture_visuals(asset: trimesh.Scene) -> trimesh.Scene:
    if not isinstance(asset, trimesh.Scene):
        return asset
    for name, geom in asset.geometry.items():
        if isinstance(geom.visual, trimesh.visual.color.ColorVisuals):
            geom.visual = geom.visual.to_texture()
    return asset


def _ensure_pbr_materials(asset: trimesh.Scene) -> trimesh.Scene:
    if not isinstance(asset, trimesh.Scene):
        return asset
    for name, geom in asset.geometry.items():
        mat = getattr(getattr(geom, 'visual', None), 'material', None)
        if isinstance(mat, trimesh.visual.material.SimpleMaterial):
            pbr = trimesh.visual.material.PBRMaterial()
            if mat.image is not None:
                img = mat.image if isinstance(mat.image, Image.Image) else Image.fromarray(mat.image)
                pbr.baseColorTexture = img
            else:
                c = np.array(mat.diffuse, dtype=np.uint8)
                pbr.baseColorFactor = c if len(c) == 4 else np.append(c[:3], 255).astype(np.uint8)
            geom.visual.material = pbr
    return asset


# ─── Voxel I/O ─────────────────────────────────────────────────────────────────

def process_glb_to_vxz(glb_path: str, vxz_path: str) -> None:
    """Convert a GLB mesh to a .vxz sparse voxel file (512³ grid + PBR attrs)."""
    asset = trimesh.load(glb_path, force='scene')
    asset = preprocess_scene_textures(asset)
    asset = _ensure_texture_visuals(asset)
    asset = _ensure_pbr_materials(asset)
    aabb = asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()
    asset.apply_translation(-center)
    asset.apply_scale(scale)
    mesh = asset.to_mesh()
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False
    )
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]
    voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
        asset, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], timing=False
    )
    vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
    mapping_mat = torch.argsort(vid_mat)
    attributes = {k: v[mapping_mat] for k, v in attributes.items()}
    dual_vertices = dual_vertices * 512 - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
    attributes['dual_vertices'] = dual_vertices
    attributes['intersected'] = intersected
    o_voxel.io.write(vxz_path, voxel_indices, attributes)


def vxz_to_latent_slat(shape_encoder, shape_decoder, tex_encoder, vxz_path: str):
    """Encode a .vxz file into shape and texture sparse latents."""
    coords, data = o_voxel.io.read(vxz_path)
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1).cuda()
    vertices = (data['dual_vertices'].cuda() / 255)
    intersected = torch.cat([data['intersected'] % 2, data['intersected'] // 2 % 2,
                             data['intersected'] // 4 % 2], dim=-1).bool().cuda()
    vertices_sparse = sp.SparseTensor(vertices, coords)
    intersected_sparse = sp.SparseTensor(intersected.float(), coords)
    _to_cuda(shape_encoder)
    with torch.no_grad():
        shape_slat = shape_encoder(vertices_sparse, intersected_sparse)
        shape_slat = sp.SparseTensor(shape_slat.feats.cuda(), shape_slat.coords.cuda())
    _offload(shape_encoder)

    _to_cuda(shape_decoder)
    with torch.no_grad():
        shape_decoder.set_resolution(512)
        meshes, subs = shape_decoder(shape_slat, return_subs=True)
    _offload(shape_decoder)

    base_color = (data['base_color'] / 255)
    metallic = (data['metallic'] / 255)
    roughness = (data['roughness'] / 255)
    alpha = (data['alpha'] / 255)
    attr = torch.cat([base_color, metallic, roughness, alpha], dim=-1).float().cuda() * 2 - 1

    _to_cuda(tex_encoder)
    with torch.no_grad():
        tex_slat = tex_encoder(sp.SparseTensor(attr, coords))
    _offload(tex_encoder)

    return shape_slat, meshes, subs, tex_slat


# ─── Image preprocessing ───────────────────────────────────────────────────────


def preprocess_image(input: Image.Image, remove_bg_fn=None) -> Image.Image:
    """Resize, optionally remove background, crop to content, and composite.

    Parameters
    ----------
    input : PIL.Image.Image
        Input image (RGB or RGBA).
    remove_bg_fn : callable, optional
        A function ``(PIL.Image) -> PIL.Image`` that removes the background
        and returns an RGBA image.  When *None* and the image has no alpha
        channel the image is used as-is (no background removal).
    """
    if input.mode != "RGB":
        bg = Image.new("RGB", input.size, (255, 255, 255))
        bg.paste(input, mask=input.split()[3])
        input = bg
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)),
                             Image.Resampling.LANCZOS)
    if has_alpha:
        output = input
    elif remove_bg_fn is not None:
        input = input.convert('RGB')
        output = remove_bg_fn(input)
    else:
        # No alpha and no background-removal function — use as-is with full-white alpha.
        input = input.convert('RGB')
        output = input.convert('RGBA')
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output


def get_cond(image_cond_model, image):
    """Extract DINOv3 conditioning features from an image."""
    image_cond_model.image_size = 512
    cond = image_cond_model(image)
    neg_cond = torch.zeros_like(cond)
    return {'cond': cond, 'neg_cond': neg_cond}


# ─── GLB export ────────────────────────────────────────────────────────────────

def slat_to_glb(meshes, tex_voxels, resolution: int = 512,
                decimation_target: int = 100000, texture_size: int = 4096,
                remesh: bool = True, remesh_band: int = 1, remesh_project: int = 0):
    """Decode sparse texture latents + meshes into a GLB scene."""
    pbr_attr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }
    out_mesh = []
    for m, v in zip(meshes, tex_voxels):
        m.fill_holes()
        out_mesh.append(
            MeshWithVoxel(
                m.vertices, m.faces,
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / resolution,
                coords=v.coords[:, 1:],
                attrs=v.feats,
                voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                layout=pbr_attr_layout
            )
        )
    mesh = out_mesh[0]
    mesh.simplify(10000000)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=remesh_band,
        remesh_project=remesh_project,
        verbose=True
    )
    # Undo the Z-up → Y-up axis swap that to_glb applies, since our input
    # GLB was already Y-up.  Inverse of (x,y,z)→(x,z,−y) is (x,−z,y) = +90° about X.
    rot = np.array([
        [1,  0,  0, 0],
        [0,  0, -1, 0],
        [0,  1,  0, 0],
        [0,  0,  0, 1],
    ], dtype=np.float64)
    glb.apply_transform(rot)
    return glb


# ─── Model loading ──────────────────────────────────────────────────────────────

def load_base_models() -> Dict[str, Any]:
    """Load and cache TRELLIS.2-4B base models (shape/tex encoder/decoder + DINOv3)."""
    if 'base' in _loaded_models:
        return _loaded_models['base']

    print("Loading base models (TRELLIS.2-4B) …")
    shape_encoder = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").eval()
    tex_encoder = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16").eval()
    shape_decoder = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").eval()
    tex_decoder = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").eval()
    image_cond_model = DinoV3FeatureExtractor(model_name="athena2634/dinov3-vitl16-pretrain-lvd1689m")

    pipeline_json_path = hf_hub_download(repo_id="microsoft/TRELLIS.2-4B", filename="pipeline.json")
    with open(pipeline_json_path, "r") as f:
        pipeline_config = json.load(f)
    pipeline_args = pipeline_config['args']

    base = {
        'shape_encoder': shape_encoder,
        'tex_encoder': tex_encoder,
        'shape_decoder': shape_decoder,
        'tex_decoder': tex_decoder,
        'image_cond_model': image_cond_model,
        'pipeline_args': pipeline_args,
    }
    _loaded_models['base'] = base
    print("Base models loaded.")
    return base


def load_seg_model(ckpt_path: str, mode: str):
    """Load and cache a SegviGen fine-tuned flow model.

    Parameters
    ----------
    ckpt_path:
        Path to the ``.ckpt`` file downloaded from ``fenghora/SegviGen``.
    mode:
        ``"interactive"``, ``"full"``, or ``"full_2d"``.
    """
    from segvigen._samplers import Gen3DSegInteractive, Gen3DSegFull, flow_forward_interactive

    cache_key = f"seg_{mode}_{ckpt_path}"
    if cache_key in _loaded_models:
        return _loaded_models[cache_key]

    print(f"Loading segmentation model from {ckpt_path} …")
    flow_model = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")

    if mode == 'interactive':
        flow_model.forward = MethodType(flow_forward_interactive, flow_model)
        gen3dseg = Gen3DSegInteractive(flow_model)
    else:
        gen3dseg = Gen3DSegFull(flow_model)

    state_dict = torch.load(ckpt_path)['state_dict']
    state_dict = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in state_dict.items()])
    gen3dseg.load_state_dict(state_dict)
    gen3dseg.eval()

    _loaded_models[cache_key] = gen3dseg
    print("Segmentation model loaded.")
    return gen3dseg


def build_sampler_params(pipeline_args: Dict[str, Any],
                          steps: int, rescale_t: float,
                          guidance_strength: float, guidance_rescale: float,
                          guidance_interval_start: float, guidance_interval_end: float) -> Dict[str, Any]:
    """Merge pipeline defaults with user-supplied sampler overrides."""
    params = dict(pipeline_args['tex_slat_sampler']['params'])
    params['steps'] = steps
    params['rescale_t'] = rescale_t
    params['guidance_strength'] = guidance_strength
    params['guidance_rescale'] = guidance_rescale
    params['guidance_interval'] = [guidance_interval_start, guidance_interval_end]
    return params
