"""guidance_map.py — 2D guidance map generators for SegviGen (Tab 3 input).

Each method produces a flat-color segmented PNG where every distinct part of
the 3D model is painted a unique solid color.  Feed the result into the
"Full Segmentation + 2D Guidance Map" tab to steer SegviGen's part boundaries.

Implemented methods
-------------------
- ``run_pixmesh`` : Pixmesh 2D render
    Single-view or multi-view grid mode.

    Single: render main isometric view → VLM describe → generate segmented image.
    Grid:   render N canonical views → describe grid → generate per-view in
            parallel → assemble segmented grid → save as single PNG.
"""

from __future__ import annotations

import base64
import concurrent.futures
import copy
import json
import math
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw

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

# POV occlusion hints: parts whose names contain an opposite-face keyword are
# likely hidden in that view.  Ported from MeshAgenticSegmenterVisual.
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
    """Compute a Blender camera-to-world 4×4 matrix.

    Blender cameras look down their local -Z axis.
    Columns of the returned matrix are [right | up | back | position].
    """
    eye    = np.array(eye,      dtype=float)
    target = np.array(target,   dtype=float)
    wup    = np.array(world_up, dtype=float)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    # Degenerate: forward nearly parallel to world_up → switch fallback
    if abs(np.dot(forward, wup)) > 0.999:
        wup = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, wup)
    right /= np.linalg.norm(right)

    up   = np.cross(right, forward)   # guaranteed unit
    back = -forward                   # camera local Z = back

    return [
        [right[0], up[0], back[0], eye[0]],
        [right[1], up[1], back[1], eye[1]],
        [right[2], up[2], back[2], eye[2]],
        [0.0,      0.0,   0.0,    1.0   ],
    ]


# Camera distance after BpyRenderer scene normalization (~unit cube at origin)
_CAM_DIST = 2.0

def _canonical_cameras() -> Dict[str, List[List[float]]]:
    """Return camera-to-world matrices for the 6 canonical orthographic views."""
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
    """Load the GLB once then render from every camera matrix.

    Reuses BpyRenderer internals to avoid reloading the scene for each view.
    """
    import bpy
    from data_toolkit.bpy_render import BpyRenderer

    renderer = BpyRenderer(resolution=resolution, engine="CYCLES")
    renderer.init_render_settings()
    renderer.init_scene()
    renderer.load_object(glb_path)
    renderer.normalize_scene()
    cam = renderer.init_camera()
    renderer.init_lighting()
    # Match the FOV used by the existing conditioning renders (transforms.json)
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
    """Render the first camera in transforms.json (main 3Q isometric view)."""
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
    """Stitch per-view images into a rows×cols grid with optional view labels."""
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
    """Infer visible/occluded part lists per view from part name keywords.

    Ported from MeshAgenticSegmenterVisual._compute_pov_visibility.
    """
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
    """Assign Kelly palette colors to every leaf part in the assembly tree."""
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
    """Send one image (single view or assembled grid) to a Gemini text model."""
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
    """Generate a flat-color segmented image for one view."""
    W, H = image_size
    n_colors = len(color_table)

    # Filter color table to parts expected to be visible in this view
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
    mode: str = "single",               # "single" | "grid"
    grid_views: Tuple[str, ...] = ("front", "back", "left", "right"),
    grid_cols: int = 2,
    resolution: int = 512,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[str, Dict[str, Any]]:
    """Pixmesh pipeline.  Returns ``(guidance_map_path, assembly_tree_dict)``.

    mode="single":
        Renders the main 3Q view from transforms.json, describes it, generates
        one segmented image.

    mode="grid":
        Renders each view in ``grid_views`` from canonical camera positions,
        assembles a grid for the describe step (full-object context), then
        generates each view's segmentation in parallel and stitches the results
        into a single grid PNG.
    """
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

    # Read the main-view matrix from transforms.json so we can render it in
    # the same scene-loading pass as the canonical views.
    with open(transforms_path) as _f:
        _t = json.load(_f)[0]
    cameras["main"] = _t["transform_matrix"]

    # Always render the union of describe views + main view in one pass.
    _DESCRIBE_VIEWS = ("front", "back", "left", "right")
    all_views_needed = dict.fromkeys(list(_DESCRIBE_VIEWS) + ["main"])  # ordered, deduped
    all_cameras = {v: cameras[v] for v in all_views_needed if v in cameras}

    print(f"Pixmesh grid [1/3]: rendering {len(all_cameras)} views …")
    all_images = _render_views_bpy(glb_path, all_cameras, resolution=resolution)

    # Describe from a fixed 4-view 2×2 grid for consistent full-object context
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

    # Generate segmentation from the main transforms.json view so the output
    # matches the angle SegviGen was conditioned on.
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
