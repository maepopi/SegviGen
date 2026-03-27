"""server.py — FastAPI server for SegviGen.

Run with:
    uvicorn server:app --host 0.0.0.0 --port 7860 --reload
"""

import logging
import os
import shutil
import tempfile
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress repetitive job-status polling from uvicorn access logs
logging.getLogger("uvicorn.access").addFilter(
    type("_PollFilter", (logging.Filter,), {
        "filter": lambda self, r: "/api/jobs/" not in r.getMessage()
    })()
)

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import segvigen as sgv
import util
import rembg

# ─── Background removal (app-level, not part of segvigen) ──────────────────────

_rembg_session = None

def _remove_bg(image):
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = rembg.new_session("isnet-general-use")
    return rembg.remove(image.convert("RGB"), session=_rembg_session)

# ─── Checkpoint auto-download ──────────────────────────────────────────────────

_CKPT_DIR = Path(__file__).parent / "ckpt"
_HF_REPO  = "fenghora/SegviGen"
_CHECKPOINTS = [
    "full_seg.ckpt",
    "full_seg_w_2d_map.ckpt",
]

def ensure_checkpoints() -> None:
    """Download any missing checkpoints from HuggingFace."""
    _CKPT_DIR.mkdir(exist_ok=True)
    missing = [n for n in _CHECKPOINTS if not (_CKPT_DIR / n).exists()]
    if not missing:
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[SegviGen] WARNING: huggingface_hub not installed — cannot auto-download checkpoints.")
        print(f"[SegviGen] Download them manually from https://huggingface.co/{_HF_REPO}")
        return

    for name in missing:
        print(f"[SegviGen] Checkpoint not found: {name}")
        print(f"[SegviGen] Downloading from {_HF_REPO} ...")
        try:
            hf_hub_download(
                repo_id=_HF_REPO,
                filename=name,
                local_dir=str(_CKPT_DIR),
            )
            print(f"[SegviGen] ✓ {name}")
        except Exception as exc:
            print(f"[SegviGen] ✗ Failed to download {name}: {exc}")
            print(f"[SegviGen] Download it manually from https://huggingface.co/{_HF_REPO}")


# ─── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_checkpoints()
    yield

app = FastAPI(title="SegviGen", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
# Serve /assets/* directly (Vite output references /assets/... not /static/assets/...)
_ASSETS_DIR = _STATIC_DIR / "assets"
if _ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")

# ─── Job store ─────────────────────────────────────────────────────────────────
# Simple in-memory job dict — good enough for local single-user tool.

_jobs: Dict[str, Dict[str, Any]] = {}


def _run_job(job_id: str, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        _jobs[job_id] = {"status": "done", "result": result, "error": None}
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb, flush=True)  # full traceback visible in server logs
        error_msg = f"{type(exc).__name__}: {exc}"
        _jobs[job_id] = {"status": "error", "result": None, "error": error_msg}


def _start_job(fn, *args, **kwargs) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "result": None, "error": None}
    t = threading.Thread(target=_run_job, args=(job_id, fn, *args), kwargs=kwargs, daemon=True)
    t.start()
    return {"job_id": job_id}


# ─── Root ───────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ─── File upload ────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile):
    suffix = Path(file.filename).suffix if file.filename else ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        shutil.copyfileobj(file.file, f)
        return {"path": f.name}


# ─── File download ──────────────────────────────────────────────────────────────

@app.get("/api/files")
def serve_file(path: str):
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


# ─── Job status ─────────────────────────────────────────────────────────────────

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


# ─── Presets ────────────────────────────────────────────────────────────────────

@app.get("/api/presets/sampler")
def sampler_presets():
    return sgv.SAMPLER_PRESETS


@app.get("/api/presets/split")
def split_presets():
    from segvigen.presets import SPLIT_PRESETS
    return SPLIT_PRESETS


# ─── Inference endpoints ────────────────────────────────────────────────────────

class FullParams(BaseModel):
    glb_path: str
    ckpt_path: str
    transforms_path: str
    rendered_img: Optional[str] = None
    steps: int = 25
    rescale_t: float = 1.0
    guidance_strength: float = 7.5
    guidance_rescale: float = 0.0
    guidance_interval_start: float = 0.0
    guidance_interval_end: float = 1.0
    decimation_target: int = 100_000
    texture_size: int = 1024
    remesh: bool = True
    remesh_band: int = 1
    remesh_project: int = 0


class Full2DParams(BaseModel):
    glb_path: str
    ckpt_path: str
    guidance_img: str
    steps: int = 25
    rescale_t: float = 1.0
    guidance_strength: float = 7.5
    guidance_rescale: float = 0.0
    guidance_interval_start: float = 0.0
    guidance_interval_end: float = 1.0
    decimation_target: int = 100_000
    texture_size: int = 1024
    remesh: bool = True
    remesh_band: int = 1
    remesh_project: int = 0


class SplitParams(BaseModel):
    in_glb_path: str
    color_quant_step: int = 16
    palette_sample_pixels: int = 2_000_000
    palette_min_pixels: int = 500
    palette_max_colors: int = 256
    palette_merge_dist: int = 32
    samples_per_face: int = 4
    flip_v: bool = True
    uv_wrap_repeat: bool = True
    transition_conf_thresh: float = 1.0
    transition_prop_iters: int = 6
    transition_neighbor_min: int = 1
    small_component_action: str = "reassign"
    small_component_min_faces: int = 50
    postprocess_iters: int = 3
    min_faces_per_part: int = 1
    bake_transforms: bool = True


class GuidanceParams(BaseModel):
    glb_path: str
    transforms_path: str
    gemini_api_key: str
    analyze_model: str = "gemini-2.5-flash"
    generate_model: str = "gemini-3-pro-image-preview"
    resolution: int = 512
    mode: str = "single"
    grid_views: List[str] = ["front", "back", "left", "right"]
    grid_cols: int = 2


@app.post("/api/jobs/full")
def start_full(params: FullParams):
    return _start_job(sgv.full.run, remove_bg_fn=_remove_bg, **params.model_dump())


@app.post("/api/jobs/full_2d")
def start_full_2d(params: Full2DParams):
    return _start_job(sgv.full_guided.run, remove_bg_fn=_remove_bg, **params.model_dump())


@app.post("/api/jobs/split")
def start_split(params: SplitParams):
    return _start_job(util.split_glb_by_texture_palette_rgb, **params.model_dump())


@app.post("/api/jobs/guidance")
def start_guidance(params: GuidanceParams):
    def _run():
        out_path, description = util.run_pixmesh(
            glb_path=params.glb_path,
            transforms_path=params.transforms_path,
            gemini_api_key=params.gemini_api_key,
            analyze_model=params.analyze_model,
            generate_model=params.generate_model,
            resolution=params.resolution,
            mode=params.mode,
            grid_views=tuple(params.grid_views),
            grid_cols=params.grid_cols,
        )
        return {"image_path": out_path, "description": description}
    return _start_job(_run)


# ─── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
