"""segvigen.full — Full 3D segmentation (no click-point conditioning).

Usage
-----
>>> from segvigen.full import FullSegmenter
>>> seg = FullSegmenter(ckpt_path="ckpt/full_seg.ckpt")
>>> out_glb = seg.run(
...     glb_path="model.glb",
...     transforms_path="data_toolkit/transforms.json",
... )
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from huggingface_hub import hf_hub_download

import trellis2.modules.sparse as sp

from segvigen._shared import (
    _to_cuda, _offload,
    process_glb_to_vxz, vxz_to_latent_slat,
    preprocess_image, get_cond, slat_to_glb,
    load_base_models, load_seg_model, build_sampler_params,
)
from segvigen._samplers import SamplerFull

_HF_REPO = "fenghora/SegviGen"
_DEFAULT_CKPT = "ckpt/full_seg.ckpt"


class FullSegmenter:
    """Full 3D segmentation (image-conditioned, no click points).

    Models are loaded lazily on the first call to :meth:`run`.

    Parameters
    ----------
    ckpt_path:
        Path to ``full_seg.ckpt``.
    """

    def __init__(self, ckpt_path: str = _DEFAULT_CKPT) -> None:
        self.ckpt_path = ckpt_path
        self._base = None
        self._gen3dseg = None
        self._sampler = None
        self._lock = threading.Lock()

    def _ensure_checkpoint(self) -> None:
        """Download the checkpoint from HuggingFace if it is missing."""
        if os.path.isfile(self.ckpt_path):
            return
        print(f"[FullSegmenter] Checkpoint not found at {self.ckpt_path!r} — downloading …")
        local_dir = str(Path(self.ckpt_path).parent)
        hf_hub_download(repo_id=_HF_REPO, filename=Path(self.ckpt_path).name,
                        local_dir=local_dir)
        print(f"[FullSegmenter] Downloaded {self.ckpt_path}")

    def load(self) -> None:
        """Load base models and the segmentation model into CPU memory."""
        if self._base is not None:
            return
        self._ensure_checkpoint()
        self._base = load_base_models()
        self._gen3dseg = load_seg_model(self.ckpt_path, 'full')
        self._sampler = SamplerFull()

    def clear_vram(self) -> None:
        """Move all model weights to CPU and flush the CUDA cache."""
        if self._base is not None:
            for m in self._base.values():
                if isinstance(m, torch.nn.Module):
                    _offload(m)
        if self._gen3dseg is not None:
            _offload(self._gen3dseg)
        torch.cuda.empty_cache()

    def run(
        self,
        glb_path: str,
        transforms_path: str,
        rendered_img: Optional[Union[str, List[str]]] = None,
        remove_bg_fn=None,
        steps: int = 25,
        rescale_t: float = 1.0,
        guidance_strength: float = 7.5,
        guidance_rescale: float = 0.0,
        guidance_interval_start: float = 0.0,
        guidance_interval_end: float = 1.0,
        decimation_target: int = 100_000,
        texture_size: int = 1024,
        remesh: bool = True,
        remesh_band: int = 1,
        remesh_project: int = 0,
    ) -> str:
        """Run full segmentation on a GLB mesh.

        Parameters
        ----------
        glb_path:
            Input GLB file path.
        transforms_path:
            Path to ``transforms.json`` used for rendering the conditioning view.
        rendered_img:
            Optional pre-rendered PNG (or list of PNGs) to use instead of the
            auto-rendered view.  Several images condition on multiple views.
        steps, rescale_t, guidance_strength, guidance_rescale,
        guidance_interval_start, guidance_interval_end:
            Diffusion sampler parameters.
        decimation_target, texture_size, remesh, remesh_band, remesh_project:
            GLB export parameters.

        Returns
        -------
        str
            Absolute path to the segmented output GLB.
        """
        try:
            from data_toolkit.bpy_render import render_from_transforms
        except ImportError as exc:
            raise ImportError(
                "data_toolkit.bpy_render is required for rendering the conditioning view. "
                "Make sure the data_toolkit module is on your Python path."
            ) from exc

        with self._lock:
            self.load()
            base = self._base
            gen3dseg = self._gen3dseg
            sampler = self._sampler

            vxz_path = img_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.vxz', delete=False) as f:
                    vxz_path = f.name
                with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
                    out_path = f.name
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    img_path = f.name

                print("GLB → VXZ …")
                process_glb_to_vxz(glb_path, vxz_path)
                shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(
                    base['shape_encoder'], base['shape_decoder'], base['tex_encoder'], vxz_path)

                if rendered_img is not None:
                    img_paths = [rendered_img] if isinstance(rendered_img, str) else list(rendered_img)
                else:
                    print("Rendering conditioning image …")
                    render_from_transforms(glb_path, transforms_path, img_path)
                    img_paths = [img_path]
                print(f"Conditioning on {len(img_paths)} image(s) …")
                images = [preprocess_image(Image.open(p), remove_bg_fn=remove_bg_fn)
                          for p in img_paths]
                _to_cuda(base['image_cond_model'])
                cond = get_cond(base['image_cond_model'], images)
                _offload(base['image_cond_model'])

                sampler_params = build_sampler_params(
                    base['pipeline_args'], steps, rescale_t, guidance_strength,
                    guidance_rescale, guidance_interval_start, guidance_interval_end)

                print("Sampling …")
                pa = base['pipeline_args']
                device = shape_slat.feats.device
                shape_std = torch.tensor(pa['shape_slat_normalization']['std'])[None].to(device)
                shape_mean = torch.tensor(pa['shape_slat_normalization']['mean'])[None].to(device)
                tex_std = torch.tensor(pa['tex_slat_normalization']['std'])[None].to(device)
                tex_mean = torch.tensor(pa['tex_slat_normalization']['mean'])[None].to(device)
                shape_slat_n = (shape_slat - shape_mean) / shape_std
                tex_slat_n = (tex_slat - tex_mean) / tex_std
                coords_len_list = [shape_slat_n.coords.shape[0]]
                noise = sp.SparseTensor(torch.randn_like(tex_slat_n.feats), shape_slat_n.coords)
                _to_cuda(gen3dseg)
                output_tex_slat = sampler.sample(gen3dseg, noise, tex_slat_n, shape_slat_n,
                                                  coords_len_list, cond, sampler_params)
                _offload(gen3dseg)
                output_tex_slat = output_tex_slat * tex_std + tex_mean

                _to_cuda(base['tex_decoder'])
                with torch.no_grad():
                    tex_voxels = base['tex_decoder'](output_tex_slat, guide_subs=subs) * 0.5 + 0.5
                _offload(base['tex_decoder'])

                print("Exporting GLB …")
                glb = slat_to_glb(meshes, tex_voxels,
                                   decimation_target=int(decimation_target),
                                   texture_size=int(texture_size),
                                   remesh=remesh,
                                   remesh_band=remesh_band,
                                   remesh_project=remesh_project)
                glb.export(out_path)
                return out_path
            finally:
                self.clear_vram()
                for _p in (vxz_path, img_path):
                    if _p is not None:
                        try:
                            os.unlink(_p)
                        except OSError:
                            pass
