"""segvigen.full_guided — Full 3D segmentation conditioned on a 2D guidance map.

The 2D guidance map is a flat-color PNG where each part of the object is
painted a distinct solid color.

Usage
-----
>>> from segvigen.full_guided import FullGuidedSegmenter
>>> seg = FullGuidedSegmenter(ckpt_path="ckpt/full_seg_w_2d_map.ckpt")
>>> out_glb = seg.run(
...     glb_path="model.glb",
...     guidance_img="guidance.png",
... )
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import List, Union

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
_DEFAULT_CKPT = "ckpt/full_seg_w_2d_map.ckpt"


class FullGuidedSegmenter:
    """Full 3D segmentation conditioned on a 2D flat-color guidance map.

    Models are loaded lazily on the first call to :meth:`run`.

    Parameters
    ----------
    ckpt_path:
        Path to ``full_seg_w_2d_map.ckpt``.
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
        print(f"[FullGuidedSegmenter] Checkpoint not found at {self.ckpt_path!r} — downloading …")
        local_dir = str(Path(self.ckpt_path).parent)
        hf_hub_download(repo_id=_HF_REPO, filename=Path(self.ckpt_path).name,
                        local_dir=local_dir)
        print(f"[FullGuidedSegmenter] Downloaded {self.ckpt_path}")

    def load(self) -> None:
        """Load base models and the segmentation model into CPU memory."""
        if self._base is not None:
            return
        self._ensure_checkpoint()
        self._base = load_base_models()
        self._gen3dseg = load_seg_model(self.ckpt_path, 'full_guided')
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
        guidance_img: Union[str, List[str]],
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
        """Run full segmentation conditioned on a 2D flat-color guidance map.

        Parameters
        ----------
        glb_path:
            Input GLB file path.
        guidance_img:
            Path to the flat-color guidance PNG, or a list of paths to
            condition on several views at once.
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
        with self._lock:
            self.load()
            base = self._base
            gen3dseg = self._gen3dseg
            sampler = self._sampler

            vxz_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.vxz', delete=False) as f:
                    vxz_path = f.name
                with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
                    out_path = f.name

                print("GLB → VXZ …")
                process_glb_to_vxz(glb_path, vxz_path)
                shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(
                    base['shape_encoder'], base['shape_decoder'], base['tex_encoder'], vxz_path)

                guidance_paths = [guidance_img] if isinstance(guidance_img, str) else list(guidance_img)
                print(f"Processing {len(guidance_paths)} 2D guidance map(s) …")
                images = [preprocess_image(Image.open(p), remove_bg_fn=remove_bg_fn)
                          for p in guidance_paths]
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
                if vxz_path is not None:
                    try:
                        os.unlink(vxz_path)
                    except OSError:
                        pass
