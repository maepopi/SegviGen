"""segvigen.full_guided — Full 3D segmentation conditioned on a 2D guidance map.

The 2D guidance map is a flat-color PNG where each part of the object is
painted a distinct solid color.

Usage
-----
>>> from segvigen.full_guided import run
>>> out_glb = run(
...     glb_path="model.glb",
...     ckpt_path="ckpt/full_seg_w_2d_map.ckpt",
...     guidance_img="guidance.png",
... )
"""

from __future__ import annotations

import tempfile

import torch
from PIL import Image

import trellis2.modules.sparse as sp

from segvigen._shared import (
    _to_cuda, _offload,
    process_glb_to_vxz, vxz_to_latent_slat,
    preprocess_image, get_cond, slat_to_glb,
    load_base_models, load_seg_model, build_sampler_params,
)
from segvigen._samplers import SamplerFull


def run(
    glb_path: str,
    ckpt_path: str,
    guidance_img: str,
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
    ckpt_path:
        Path to ``full_seg_w_2d_map.ckpt``.
    guidance_img:
        Path to the flat-color guidance PNG.
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
    base = load_base_models()
    gen3dseg = load_seg_model(ckpt_path, 'full_guided')
    sampler = SamplerFull()

    with tempfile.NamedTemporaryFile(suffix='.vxz', delete=False) as f:
        vxz_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
        out_path = f.name

    print("GLB → VXZ …")
    process_glb_to_vxz(glb_path, vxz_path)
    shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(
        base['shape_encoder'], base['shape_decoder'], base['tex_encoder'], vxz_path)

    print("Processing 2D guidance map …")
    image = Image.open(guidance_img)
    image = preprocess_image(image, remove_bg_fn=remove_bg_fn)
    _to_cuda(base['image_cond_model'])
    cond = get_cond(base['image_cond_model'], [image])
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
