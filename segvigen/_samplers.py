"""segvigen._samplers — Flow-matching sampler classes and DiT model wrappers.

Internal module; not part of the public API.
"""

from __future__ import annotations

from types import MethodType
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

import trellis2.modules.sparse as sp
from trellis2.modules.utils import manual_cast


# ─── Interactive flow model forward override ───────────────────────────────────

def flow_forward_interactive(self, x, t, cond, concat_cond, point_embeds, coords_len_list):
    """Monkey-patched ``forward`` injected onto the flow model for interactive mode.

    Interleaves point-embedding tokens into the sparse attention sequence so the
    DiT can attend to user-specified click positions.
    """
    x = sp.sparse_cat([x, concat_cond], dim=-1)
    if isinstance(cond, list):
        cond = sp.VarLenTensor.from_tensor_list(cond)
    h = self.input_layer(x)
    h = manual_cast(h, self.dtype)
    t_emb = self.t_embedder(t)
    t_emb = self.adaLN_modulation(t_emb)
    t_emb = manual_cast(t_emb, self.dtype)
    cond = manual_cast(cond, self.dtype)
    point_embeds = manual_cast(point_embeds, self.dtype)

    h_feats_list = []
    h_coords_list = []
    begin = 0
    for i, coords_len in enumerate(coords_len_list):
        end = begin + 2 * coords_len
        h_feats_list.append(h.feats[begin:end])
        h_coords_list.append(h.coords[begin:end])
        h_feats_list.append(point_embeds.feats[i * 10:(i + 1) * 10])
        h_coords_list.append(point_embeds.coords[i * 10:(i + 1) * 10])
        begin = end + 10
    h = sp.SparseTensor(torch.cat(h_feats_list), torch.cat(h_coords_list))

    for block in self.blocks:
        h = block(h, t_emb, cond)

    h_feats_list = []
    h_coords_list = []
    begin = 0
    for i, coords_len in enumerate(coords_len_list):
        end = begin + 2 * coords_len
        h_feats_list.append(h.feats[begin:end])
        h_coords_list.append(h.coords[begin:end])
        begin = end
    h = sp.SparseTensor(torch.cat(h_feats_list), torch.cat(h_coords_list))

    h = manual_cast(h, x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = self.out_layer(h)
    return h


# ─── Sampler (interactive) ─────────────────────────────────────────────────────

class SamplerInteractive:
    """Flow-matching sampler conditioned on both an image and 3D click points."""

    def _inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond):
        t = torch.tensor([t * 1000] * x_t.shape[0], dtype=torch.float32).cuda()
        return model(x_t, tex_slat, shape_slat, t, cond, input_points, coords_len_list)

    def guidance_inference_model(self, model, x_t, tex_slat, shape_slat, input_points,
                                  coords_len_list, t, cond_dict, guidance_strength, guidance_rescale=0.0):
        if guidance_strength == 1:
            return self._inference_model(model, x_t, tex_slat, shape_slat,
                                          input_points, coords_len_list, t, cond_dict['cond'])
        if guidance_strength == 0:
            return self._inference_model(model, x_t, tex_slat, shape_slat,
                                          input_points, coords_len_list, t, cond_dict['neg_cond'])
        pred_pos = self._inference_model(model, x_t, tex_slat, shape_slat,
                                          input_points, coords_len_list, t, cond_dict['cond'])
        pred_neg = self._inference_model(model, x_t, tex_slat, shape_slat,
                                          input_points, coords_len_list, t, cond_dict['neg_cond'])
        pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
        if guidance_rescale > 0:
            std_pos = pred_pos.std(dim=list(range(1, pred_pos.ndim)), keepdim=True)
            std_cfg = pred.std(dim=list(range(1, pred.ndim)), keepdim=True)
            pred = guidance_rescale * pred * (std_pos / std_cfg) + (1 - guidance_rescale) * pred
        return pred

    def interval_inference_model(self, model, x_t, tex_slat, shape_slat, input_points,
                                   coords_len_list, t, cond_dict, sampler_params):
        gs = sampler_params['guidance_strength']
        gi = sampler_params['guidance_interval']
        gr = sampler_params['guidance_rescale']
        if gi[0] <= t <= gi[1]:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat,
                                                  input_points, coords_len_list, t, cond_dict, gs, gr)
        return self.guidance_inference_model(model, x_t, tex_slat, shape_slat,
                                              input_points, coords_len_list, t, cond_dict, 1, gr)

    @torch.no_grad()
    def sample_once(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list,
                    t, t_prev, cond_dict, sampler_params):
        pred_v = self.interval_inference_model(model, x_t, tex_slat, shape_slat,
                                               input_points, coords_len_list, t, cond_dict, sampler_params)
        return x_t - (t - t_prev) * pred_v

    @torch.no_grad()
    def sample(self, model, noise, tex_slat, shape_slat, input_points, coords_len_list,
               cond_dict, sampler_params):
        sample = noise
        steps = sampler_params['steps']
        rescale_t = sampler_params['rescale_t']
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
        for t, t_prev in tqdm(t_pairs, desc="Sampling"):
            sample = self.sample_once(model, sample, tex_slat, shape_slat,
                                      input_points, coords_len_list, t, t_prev, cond_dict, sampler_params)
        return sample


# ─── Sampler (full / full-2d) ──────────────────────────────────────────────────

class SamplerFull:
    """Flow-matching sampler conditioned on a single image (no click points)."""

    def _inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, cond):
        t = torch.tensor([t * 1000] * x_t.shape[0], dtype=torch.float32).cuda()
        return model(x_t, tex_slat, shape_slat, t, cond, coords_len_list)

    def guidance_inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list,
                                  t, cond_dict, guidance_strength, guidance_rescale=0.0):
        if guidance_strength == 1:
            return self._inference_model(model, x_t, tex_slat, shape_slat,
                                          coords_len_list, t, cond_dict['cond'])
        if guidance_strength == 0:
            return self._inference_model(model, x_t, tex_slat, shape_slat,
                                          coords_len_list, t, cond_dict['neg_cond'])
        pred_pos = self._inference_model(model, x_t, tex_slat, shape_slat,
                                          coords_len_list, t, cond_dict['cond'])
        pred_neg = self._inference_model(model, x_t, tex_slat, shape_slat,
                                          coords_len_list, t, cond_dict['neg_cond'])
        pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
        if guidance_rescale > 0:
            std_pos = pred_pos.std(dim=list(range(1, pred_pos.ndim)), keepdim=True)
            std_cfg = pred.std(dim=list(range(1, pred.ndim)), keepdim=True)
            pred = guidance_rescale * pred * (std_pos / std_cfg) + (1 - guidance_rescale) * pred
        return pred

    def interval_inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list,
                                   t, cond_dict, sampler_params):
        gs = sampler_params['guidance_strength']
        gi = sampler_params['guidance_interval']
        gr = sampler_params['guidance_rescale']
        if gi[0] <= t <= gi[1]:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat,
                                                  coords_len_list, t, cond_dict, gs, gr)
        return self.guidance_inference_model(model, x_t, tex_slat, shape_slat,
                                              coords_len_list, t, cond_dict, 1, gr)

    @torch.no_grad()
    def sample_once(self, model, x_t, tex_slat, shape_slat, coords_len_list,
                    t, t_prev, cond_dict, sampler_params):
        pred_v = self.interval_inference_model(model, x_t, tex_slat, shape_slat,
                                               coords_len_list, t, cond_dict, sampler_params)
        return x_t - (t - t_prev) * pred_v

    @torch.no_grad()
    def sample(self, model, noise, tex_slat, shape_slat, coords_len_list, cond_dict, sampler_params):
        sample = noise
        steps = sampler_params['steps']
        rescale_t = sampler_params['rescale_t']
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
        for t, t_prev in tqdm(t_pairs, desc="Sampling"):
            sample = self.sample_once(model, sample, tex_slat, shape_slat,
                                      coords_len_list, t, t_prev, cond_dict, sampler_params)
        return sample


# ─── Gen3DSeg model wrappers ────────────────────────────────────────────────────

class Gen3DSegInteractive(nn.Module):
    """Wraps the TRELLIS.2 flow model for interactive (click-point) segmentation."""

    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model
        self.seg_embeddings = nn.Embedding(1, 1536)

    def get_positional_encoding(self, input_points):
        point_feats_embed = torch.zeros((10, 1536), dtype=torch.float32).to(
            input_points['point_slats'].feats.device)
        labels = input_points['point_labels'].squeeze(-1)
        point_feats_embed[labels == 1] = self.seg_embeddings.weight
        return sp.SparseTensor(point_feats_embed, input_points['point_slats'].coords)

    def forward(self, x_t, tex_slats, shape_slats, t, cond, input_points, coords_len_list):
        input_tex_feats_list, input_tex_coords_list = [], []
        shape_feats_list, shape_coords_list = [], []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            input_tex_feats_list += [x_t.feats[begin:end], tex_slats.feats[begin:end]]
            input_tex_coords_list += [x_t.coords[begin:end], tex_slats.coords[begin:end]]
            shape_feats_list += [shape_slats.feats[begin:end], shape_slats.feats[begin:end]]
            shape_coords_list += [shape_slats.coords[begin:end], shape_slats.coords[begin:end]]
            begin = end
        x_t = sp.SparseTensor(torch.cat(input_tex_feats_list), torch.cat(input_tex_coords_list))
        shape_slats = sp.SparseTensor(torch.cat(shape_feats_list), torch.cat(shape_coords_list))
        point_embeds = self.get_positional_encoding(input_points)
        output_tex_slats = self.flow_model(x_t, t, cond, shape_slats, point_embeds, coords_len_list)
        out_feats, out_coords = [], []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            out_feats.append(output_tex_slats.feats[begin:end])
            out_coords.append(output_tex_slats.coords[begin:end])
            begin = begin + 2 * coords_len
        return sp.SparseTensor(torch.cat(out_feats), torch.cat(out_coords))


class Gen3DSegFull(nn.Module):
    """Wraps the TRELLIS.2 flow model for full / full-2D segmentation (no click points)."""

    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x_t, tex_slats, shape_slats, t, cond, coords_len_list):
        input_tex_feats_list, input_tex_coords_list = [], []
        shape_feats_list, shape_coords_list = [], []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            input_tex_feats_list += [x_t.feats[begin:end], tex_slats.feats[begin:end]]
            input_tex_coords_list += [x_t.coords[begin:end], tex_slats.coords[begin:end]]
            shape_feats_list += [shape_slats.feats[begin:end], shape_slats.feats[begin:end]]
            shape_coords_list += [shape_slats.coords[begin:end], shape_slats.coords[begin:end]]
            begin = end
        x_t = sp.SparseTensor(torch.cat(input_tex_feats_list), torch.cat(input_tex_coords_list))
        shape_slats = sp.SparseTensor(torch.cat(shape_feats_list), torch.cat(shape_coords_list))
        output_tex_slats = self.flow_model(x_t, t, cond, shape_slats)
        out_feats, out_coords = [], []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            out_feats.append(output_tex_slats.feats[begin:end])
            out_coords.append(output_tex_slats.coords[begin:end])
            begin = begin + 2 * coords_len
        return sp.SparseTensor(torch.cat(out_feats), torch.cat(out_coords))
