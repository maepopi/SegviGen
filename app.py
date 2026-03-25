import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from huggingface_hub import hf_hub_download
import torch
import trimesh
import o_voxel
import numpy as np
import torch.nn as nn
import tempfile
import trellis2.modules.sparse as sp

from PIL import Image
from tqdm import tqdm
from trellis2 import models
from types import MethodType
from collections import OrderedDict
from torch.nn import functional as F
import rembg
from trellis2.modules.utils import manual_cast
from trellis2.representations import MeshWithVoxel
from data_toolkit.bpy_render import render_from_transforms
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

import gradio as gr
import split as splitter
import guidance_map as gmap

# ─── Global model cache ────────────────────────────────────────────────────────
_loaded_models = {}

# ─── VRAM helpers — keep peak usage under 12 GB by offloading models ──────────

def _to_cuda(model):
    """Move a model to GPU."""
    return model.cuda()

def _offload(model):
    """Move a model to CPU and free cached VRAM."""
    model.cpu()
    torch.cuda.empty_cache()
    return model

# ─── Shared utilities (identical in both inference scripts) ────────────────────

def make_texture_square_pow2(img: Image.Image, target_size=None):
    w, h = img.size
    max_side = max(w, h)
    pow2 = 1
    while pow2 < max_side:
        pow2 *= 2
    if target_size is not None:
        pow2 = target_size
    pow2 = min(pow2, 2048)
    return img.resize((pow2, pow2), Image.BILINEAR)


def preprocess_scene_textures(asset):
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


def _ensure_texture_visuals(asset):
    """Convert any ColorVisuals geometry to TextureVisuals (required by o_voxel)."""
    if not isinstance(asset, trimesh.Scene):
        return asset
    for name, geom in asset.geometry.items():
        if isinstance(geom.visual, trimesh.visual.color.ColorVisuals):
            geom.visual = geom.visual.to_texture()
    return asset


def _ensure_pbr_materials(asset):
    """Convert SimpleMaterial to PBRMaterial (required by o_voxel)."""
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


def process_glb_to_vxz(glb_path, vxz_path):
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


def vxz_to_latent_slat(shape_encoder, shape_decoder, tex_encoder, vxz_path):
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


_rembg_session = None


def _remove_background(image: Image.Image) -> Image.Image:
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = rembg.new_session("isnet-general-use")
    return rembg.remove(image.convert("RGB"), session=_rembg_session)


def preprocess_image(input):
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
    else:
        input = input.convert('RGB')
        output = _remove_background(input)
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
    image_cond_model.image_size = 512
    cond = image_cond_model(image)
    neg_cond = torch.zeros_like(cond)
    return {'cond': cond, 'neg_cond': neg_cond}


def slat_to_glb(meshes, tex_voxels, resolution=512,
                decimation_target=100000, texture_size=4096,
                remesh=True, remesh_band=1, remesh_project=0):
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
    return glb


# ─── Sampler (interactive variant, with point embeddings) ─────────────────────

class SamplerInteractive:
    def _inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond):
        t = torch.tensor([t * 1000] * x_t.shape[0], dtype=torch.float32).cuda()
        return model(x_t, tex_slat, shape_slat, t, cond, input_points, coords_len_list)

    def guidance_inference_model(self, model, x_t, tex_slat, shape_slat, input_points,
                                  coords_len_list, t, cond_dict, guidance_strength, guidance_rescale=0.0):
        if guidance_strength == 1:
            return self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['cond'])
        elif guidance_strength == 0:
            return self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['neg_cond'])
        else:
            pred_pos = self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['cond'])
            pred_neg = self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['neg_cond'])
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            if guidance_rescale > 0:
                x_0_pos = pred_pos  # simplified
                x_0_cfg = pred
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                pred = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
            return pred

    def interval_inference_model(self, model, x_t, tex_slat, shape_slat, input_points,
                                   coords_len_list, t, cond_dict, sampler_params):
        guidance_strength = sampler_params['guidance_strength']
        guidance_interval = sampler_params['guidance_interval']
        guidance_rescale = sampler_params['guidance_rescale']
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, input_points,
                                                  coords_len_list, t, cond_dict, guidance_strength, guidance_rescale)
        else:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, input_points,
                                                  coords_len_list, t, cond_dict, 1, guidance_rescale)

    @torch.no_grad()
    def sample_once(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list,
                    t, t_prev, cond_dict, sampler_params):
        pred_v = self.interval_inference_model(model, x_t, tex_slat, shape_slat, input_points,
                                               coords_len_list, t, cond_dict, sampler_params)
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
            sample = self.sample_once(model, sample, tex_slat, shape_slat, input_points,
                                      coords_len_list, t, t_prev, cond_dict, sampler_params)
        return sample


# ─── Sampler (full variant, no points) ────────────────────────────────────────

class SamplerFull:
    def _inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, cond):
        t = torch.tensor([t * 1000] * x_t.shape[0], dtype=torch.float32).cuda()
        return model(x_t, tex_slat, shape_slat, t, cond, coords_len_list)

    def guidance_inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list,
                                  t, cond_dict, guidance_strength, guidance_rescale=0.0):
        if guidance_strength == 1:
            return self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['cond'])
        elif guidance_strength == 0:
            return self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['neg_cond'])
        else:
            pred_pos = self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['cond'])
            pred_neg = self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['neg_cond'])
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            if guidance_rescale > 0:
                std_pos = pred_pos.std(dim=list(range(1, pred_pos.ndim)), keepdim=True)
                std_cfg = pred.std(dim=list(range(1, pred.ndim)), keepdim=True)
                x_0_rescaled = pred * (std_pos / std_cfg)
                pred = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * pred
            return pred

    def interval_inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list,
                                   t, cond_dict, sampler_params):
        guidance_strength = sampler_params['guidance_strength']
        guidance_interval = sampler_params['guidance_interval']
        guidance_rescale = sampler_params['guidance_rescale']
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list,
                                                  t, cond_dict, guidance_strength, guidance_rescale)
        else:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list,
                                                  t, cond_dict, 1, guidance_rescale)

    @torch.no_grad()
    def sample_once(self, model, x_t, tex_slat, shape_slat, coords_len_list,
                    t, t_prev, cond_dict, sampler_params):
        pred_v = self.interval_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list,
                                               t, cond_dict, sampler_params)
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
            sample = self.sample_once(model, sample, tex_slat, shape_slat, coords_len_list,
                                      t, t_prev, cond_dict, sampler_params)
        return sample


# ─── Gen3DSeg models ──────────────────────────────────────────────────────────

def flow_forward_interactive(self, x, t, cond, concat_cond, point_embeds, coords_len_list):
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


class Gen3DSegInteractive(nn.Module):
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
        input_tex_feats_list = []
        input_tex_coords_list = []
        shape_feats_list = []
        shape_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            input_tex_feats_list.append(x_t.feats[begin:end])
            input_tex_feats_list.append(tex_slats.feats[begin:end])
            input_tex_coords_list.append(x_t.coords[begin:end])
            input_tex_coords_list.append(tex_slats.coords[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            begin = end
        x_t = sp.SparseTensor(torch.cat(input_tex_feats_list), torch.cat(input_tex_coords_list))
        shape_slats = sp.SparseTensor(torch.cat(shape_feats_list), torch.cat(shape_coords_list))
        point_embeds = self.get_positional_encoding(input_points)
        output_tex_slats = self.flow_model(x_t, t, cond, shape_slats, point_embeds, coords_len_list)
        output_tex_feats_list = []
        output_tex_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            output_tex_feats_list.append(output_tex_slats.feats[begin:end])
            output_tex_coords_list.append(output_tex_slats.coords[begin:end])
            begin = begin + 2 * coords_len
        return sp.SparseTensor(torch.cat(output_tex_feats_list), torch.cat(output_tex_coords_list))


class Gen3DSegFull(nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x_t, tex_slats, shape_slats, t, cond, coords_len_list):
        input_tex_feats_list = []
        input_tex_coords_list = []
        shape_feats_list = []
        shape_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            input_tex_feats_list.append(x_t.feats[begin:end])
            input_tex_feats_list.append(tex_slats.feats[begin:end])
            input_tex_coords_list.append(x_t.coords[begin:end])
            input_tex_coords_list.append(tex_slats.coords[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            begin = end
        x_t = sp.SparseTensor(torch.cat(input_tex_feats_list), torch.cat(input_tex_coords_list))
        shape_slats = sp.SparseTensor(torch.cat(shape_feats_list), torch.cat(shape_coords_list))
        output_tex_slats = self.flow_model(x_t, t, cond, shape_slats)
        output_tex_feats_list = []
        output_tex_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            output_tex_feats_list.append(output_tex_slats.feats[begin:end])
            output_tex_coords_list.append(output_tex_slats.coords[begin:end])
            begin = begin + 2 * coords_len
        return sp.SparseTensor(torch.cat(output_tex_feats_list), torch.cat(output_tex_coords_list))


# ─── Model loading ─────────────────────────────────────────────────────────────

def load_base_models():
    """Load TRELLIS backbone + auxiliary models (cached globally)."""
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
    """Load a segmentation model (cached per ckpt_path)."""
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


def build_sampler_params(pipeline_args, steps, rescale_t, guidance_strength,
                          guidance_rescale, guidance_interval_start, guidance_interval_end):
    params = dict(pipeline_args['tex_slat_sampler']['params'])
    params['steps'] = steps
    params['rescale_t'] = rescale_t
    params['guidance_strength'] = guidance_strength
    params['guidance_rescale'] = guidance_rescale
    params['guidance_interval'] = [guidance_interval_start, guidance_interval_end]
    return params


# ─── Inference functions ───────────────────────────────────────────────────────

def run_interactive(
    glb_path, ckpt_path, transforms_path, rendered_img,
    points_str,
    steps, rescale_t, guidance_strength, guidance_rescale,
    guidance_interval_start, guidance_interval_end,
    decimation_target, texture_size, remesh, remesh_band, remesh_project,
):
    base = load_base_models()
    gen3dseg = load_seg_model(ckpt_path, 'interactive')
    sampler = SamplerInteractive()

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

    print("Rendering conditioning image …")
    render_from_transforms(glb_path, transforms_path, img_path)
    if rendered_img is not None:
        img_path = rendered_img
    image = Image.open(img_path)
    image = preprocess_image(image)
    _to_cuda(base['image_cond_model'])
    cond = get_cond(base['image_cond_model'], [image])
    _offload(base['image_cond_model'])

    # Parse points
    flat = [int(v) for v in points_str.split()]
    if len(flat) % 3 != 0:
        raise ValueError("Points must be multiples of 3 (x y z per point).")
    input_vxz_points_list = [flat[i:i+3] for i in range(0, len(flat), 3)]

    print("Encoding click points …")
    vxz_points_coords = torch.tensor(input_vxz_points_list, dtype=torch.int32).cuda()
    vxz_points_coords = torch.cat(
        [torch.zeros((vxz_points_coords.shape[0], 1), dtype=torch.int32).cuda(), vxz_points_coords], dim=1)
    _to_cuda(base['tex_encoder'])
    input_points_coords = base['tex_encoder'](
        sp.SparseTensor(torch.zeros((vxz_points_coords.shape[0], 6), dtype=torch.float32).cuda(),
                        vxz_points_coords)).coords
    _offload(base['tex_encoder'])
    input_points_coords = torch.unique(input_points_coords, dim=0)
    point_num = input_points_coords.shape[0]
    if point_num >= 10:
        input_points_coords = input_points_coords[:10]
        point_labels = torch.tensor([[1]] * 10, dtype=torch.int32).cuda()
    else:
        input_points_coords = torch.cat(
            [input_points_coords, torch.zeros((10 - point_num, 4), dtype=torch.int32).cuda()], dim=0)
        point_labels = torch.tensor([[1]] * point_num + [[0]] * (10 - point_num), dtype=torch.int32).cuda()
    input_points = {'point_slats': sp.SparseTensor(input_points_coords, input_points_coords),
                    'point_labels': point_labels}

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
                                      input_points, coords_len_list, cond, sampler_params)
    _offload(gen3dseg)
    output_tex_slat = output_tex_slat * tex_std + tex_mean

    _to_cuda(base['tex_decoder'])
    with torch.no_grad():
        tex_voxels = base['tex_decoder'](output_tex_slat, guide_subs=subs) * 0.5 + 0.5
    _offload(base['tex_decoder'])

    print("Exporting GLB …")
    glb = slat_to_glb(meshes, tex_voxels, decimation_target=int(decimation_target),
                       texture_size=int(texture_size), remesh=remesh,
                       remesh_band=remesh_band, remesh_project=remesh_project)
    glb.export(out_path)
    return out_path


def run_full(
    glb_path, ckpt_path, transforms_path, rendered_img,
    steps, rescale_t, guidance_strength, guidance_rescale,
    guidance_interval_start, guidance_interval_end,
    decimation_target, texture_size, remesh, remesh_band, remesh_project,
):
    base = load_base_models()
    gen3dseg = load_seg_model(ckpt_path, 'full')
    sampler = SamplerFull()

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

    print("Rendering conditioning image …")
    render_from_transforms(glb_path, transforms_path, img_path)
    if rendered_img is not None:
        img_path = rendered_img
    image = Image.open(img_path)
    image = preprocess_image(image)
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
    glb = slat_to_glb(meshes, tex_voxels, decimation_target=int(decimation_target),
                       texture_size=int(texture_size), remesh=remesh,
                       remesh_band=remesh_band, remesh_project=remesh_project)
    glb.export(out_path)
    return out_path


def run_full_2d(
    glb_path, ckpt_path, guidance_img,
    steps, rescale_t, guidance_strength, guidance_rescale,
    guidance_interval_start, guidance_interval_end,
    decimation_target, texture_size, remesh, remesh_band, remesh_project,
):
    base = load_base_models()
    gen3dseg = load_seg_model(ckpt_path, 'full_2d')
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
    image = preprocess_image(image)
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
    glb = slat_to_glb(meshes, tex_voxels, decimation_target=int(decimation_target),
                       texture_size=int(texture_size), remesh=remesh,
                       remesh_band=remesh_band, remesh_project=remesh_project)
    glb.export(out_path)
    return out_path


# ─── Split helper ─────────────────────────────────────────────────────────────

def run_split(
    seg_glb_path,
    color_quant_step, palette_sample_pixels, palette_min_pixels,
    palette_max_colors, palette_merge_dist, samples_per_face, flip_v,
    uv_wrap_repeat, transition_conf_thresh, transition_prop_iters,
    transition_neighbor_min, small_component_action, small_component_min_faces,
    postprocess_iters, min_faces_per_part, bake_transforms,
):
    if seg_glb_path is None or not os.path.isfile(seg_glb_path):
        raise gr.Error("Run segmentation first — output GLB is missing.")
    out_dir = os.path.dirname(seg_glb_path)
    out_parts_glb = os.path.join(out_dir, "segmented_parts.glb")
    splitter.split_glb_by_texture_palette_rgb(
        in_glb_path=seg_glb_path,
        out_glb_path=out_parts_glb,
        min_faces_per_part=int(min_faces_per_part),
        bake_transforms=bool(bake_transforms),
        color_quant_step=int(color_quant_step),
        palette_sample_pixels=int(palette_sample_pixels),
        palette_min_pixels=int(palette_min_pixels),
        palette_max_colors=int(palette_max_colors),
        palette_merge_dist=int(palette_merge_dist),
        samples_per_face=int(samples_per_face),
        flip_v=bool(flip_v),
        uv_wrap_repeat=bool(uv_wrap_repeat),
        transition_conf_thresh=float(transition_conf_thresh),
        transition_prop_iters=int(transition_prop_iters),
        transition_neighbor_min=int(transition_neighbor_min),
        small_component_action=str(small_component_action),
        small_component_min_faces=int(small_component_min_faces),
        postprocess_iters=int(postprocess_iters),
        debug_print=True,
    )
    if not os.path.isfile(out_parts_glb):
        raise gr.Error("Split failed: output parts GLB not found.")
    return out_parts_glb


def _make_split_controls(prefix):
    """Return a dict of Gradio components for the splitter accordion."""
    with gr.Accordion("Split parameters", open=False):
        gr.Markdown("**Presets:**")
        with gr.Row():
            _spl_max = gr.Button("🔪 Max Parts", size="sm")
            _spl_bal = gr.Button("⚖️ Balanced",  size="sm", variant="primary")
            _spl_cln = gr.Button("✨ Cleanest",  size="sm")

        gr.Markdown("##### Color palette")
        color_quant_step = gr.Slider(1, 64, value=16, step=1, label="Color quantization step",
            info="Snaps RGB to a coarser grid before reading the palette. Lower = more parts. "
                 "→ 16 default · 1 for max parts · 32–64 for fewer, larger parts")
        palette_sample_pixels = gr.Number(value=2_000_000, precision=0, label="Palette sample pixels",
            info="Pixels sampled from the texture to discover part colors. "
                 "→ 2 000 000 default. Raise to 5 000 000+ only if small parts are missing on large textures.")
        palette_min_pixels = gr.Number(value=500, precision=0, label="Palette min pixels",
            info="Drop colors with fewer pixels than this (removes noise/compression artifacts). "
                 "→ 500 default · 1 for max parts · 2 000+ for aggressive cleanup")
        palette_max_colors = gr.Number(value=256, precision=0, label="Palette max colors",
            info="Hard cap on the number of detected parts. Raise if some parts are missing. "
                 "→ 256 default · 1024 for max parts on complex models")
        palette_merge_dist = gr.Number(value=32, precision=0, label="Palette merge distance",
            info="Merges two colors if their RGB distance is below this — fewer parts. "
                 "→ 32 default · 0 for max parts · 64–100 for fewer, cleaner parts")

        gr.Markdown("##### Face sampling")
        samples_per_face = gr.Dropdown(choices=[1, 4], value=4, label="Samples per face",
            info="UV samples per triangle. 4 = robust at part edges (recommended). "
                 "1 = faster on very dense meshes.")
        flip_v = gr.Checkbox(value=True, label="Flip V (glTF convention)",
            info="Required for standard GLB files. Disable only if parts look vertically mirrored.")
        uv_wrap_repeat = gr.Checkbox(value=True, label="UV wrap repeat",
            info="Handles UV coords outside [0,1] with wrapping. Keep enabled for SegviGen output.")

        gr.Markdown("##### Boundary refinement")
        transition_conf_thresh = gr.Slider(0.25, 1.0, value=1.0, step=0.25,
            label="Transition confidence threshold",
            info="At 1.0, disabled (recommended). Lower to 0.5–0.75 to smooth noisy boundaries. "
                 "Lower values = fewer, larger parts.")
        transition_prop_iters = gr.Number(value=6, precision=0, label="Transition propagation iterations",
            info="Passes when threshold < 1.0. No effect at 1.0. → 6 default")
        transition_neighbor_min = gr.Number(value=1, precision=0, label="Transition neighbor minimum",
            info="Agreeing neighbors needed to relabel a face. Active only when threshold < 1.0. → 1 default")

        gr.Markdown("##### Small component cleanup")
        small_component_action = gr.Dropdown(choices=["reassign", "drop"], value="reassign",
            label="Small component action",
            info="What to do with tiny isolated face groups. "
                 "Reassign = merge into nearest neighbor · Drop = remove entirely")
        small_component_min_faces = gr.Number(value=50, precision=0, label="Small component min faces",
            info="Groups smaller than this are treated as noise. "
                 "→ 50 default · 1 for max parts · 100–200 for cleaner output · 500+ very aggressive")
        postprocess_iters = gr.Number(value=3, precision=0, label="Post-process iterations",
            info="Extra smoothing passes after splitting. "
                 "→ 3 default · 0 for max parts · 5–8 for cleaner output · 10+ very smooth")

        gr.Markdown("##### Output")
        min_faces_per_part = gr.Number(value=1, precision=0, label="Min faces per part",
            info="Parts with fewer faces are dropped from the export. "
                 "→ 1 default (keep all) · 50–100 as a final cleanup filter")
        bake_transforms = gr.Checkbox(value=True, label="Bake transforms",
            info="Applies scene-graph transforms to vertex positions for consistent world-space coords. "
                 "Keep enabled for all standard workflows.")

    _spl_outs = [color_quant_step, palette_sample_pixels, palette_min_pixels,
                 palette_max_colors, palette_merge_dist, samples_per_face,
                 flip_v, uv_wrap_repeat, transition_conf_thresh, transition_prop_iters,
                 transition_neighbor_min, small_component_action, small_component_min_faces,
                 postprocess_iters, min_faces_per_part, bake_transforms]
    _spl_max.click(fn=lambda: _split_preset_vals("max_parts"), outputs=_spl_outs)
    _spl_bal.click(fn=lambda: _split_preset_vals("balanced"),  outputs=_spl_outs)
    _spl_cln.click(fn=lambda: _split_preset_vals("cleanest"),  outputs=_spl_outs)

    return dict(
        color_quant_step=color_quant_step,
        palette_sample_pixels=palette_sample_pixels,
        palette_min_pixels=palette_min_pixels,
        palette_max_colors=palette_max_colors,
        palette_merge_dist=palette_merge_dist,
        samples_per_face=samples_per_face,
        flip_v=flip_v,
        uv_wrap_repeat=uv_wrap_repeat,
        transition_conf_thresh=transition_conf_thresh,
        transition_prop_iters=transition_prop_iters,
        transition_neighbor_min=transition_neighbor_min,
        small_component_action=small_component_action,
        small_component_min_faces=small_component_min_faces,
        postprocess_iters=postprocess_iters,
        min_faces_per_part=min_faces_per_part,
        bake_transforms=bake_transforms,
    )


# ─── Preset values ────────────────────────────────────────────────────────────

_SAMPLER_PRESETS = {
    "fast":     dict(steps=12,  rescale_t=1.0, guidance=7.5, guidance_rescale=0.0,
                     gi_start=0.0, gi_end=1.0, decimation=50_000,  tex_size=512,
                     remesh=True, remesh_band=1, remesh_proj=0),
    "balanced": dict(steps=25,  rescale_t=1.0, guidance=7.5, guidance_rescale=0.0,
                     gi_start=0.0, gi_end=1.0, decimation=100_000, tex_size=1024,
                     remesh=True, remesh_band=1, remesh_proj=0),
    "quality":  dict(steps=50,  rescale_t=1.5, guidance=7.5, guidance_rescale=0.0,
                     gi_start=0.1, gi_end=0.9, decimation=300_000, tex_size=2048,
                     remesh=True, remesh_band=0, remesh_proj=1),
}

_SPLIT_PRESETS = {
    "max_parts": dict(color_quant_step=1,  palette_sample_pixels=2_000_000,
                      palette_min_pixels=1,    palette_max_colors=1024, palette_merge_dist=0,
                      samples_per_face=4, flip_v=True, uv_wrap_repeat=True,
                      transition_conf_thresh=1.0, transition_prop_iters=6, transition_neighbor_min=1,
                      small_component_action="reassign", small_component_min_faces=1,
                      postprocess_iters=0, min_faces_per_part=1, bake_transforms=True),
    "balanced":  dict(color_quant_step=16, palette_sample_pixels=2_000_000,
                      palette_min_pixels=500,  palette_max_colors=256,  palette_merge_dist=32,
                      samples_per_face=4, flip_v=True, uv_wrap_repeat=True,
                      transition_conf_thresh=1.0, transition_prop_iters=6, transition_neighbor_min=1,
                      small_component_action="reassign", small_component_min_faces=50,
                      postprocess_iters=3, min_faces_per_part=1, bake_transforms=True),
    "cleanest":  dict(color_quant_step=32, palette_sample_pixels=2_000_000,
                      palette_min_pixels=2000, palette_max_colors=128,  palette_merge_dist=64,
                      samples_per_face=4, flip_v=True, uv_wrap_repeat=True,
                      transition_conf_thresh=1.0, transition_prop_iters=6, transition_neighbor_min=1,
                      small_component_action="reassign", small_component_min_faces=200,
                      postprocess_iters=8, min_faces_per_part=50, bake_transforms=True),
}


def _sampler_preset_vals(name):
    p = _SAMPLER_PRESETS[name]
    return (p["steps"], p["rescale_t"], p["guidance"], p["guidance_rescale"],
            p["gi_start"], p["gi_end"], p["decimation"], p["tex_size"],
            p["remesh"], p["remesh_band"], p["remesh_proj"])


def _split_preset_vals(name):
    p = _SPLIT_PRESETS[name]
    return (p["color_quant_step"], p["palette_sample_pixels"], p["palette_min_pixels"],
            p["palette_max_colors"], p["palette_merge_dist"], p["samples_per_face"],
            p["flip_v"], p["uv_wrap_repeat"], p["transition_conf_thresh"],
            p["transition_prop_iters"], p["transition_neighbor_min"],
            p["small_component_action"], p["small_component_min_faces"],
            p["postprocess_iters"], p["min_faces_per_part"], p["bake_transforms"])


# ─── Sampler / export controls helper ─────────────────────────────────────────

def _make_sampler_export_controls(default_steps=25, default_guidance=7.5):
    """Render sampler + export parameter widgets inside an accordion."""
    with gr.Accordion("Sampler & export parameters", open=False):
        gr.Markdown("**Presets:**")
        with gr.Row():
            _sp_fast = gr.Button("⚡ Fast", size="sm")
            _sp_bal  = gr.Button("⚖️ Balanced", size="sm", variant="primary")
            _sp_qual = gr.Button("✨ Quality", size="sm")

        gr.Markdown("##### Sampler")
        steps = gr.Slider(1, 100, value=default_steps, step=1, label="Steps",
            info="How many denoising passes the model runs. "
                 "More = better quality, slower. → Fast: 12 · Balanced: 25 · Quality: 50")
        rescale_t = gr.Slider(0.1, 5.0, value=1.0, step=0.05, label="Rescale T",
            info="Warps the denoising schedule. >1 spends more steps on fine detail at the end. "
                 "→ Keep at 1.0. Try 1.5–2.0 if part boundaries look blurry.")
        guidance = gr.Slider(0.0, 10.0, value=default_guidance, step=0.1,
            label="Guidance strength (CFG)",
            info="How strictly the model follows the input image. Higher = crisper parts, "
                 "but can over-saturate. → 7.5 default. Lower to 4–6 if over-segmenting. "
                 "Raise to 9–10 if parts are merging. Use 5–7 for Interactive mode.")
        guidance_rescale = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
            label="Guidance rescale",
            info="Prevents color blowout when CFG is high. "
                 "→ Keep at 0. Enable (0.5–0.7) only if colors look washed out at CFG > 8.")
        gi_start = gr.Slider(0.0, 1.0, value=0.0, step=0.01,
            label="Guidance interval — start",
            info="CFG only applies between [start, end] of the denoising trajectory. "
                 "→ Keep at 0.0. Try 0.1 if artifacts appear on complex geometry.")
        gi_end = gr.Slider(0.0, 1.0, value=1.0, step=0.01,
            label="Guidance interval — end",
            info="Upper bound of the CFG window. "
                 "→ Keep at 1.0. Try 0.9 together with start=0.1 for softer boundaries.")

        gr.Markdown("##### Export")
        decimation = gr.Number(value=100000, label="Decimation target (faces)",
            info="Max faces in the output mesh after simplification. "
                 "→ 100k default · 300k+ if you plan to split into parts · 30–50k for lightweight")
        tex_size = gr.Dropdown([512, 1024, 2048, 4096], value=1024,
            label="Texture size (px)",
            info="Resolution of the baked texture. Higher = sharper part color boundaries. "
                 "→ 1024 default · 2048 if splitting · 4096 for hero assets · 512 for quick tests")
        remesh = gr.Checkbox(value=True, label="Remesh",
            info="Rebalances triangle sizes before baking, improving texture quality. "
                 "→ Keep on. Disable only if preserving exact mesh topology for rigging.")
        remesh_band = gr.Slider(0, 4, value=1, step=1, label="Remesh band",
            info="Remesh coarseness: 0 = finest, 4 = coarsest. "
                 "→ 1 default · 0 for detailed models · 2–3 for smooth organic shapes")
        remesh_proj = gr.Slider(0, 4, value=0, step=1, label="Remesh project",
            info="Snaps the remeshed surface back onto the original shape. "
                 "→ 0 default · 1–2 for mechanical parts · 3–4 for curved organic shapes")

    _sp_outs = [steps, rescale_t, guidance, guidance_rescale, gi_start, gi_end,
                decimation, tex_size, remesh, remesh_band, remesh_proj]
    _sp_fast.click(fn=lambda: _sampler_preset_vals("fast"),     outputs=_sp_outs)
    _sp_bal .click(fn=lambda: _sampler_preset_vals("balanced"), outputs=_sp_outs)
    _sp_qual.click(fn=lambda: _sampler_preset_vals("quality"),  outputs=_sp_outs)

    return dict(
        steps=steps, rescale_t=rescale_t, guidance=guidance,
        guidance_rescale=guidance_rescale, gi_start=gi_start, gi_end=gi_end,
        decimation=decimation, tex_size=tex_size, remesh=remesh,
        remesh_band=remesh_band, remesh_proj=remesh_proj,
    )


# ─── Gradio UI ────────────────────────────────────────────────────────────────

_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")

def _ckpt(name):
    p = os.path.join(_CKPT_DIR, name)
    return p if os.path.isfile(p) else ""

DEFAULT_TRANSFORMS   = os.path.abspath("data_toolkit/transforms.json")
DEFAULT_CKPT_INTER   = _ckpt("interactive_seg.ckpt")
DEFAULT_CKPT_FULL    = _ckpt("full_seg.ckpt")
DEFAULT_CKPT_FULL_2D = _ckpt("full_seg_w_2d_map.ckpt")

with gr.Blocks(title="SegviGen — 3D Part Segmentation") as demo:
    gr.Markdown("# SegviGen — 3D Part Segmentation")
    gr.Markdown(
        "Upload a 3D model (GLB), choose a segmentation method, tune the parameters, and run inference."
    )

    gr.Markdown("## Input Model")
    input_model = gr.Model3D(label="Upload GLB / OBJ / PLY", clear_color=[0.1, 0.1, 0.15, 1])

    # ── Method tabs ──────────────────────────────────────────────────────────
    with gr.Tabs():

        # ────────────────────────────────────────────────────────────────────
        # TAB 1 — Interactive
        # ────────────────────────────────────────────────────────────────────
        with gr.Tab("Interactive Part Segmentation"):
            gr.Markdown(
                "### Click-based segmentation\n"
                "Specify 3-D voxel coordinates (in the 0–511 grid) of the part you want to isolate."
            )
            with gr.Row():
                i_ckpt = gr.Textbox(label="Checkpoint path (.ckpt)", value=DEFAULT_CKPT_INTER, placeholder="path/to/interactive_seg.ckpt")
                i_transforms = gr.Textbox(label="Transforms JSON", value=DEFAULT_TRANSFORMS,
                                          placeholder="data_toolkit/transforms.json")
                i_rendered_img = gr.Image(label="Override rendered image (optional, PNG)",
                                          type="filepath", value=None)
                i_points = gr.Textbox(
                    label="Voxel click points  (x y z per point, space-separated; up to 10 points)",
                    placeholder="388 448 392   256 256 256",
                    value="388 448 392",
                )

            i_run = gr.Button("Run Interactive Segmentation", variant="primary")

            # ── Viewers ──────────────────────────────────────────────────────
            with gr.Row():
                i_seg_model   = gr.Model3D(label="Segmented Output",   clear_color=[0.1, 0.1, 0.15, 1])
                i_parts_model = gr.Model3D(label="Split Parts Output", clear_color=[0.1, 0.1, 0.15, 1])

            # ── Parameters (side by side) ─────────────────────────────────
            with gr.Row():
                with gr.Column():
                    _i = _make_sampler_export_controls()
                    i_steps, i_rescale_t, i_guidance, i_guidance_rescale, i_gi_start, i_gi_end = (
                        _i["steps"], _i["rescale_t"], _i["guidance"], _i["guidance_rescale"], _i["gi_start"], _i["gi_end"])
                    i_decimation, i_tex_size, i_remesh, i_remesh_band, i_remesh_proj = (
                        _i["decimation"], _i["tex_size"], _i["remesh"], _i["remesh_band"], _i["remesh_proj"])
                with gr.Column():
                    i_split_ctrl = _make_split_controls("i")
                    i_split_btn = gr.Button("Split into Parts", variant="secondary")

            i_seg_state = gr.State(None)

            def _run_interactive_tab(*args):
                path = run_interactive(*args)
                return path, path

            i_run.click(
                fn=_run_interactive_tab,
                inputs=[
                    input_model, i_ckpt, i_transforms, i_rendered_img,
                    i_points,
                    i_steps, i_rescale_t, i_guidance, i_guidance_rescale,
                    i_gi_start, i_gi_end,
                    i_decimation, i_tex_size, i_remesh, i_remesh_band, i_remesh_proj,
                ],
                outputs=[i_seg_model, i_seg_state],
            )

            i_split_btn.click(
                fn=run_split,
                inputs=[i_seg_state] + list(i_split_ctrl.values()),
                outputs=i_parts_model,
            )

        # ────────────────────────────────────────────────────────────────────
        # TAB 2 — Full Segmentation
        # ────────────────────────────────────────────────────────────────────
        with gr.Tab("Full Segmentation"):
            gr.Markdown(
                "### Automatic full-part segmentation\n"
                "The model segments all parts simultaneously, conditioned on a rendered view of the input model."
            )
            with gr.Row():
                f_ckpt = gr.Textbox(label="Checkpoint path (.ckpt)", value=DEFAULT_CKPT_FULL, placeholder="path/to/full_seg.ckpt")
                f_transforms = gr.Textbox(label="Transforms JSON", value=DEFAULT_TRANSFORMS,
                                          placeholder="data_toolkit/transforms.json")
                f_rendered_img = gr.Image(label="Override rendered image (optional, PNG)",
                                         type="filepath", value=None)

            f_run = gr.Button("Run Full Segmentation", variant="primary")

            # ── Viewers ──────────────────────────────────────────────────────
            with gr.Row():
                f_seg_model   = gr.Model3D(label="Segmented Output",   clear_color=[0.1, 0.1, 0.15, 1])
                f_parts_model = gr.Model3D(label="Split Parts Output", clear_color=[0.1, 0.1, 0.15, 1])

            # ── Parameters (side by side) ─────────────────────────────────
            with gr.Row():
                with gr.Column():
                    _f = _make_sampler_export_controls()
                    f_steps, f_rescale_t, f_guidance, f_guidance_rescale, f_gi_start, f_gi_end = (
                        _f["steps"], _f["rescale_t"], _f["guidance"], _f["guidance_rescale"], _f["gi_start"], _f["gi_end"])
                    f_decimation, f_tex_size, f_remesh, f_remesh_band, f_remesh_proj = (
                        _f["decimation"], _f["tex_size"], _f["remesh"], _f["remesh_band"], _f["remesh_proj"])
                with gr.Column():
                    f_split_ctrl = _make_split_controls("f")
                    f_split_btn = gr.Button("Split into Parts", variant="secondary")

            f_seg_state = gr.State(None)

            def _run_full_tab(*args):
                path = run_full(*args)
                return path, path

            f_run.click(
                fn=_run_full_tab,
                inputs=[
                    input_model, f_ckpt, f_transforms, f_rendered_img,
                    f_steps, f_rescale_t, f_guidance, f_guidance_rescale,
                    f_gi_start, f_gi_end,
                    f_decimation, f_tex_size, f_remesh, f_remesh_band, f_remesh_proj,
                ],
                outputs=[f_seg_model, f_seg_state],
            )

            f_split_btn.click(
                fn=run_split,
                inputs=[f_seg_state] + list(f_split_ctrl.values()),
                outputs=f_parts_model,
            )

        # ────────────────────────────────────────────────────────────────────
        # TAB 3 — Full Segmentation + 2D Guidance Map
        # ────────────────────────────────────────────────────────────────────
        with gr.Tab("Full Segmentation + 2D Guidance Map"):
            gr.Markdown(
                "### 2D-guided full segmentation\n"
                "Upload a 2D semantic map image (different solid colors per part) to control segmentation granularity."
            )
            with gr.Row():
                t_ckpt = gr.Textbox(label="Checkpoint path (.ckpt)",
                                    value=DEFAULT_CKPT_FULL_2D, placeholder="path/to/full_seg_w_2d_map.ckpt")
                t_guidance_img = gr.Image(label="2D Guidance Map (PNG — unique color per part)",
                                          type="filepath")

            t_run = gr.Button("Run 2D-Guided Segmentation", variant="primary")

            # ── Viewers ──────────────────────────────────────────────────────
            with gr.Row():
                t_seg_model   = gr.Model3D(label="Segmented Output",   clear_color=[0.1, 0.1, 0.15, 1])
                t_parts_model = gr.Model3D(label="Split Parts Output", clear_color=[0.1, 0.1, 0.15, 1])

            # ── Parameters (side by side) ─────────────────────────────────
            with gr.Row():
                with gr.Column():
                    _t = _make_sampler_export_controls()
                    t_steps, t_rescale_t, t_guidance, t_guidance_rescale, t_gi_start, t_gi_end = (
                        _t["steps"], _t["rescale_t"], _t["guidance"], _t["guidance_rescale"], _t["gi_start"], _t["gi_end"])
                    t_decimation, t_tex_size, t_remesh, t_remesh_band, t_remesh_proj = (
                        _t["decimation"], _t["tex_size"], _t["remesh"], _t["remesh_band"], _t["remesh_proj"])
                with gr.Column():
                    t_split_ctrl = _make_split_controls("t")
                    t_split_btn = gr.Button("Split into Parts", variant="secondary")

            t_seg_state = gr.State(None)

            def _run_full_2d_tab(*args):
                path = run_full_2d(*args)
                return path, path

            t_run.click(
                fn=_run_full_2d_tab,
                inputs=[
                    input_model, t_ckpt, t_guidance_img,
                    t_steps, t_rescale_t, t_guidance, t_guidance_rescale,
                    t_gi_start, t_gi_end,
                    t_decimation, t_tex_size, t_remesh, t_remesh_band, t_remesh_proj,
                ],
                outputs=[t_seg_model, t_seg_state],
            )

            t_split_btn.click(
                fn=run_split,
                inputs=[t_seg_state] + list(t_split_ctrl.values()),
                outputs=t_parts_model,
            )

        # ────────────────────────────────────────────────────────────────────
        # TAB 4 — Prepare 2D Guidance Map
        # ────────────────────────────────────────────────────────────────────
        with gr.Tab("Prepare 2D Guidance Map"):
            gr.Markdown(
                "### Prepare a 2D guidance map for Tab 3\n"
                "Generate a flat-color segmented image from your 3D model. "
                "Each part gets a unique solid color — use the output as the "
                "**2D Guidance Map** input in the *Full Segmentation + 2D Guidance Map* tab."
            )

            gmap_method = gr.Radio(
                choices=["Pixmesh 2D render"],
                value="Pixmesh 2D render",
                label="Method",
                info="More methods can be added here.",
            )

            # ── Pixmesh 2D render controls ───────────────────────────────
            with gr.Column(visible=True) as _pixmesh_col:
                gr.Markdown(
                    "**How it works:** Renders one or more views of your model → "
                    "sends them to a VLM to identify parts → assigns each part a unique "
                    "solid color → asks an image-gen model to flood-fill each view → "
                    "outputs the flat-color image (or grid) as your guidance map."
                )
                with gr.Row():
                    gmap_glb = gr.Textbox(
                        label="GLB path",
                        placeholder="Path to your .glb file (or use the Input Model above)",
                        info="If left empty, the Input Model above is used.",
                    )
                    gmap_transforms = gr.Textbox(
                        label="Transforms JSON",
                        value=DEFAULT_TRANSFORMS,
                        placeholder="data_toolkit/transforms.json",
                        info="Used only in Single-view mode for the main camera position.",
                    )
                with gr.Row():
                    gmap_gemini_key = gr.Textbox(
                        label="Gemini API key",
                        type="password",
                        placeholder="AIza…",
                        info="Required. Get one at aistudio.google.com/apikey",
                    )
                    gmap_resolution = gr.Slider(
                        256, 1024, value=512, step=128,
                        label="Render resolution (px)",
                        info="Resolution per view tile. Grid output = cols × resolution wide.",
                    )

                # ── View mode ─────────────────────────────────────────────
                gmap_mode = gr.Radio(
                    choices=["Single view", "Multi-view grid"],
                    value="Single view",
                    label="View mode",
                    info="Single: one isometric view from transforms.json. "
                         "Grid: 2–6 canonical views assembled into a grid.",
                )

                with gr.Column(visible=False) as _grid_col:
                    gr.Markdown(
                        "**Grid mode:** renders each selected view separately, generates "
                        "a segmented image per view in parallel, then stitches them into "
                        "a single flat-color grid PNG to feed into Tab 3."
                    )
                    with gr.Row():
                        gmap_view_front  = gr.Checkbox(value=True,  label="Front")
                        gmap_view_back   = gr.Checkbox(value=True,  label="Back")
                        gmap_view_left   = gr.Checkbox(value=True,  label="Left")
                        gmap_view_right  = gr.Checkbox(value=True,  label="Right")
                        gmap_view_top    = gr.Checkbox(value=False, label="Top")
                        gmap_view_bottom = gr.Checkbox(value=False, label="Bottom")
                    gmap_grid_cols = gr.Slider(
                        1, 3, value=2, step=1,
                        label="Grid columns",
                        info="Number of columns in the stitched output grid.",
                    )

                # Show/hide grid controls when mode changes
                gmap_mode.change(
                    fn=lambda m: gr.update(visible=(m == "Multi-view grid")),
                    inputs=gmap_mode,
                    outputs=_grid_col,
                )

                with gr.Accordion("Model selection", open=False):
                    gmap_analyze_model = gr.Dropdown(
                        choices=[
                            "gemini-2.5-flash",
                            "gemini-2.5-pro",
                            "gemini-3-flash-preview",
                            "gemini-3-pro-preview",
                            "gemini-3.1-pro-preview",
                            "claude-sonnet-4-6",
                            "claude-opus-4-6",
                            "claude-haiku-4-5",
                            "gpt-4o",
                            "gpt-5-mini",
                            "gpt-5.2",
                        ],
                        value="gemini-2.5-flash",
                        label="Analyze model (describe step)",
                        info="VLM used to identify parts. Pro/Opus = more detail; Flash/Haiku = faster.",
                    )
                    gmap_generate_model = gr.Dropdown(
                        choices=[
                            "gemini-3-pro-image-preview",
                            "gemini-3-pro-preview",
                        ],
                        value="gemini-3-pro-image-preview",
                        label="Generate model (segmentation step)",
                        info="Image-generation model used to flood-fill the parts.",
                    )

            gmap_run = gr.Button("Generate 2D Guidance Map", variant="primary")

            with gr.Row():
                gmap_output = gr.Image(
                    label="Generated guidance map",
                    type="filepath",
                    interactive=False,
                )
                with gr.Column():
                    gr.Markdown("**Assembly tree** (identified parts)")
                    gmap_json_out = gr.JSON(label=None)

            gmap_use_btn = gr.Button(
                "→ Use this map as guidance input in Tab 3",
                variant="secondary",
            )

            def _run_pixmesh(
                glb_input, glb_override, transforms, key,
                analyze_model, generate_model, resolution,
                mode,
                v_front, v_back, v_left, v_right, v_top, v_bottom,
                grid_cols,
            ):
                path = glb_override.strip() if glb_override and glb_override.strip() else glb_input
                if not path:
                    raise gr.Error("No GLB path — upload a model or enter a path.")

                selected_views = tuple(
                    v for v, checked in zip(
                        gmap.CANONICAL_VIEW_NAMES,
                        [v_front, v_back, v_left, v_right, v_top, v_bottom],
                    ) if checked
                )
                if mode == "Multi-view grid" and not selected_views:
                    raise gr.Error("Select at least one view for grid mode.")

                out_path, description = gmap.run_pixmesh(
                    glb_path=path,
                    transforms_path=transforms,
                    gemini_api_key=key,
                    analyze_model=analyze_model,
                    generate_model=generate_model,
                    resolution=int(resolution),
                    mode="grid" if mode == "Multi-view grid" else "single",
                    grid_views=selected_views,
                    grid_cols=int(grid_cols),
                )
                return out_path, out_path, description

            _gmap_img_state = gr.State(None)

            gmap_run.click(
                fn=_run_pixmesh,
                inputs=[
                    input_model, gmap_glb, gmap_transforms,
                    gmap_gemini_key, gmap_analyze_model, gmap_generate_model,
                    gmap_resolution,
                    gmap_mode,
                    gmap_view_front, gmap_view_back, gmap_view_left,
                    gmap_view_right, gmap_view_top, gmap_view_bottom,
                    gmap_grid_cols,
                ],
                outputs=[gmap_output, _gmap_img_state, gmap_json_out],
            )

            gmap_use_btn.click(
                fn=lambda p: p,
                inputs=_gmap_img_state,
                outputs=t_guidance_img,
            )

    gr.Markdown(
        "---\n"
        "**Tip:** Base models (TRELLIS.2-4B) are loaded once on first run and cached. "
        "Segmentation checkpoints are also cached per path. "
        "Intermediate VXZ files are written to a system temp directory and cleaned up automatically."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
