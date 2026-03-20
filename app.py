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
from trellis2.pipelines.rembg import BiRefNet
from trellis2.modules.utils import manual_cast
from trellis2.representations import MeshWithVoxel
from data_toolkit.bpy_render import render_from_transforms
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

import gradio as gr
import split as splitter

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


def preprocess_image(rembg_model, input):
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
        output = rembg_model(input)
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
    rembg_model = BiRefNet(model_name="briaai/RMBG-2.0")
    image_cond_model = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")

    pipeline_json_path = hf_hub_download(repo_id="microsoft/TRELLIS.2-4B", filename="pipeline.json")
    with open(pipeline_json_path, "r") as f:
        pipeline_config = json.load(f)
    pipeline_args = pipeline_config['args']

    base = {
        'shape_encoder': shape_encoder,
        'tex_encoder': tex_encoder,
        'shape_decoder': shape_decoder,
        'tex_decoder': tex_decoder,
        'rembg_model': rembg_model,
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
    _to_cuda(base['rembg_model'])
    image = preprocess_image(base['rembg_model'], image)
    _offload(base['rembg_model'])
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
    _to_cuda(base['rembg_model'])
    image = preprocess_image(base['rembg_model'], image)
    _offload(base['rembg_model'])
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
    _to_cuda(base['rembg_model'])
    image = preprocess_image(base['rembg_model'], image)
    _offload(base['rembg_model'])
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
    with gr.Accordion("Advanced segmentation (split) options", open=False):

        gr.Markdown(
            "**How splitting works:** SegviGen encodes part labels as distinct solid colors in "
            "the output texture (the paper calls this *part-indicative colorization*). "
            "Splitting reads that texture, groups faces by color, and exports one mesh object "
            "per color group. These parameters control palette detection and cleanup.\n\n"
            "**Presets:** "
            "Most parts → step=1, min px=1, max colors=1024, merge dist=0, min faces=1, iters=0 | "
            "Balanced (default) → step=16, min px=500, merge dist=32, min faces=50, iters=3 | "
            "Fewest/cleanest → step=32, min px=2000, merge dist=64, min faces=200, iters=8"
        )

        gr.Markdown("##### Color palette")
        color_quant_step = gr.Slider(1, 64, value=16, step=1, label="Color quantization step",
            info="Snaps each RGB channel to a grid before building the palette. "
                 "SegviGen assigns solid colors to parts during training, but texture compression "
                 "and anti-aliasing introduce slight variation within each color. "
                 "→ Recommended: 16 (default) — handles compression artifacts while keeping "
                 "distinct part colors separate. "
                 "Use 1–4 to maximize the number of parts (every shade is its own entry). "
                 "Use 32–64 to merge similar hues and produce fewer, larger parts.")
        palette_sample_pixels = gr.Number(value=2_000_000, precision=0, label="Palette sample pixels",
            info="Pixels randomly sampled from the texture to discover the color palette. "
                 "→ Recommended: 2 000 000 (default) — covers even 4K textures reliably. "
                 "Raise to 5 000 000+ only if small parts on very large textures are being missed. "
                 "Lowering speeds things up but risks dropping rare part colors from the palette.")
        palette_min_pixels = gr.Number(value=500, precision=0, label="Palette min pixels",
            info="Minimum pixel count for a color to be kept in the palette. "
                 "Filters out anti-aliasing blends and JPEG artifacts at part edges. "
                 "→ Recommended: 500 (default) for balanced results. "
                 "Set to 1 for maximum parts (every color survives, including noise). "
                 "Raise to 1 000–5 000 if you see many tiny noise fragments in the split output. "
                 "Raise to 10 000+ for very aggressive cleanup on blurry or compressed textures.")
        palette_max_colors = gr.Number(value=256, precision=0, label="Palette max colors",
            info="Hard cap on palette entries after ranking by pixel count. "
                 "SegviGen can produce many distinct part colors especially in Full mode. "
                 "→ Recommended: 256 (default) covers most models. "
                 "Raise to 512 or 1024 if you run Full segmentation on a complex mesh with "
                 "many semantic parts and some are being dropped. "
                 "Lowering forces fewer parts regardless of other settings.")
        palette_merge_dist = gr.Number(value=32, precision=0, label="Palette merge distance",
            info="Merges two palette entries if their Euclidean RGB distance is below this. "
                 "Collapses near-duplicate colors from texture compression or slight shading "
                 "variation within one semantic region. "
                 "→ Recommended: 32 (default) — collapses duplicates without touching "
                 "semantically distinct colors. "
                 "Set to 0 to disable (more parts, noisier). "
                 "Raise to 64 to merge visually similar but technically different colors. "
                 "Raise to 100+ to dramatically reduce the part count.")

        gr.Markdown("##### Face sampling")
        samples_per_face = gr.Dropdown(choices=[1, 4], value=4, label="Samples per face",
            info="UV sample points per triangle for color label voting. "
                 "→ Recommended: 4 (default) — samples centroid + 3 near-vertex points, "
                 "much more robust at seams and part boundaries. "
                 "Use 1 (centroid only) only if splitting is too slow on very dense meshes.")
        flip_v = gr.Checkbox(value=True, label="Flip V (glTF convention)",
            info="glTF stores UV with V=0 at the top; most image libraries use V=0 at the bottom. "
                 "→ Recommended: enabled (default) for all GLB files exported by SegviGen. "
                 "Disable only if parts appear vertically mirrored in the split output.")
        uv_wrap_repeat = gr.Checkbox(value=True, label="UV wrap repeat",
            info="How out-of-range UV coordinates (outside [0,1]) are handled. "
                 "→ Recommended: enabled (default, mod-1 wrapping) — correct for SegviGen's "
                 "output textures which are not tiled but may have UVs slightly outside range. "
                 "Disable (clamp) only if you see incorrect colors on mesh borders.")

        gr.Markdown("##### Boundary refinement")
        transition_conf_thresh = gr.Slider(0.25, 1.0, value=1.0, step=0.25,
            label="Transition confidence threshold",
            info="Confidence required before a face's label is changed during boundary propagation. "
                 "At 1.0 this pass is fully disabled — original assignments are kept as-is. "
                 "→ Recommended: 1.0 (default) — SegviGen's colorization is already clean; "
                 "boundary propagation is rarely needed. "
                 "Try 0.75 only if you see many single-face misclassifications at part edges. "
                 "Use 0.25–0.5 for aggressive boundary smoothing (fewer, larger parts).")
        transition_prop_iters = gr.Number(value=6, precision=0, label="Transition propagation iterations",
            info="How many propagation passes run when threshold < 1.0. Has no effect at 1.0. "
                 "→ Recommended: leave at 6 (default). If you lower the threshold, "
                 "more iterations spread corrected labels further inward from boundaries.")
        transition_neighbor_min = gr.Number(value=1, precision=0, label="Transition neighbor minimum",
            info="Minimum agreeing physical neighbors required to relabel a face during propagation. "
                 "Only active when threshold < 1.0. "
                 "→ Recommended: 1 (default). Raise to 2–3 for more conservative relabelling "
                 "(fewer changes, sharper boundaries) if the propagation over-smooths.")

        gr.Markdown("##### Small component cleanup")
        small_component_action = gr.Dropdown(choices=["reassign", "drop"], value="reassign",
            label="Small component action",
            info="What to do with connected face groups smaller than the threshold below. "
                 "→ Recommended: reassign (default) — absorbs fragments into the neighboring "
                 "part with the most shared edges. Keeps all faces in the output. "
                 "Use drop when you want clean isolated parts with no noise fragments, "
                 "and don't mind losing a few faces at part edges.")
        small_component_min_faces = gr.Number(value=50, precision=0, label="Small component min faces",
            info="Connected regions with fewer faces than this are treated as noise fragments. "
                 "→ Recommended: 50 (default) cleans up texture-boundary specks. "
                 "Set to 1 to disable all cleanup (maximum parts, noisiest output). "
                 "Raise to 100–200 for typical full-segmentation results. "
                 "Raise to 500+ for aggressive cleanup that merges all small islands.")
        postprocess_iters = gr.Number(value=3, precision=0, label="Post-process iterations",
            info="Topology-smoothing passes: each pass finds same-color components smaller "
                 "than the min-faces threshold and absorbs them into larger neighbors. "
                 "→ Recommended: 3 (default) for light cleanup. "
                 "Set to 0 for maximum parts with no post-processing. "
                 "Use 5–8 for moderately clean output on complex meshes. "
                 "Use 10+ for heavily smoothed output with fewer, larger parts.")

        gr.Markdown("##### Output")
        min_faces_per_part = gr.Number(value=1, precision=0, label="Min faces per part",
            info="Parts with fewer faces than this are dropped from the exported GLB. "
                 "→ Recommended: 1 (default) — keep everything. "
                 "Raise to 50–100 as a final safety filter after post-processing "
                 "to drop any remaining micro-fragments without affecting the main parts. "
                 "Raise to 500–1000 for game-ready exports where tiny parts are unwanted.")
        bake_transforms = gr.Checkbox(value=True, label="Bake transforms",
            info="Bakes each node's scene-graph transform into vertex positions before splitting, "
                 "so all output parts share one consistent world-space coordinate system. "
                 "→ Recommended: enabled (default) for all standard workflows. "
                 "Disable only if you need to preserve the original node hierarchy for "
                 "downstream tools that re-apply transforms (e.g. game engines with "
                 "prefab-based rigs).")

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


# ─── Sampler / export controls helper ─────────────────────────────────────────

def _make_sampler_export_controls(default_steps=25, default_guidance=7.5):
    """Render sampler + export parameter widgets with full descriptions."""
    gr.Markdown("#### Sampler parameters")
    steps = gr.Slider(1, 100, value=default_steps, step=1, label="Steps",
        info="Number of flow-matching denoising steps. SegviGen inherits "
             "TRELLIS.2's sampler: each step refines the part-color prediction "
             "from pure noise toward a clean colorization. "
             "→ Recommended: 25 (default, good balance). "
             "Use 12–15 for fast iteration when exploring parameters. "
             "Use 50 for high-quality final exports on complex or detailed meshes. "
             "Going above 50 rarely improves results further.")
    rescale_t = gr.Slider(0.1, 5.0, value=1.0, step=0.05, label="Rescale T",
        info="Warps the denoising time schedule. At 1.0 steps are evenly spaced "
             "from t=1 (noise) to t=0 (output). Values > 1 compress more steps "
             "near t=0, giving extra refinement passes at the end — useful when "
             "part boundaries look fuzzy. Values < 1 do the opposite. "
             "→ Recommended: 1.0 (default) for most cases. "
             "Try 1.5–2.0 if color boundaries between parts are blurry. "
             "Avoid going below 0.5 or above 3.0.")
    guidance = gr.Slider(0.0, 10.0, value=default_guidance, step=0.1,
        label="Guidance strength (CFG)",
        info="Controls how strongly the model follows the conditioning signal "
             "(rendered view for Full/Interactive, 2D map for Guided). "
             "SegviGen uses CFG to align part colorization with the visible "
             "structure of the object — higher values produce crisper but "
             "potentially over-saturated part boundaries. "
             "→ Recommended: 7.5 (default) for all three modes. "
             "Lower to 4–6 if the model over-segments or produces noisy colors. "
             "Raise to 9–10 if important parts are being merged together. "
             "Interactive mode: 5–7 is often sufficient since click points "
             "already constrain the target region.")
    guidance_rescale = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
        label="Guidance rescale",
        info="Corrects the variance of the CFG-guided output to match the "
             "unguided output, preventing color over-saturation at high CFG. "
             "→ Recommended: 0.0 (default, disabled) at CFG ≤ 7.5. "
             "Enable with 0.5–0.7 if you raise CFG above 8 and the output "
             "colors look washed-out or blown-out at part boundaries.")
    gi_start = gr.Slider(0.0, 1.0, value=0.0, step=0.01,
        label="Guidance interval — start",
        info="CFG is only active while the denoising timestep t is between "
             "[start, end]. Delaying the start lets the model freely establish "
             "coarse part structure before guidance locks it in. "
             "→ Recommended: 0.0 (default, guidance on from the very start). "
             "Try 0.1–0.2 if the output has geometric artifacts on complex meshes "
             "(e.g. thin features or many cavities).")
    gi_end = gr.Slider(0.0, 1.0, value=1.0, step=0.01,
        label="Guidance interval — end",
        info="Upper bound of the guidance window. Ending before t=0 lets the "
             "model freely refine fine details in the last steps without CFG "
             "pressure, which can sharpen texture details within each part. "
             "→ Recommended: 1.0 (default). "
             "Try 0.8–0.9 together with start=0.1 for a 'soft guidance' approach "
             "that reduces boundary fringing on high-detail models.")

    gr.Markdown("#### Export parameters")
    decimation = gr.Number(value=100000, label="Decimation target (faces)",
        info="Target face count after QEM mesh simplification. The mesh is "
             "decimated before texture baking, so this directly affects how "
             "well part boundaries are preserved in the color texture. "
             "→ Recommended: 100 000 (default) for general use. "
             "Use 200 000–500 000 if you plan to split the result into parts — "
             "more faces = sharper color boundaries = cleaner splits. "
             "Use 20 000–50 000 for lightweight game-ready or real-time meshes.")
    tex_size = gr.Dropdown([512, 1024, 2048, 4096], value=1024,
        label="Texture size (px)",
        info="Resolution of the square atlas texture baked onto the output mesh. "
             "Since SegviGen encodes part labels as colors in the texture, "
             "higher resolution means sharper, more accurate part boundaries. "
             "→ Recommended: 2048 if you intend to split the mesh into parts "
             "(color precision at boundaries matters). "
             "1024 (default) for general viewing. "
             "4096 for hero assets with many fine parts. "
             "512 for fast iteration or mobile targets.")
    remesh = gr.Checkbox(value=True, label="Remesh",
        info="Runs isotropic remeshing before texture baking to produce uniform "
             "triangle sizes. Uniform topology improves how cleanly the "
             "part-color texture projects onto the mesh. "
             "→ Recommended: enabled (default) for most workflows. "
             "Disable if you need to preserve the exact original mesh topology "
             "for rigging, morph targets, or downstream tools that are "
             "sensitive to vertex ordering.")
    remesh_band = gr.Slider(0, 4, value=1, step=1, label="Remesh band",
        info="Spatial scale of the remeshing grid: 0 = finest triangles, "
             "4 = coarsest. Lower values preserve small geometric features "
             "at the cost of more triangles before decimation. "
             "→ Recommended: 1 (default) for a light pass that keeps most "
             "surface detail. Use 0 for highly detailed models (characters, "
             "mechanical parts). Use 2–3 for smooth organic shapes where "
             "fine topology is less important.")
    remesh_proj = gr.Slider(0, 4, value=0, step=1, label="Remesh project",
        info="Projection iterations that snap the remeshed surface back onto "
             "the original mesh geometry after remeshing. More iterations = "
             "tighter geometric fit but slower processing. "
             "→ Recommended: 0 (default) for speed with minimal surface drift. "
             "Use 1–2 when shape accuracy matters (e.g. mechanical parts, "
             "tight-fitting joints). Use 3–4 for highly curved organic shapes "
             "where remeshing tends to round off sharp features.")

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

    with gr.Row():
        # ── Left column: input ──────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("## Input Model")
            input_model = gr.Model3D(label="Upload GLB / OBJ / PLY", clear_color=[0.1, 0.1, 0.15, 1])

        # ── Right column: output ────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("## Output Model")
            output_model = gr.Model3D(label="Segmented Output", clear_color=[0.1, 0.1, 0.15, 1])

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
                with gr.Column():
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

                with gr.Column():
                    _i = _make_sampler_export_controls()
                    i_steps, i_rescale_t, i_guidance, i_guidance_rescale, i_gi_start, i_gi_end = (
                        _i["steps"], _i["rescale_t"], _i["guidance"], _i["guidance_rescale"], _i["gi_start"], _i["gi_end"])
                    i_decimation, i_tex_size, i_remesh, i_remesh_band, i_remesh_proj = (
                        _i["decimation"], _i["tex_size"], _i["remesh"], _i["remesh_band"], _i["remesh_proj"])

            i_run = gr.Button("Run Interactive Segmentation", variant="primary")

            with gr.Row():
                with gr.Column():
                    i_split_ctrl = _make_split_controls("i")
                    i_split_btn = gr.Button("Split into Parts", variant="secondary")
                with gr.Column():
                    i_parts_model = gr.Model3D(label="Split Parts Output", clear_color=[0.1, 0.1, 0.15, 1])

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
                outputs=[output_model, i_seg_state],
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
                with gr.Column():
                    f_ckpt = gr.Textbox(label="Checkpoint path (.ckpt)", value=DEFAULT_CKPT_FULL, placeholder="path/to/full_seg.ckpt")
                    f_transforms = gr.Textbox(label="Transforms JSON", value=DEFAULT_TRANSFORMS,
                                              placeholder="data_toolkit/transforms.json")
                    f_rendered_img = gr.Image(label="Override rendered image (optional, PNG)",
                                             type="filepath", value=None)

                with gr.Column():
                    _f = _make_sampler_export_controls()
                    f_steps, f_rescale_t, f_guidance, f_guidance_rescale, f_gi_start, f_gi_end = (
                        _f["steps"], _f["rescale_t"], _f["guidance"], _f["guidance_rescale"], _f["gi_start"], _f["gi_end"])
                    f_decimation, f_tex_size, f_remesh, f_remesh_band, f_remesh_proj = (
                        _f["decimation"], _f["tex_size"], _f["remesh"], _f["remesh_band"], _f["remesh_proj"])

            f_run = gr.Button("Run Full Segmentation", variant="primary")

            with gr.Row():
                with gr.Column():
                    f_split_ctrl = _make_split_controls("f")
                    f_split_btn = gr.Button("Split into Parts", variant="secondary")
                with gr.Column():
                    f_parts_model = gr.Model3D(label="Split Parts Output", clear_color=[0.1, 0.1, 0.15, 1])

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
                outputs=[output_model, f_seg_state],
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
                with gr.Column():
                    t_ckpt = gr.Textbox(label="Checkpoint path (.ckpt)",
                                        value=DEFAULT_CKPT_FULL_2D, placeholder="path/to/full_seg_w_2d_map.ckpt")
                    t_guidance_img = gr.Image(label="2D Guidance Map (PNG — unique color per part)",
                                              type="filepath")

                with gr.Column():
                    _t = _make_sampler_export_controls()
                    t_steps, t_rescale_t, t_guidance, t_guidance_rescale, t_gi_start, t_gi_end = (
                        _t["steps"], _t["rescale_t"], _t["guidance"], _t["guidance_rescale"], _t["gi_start"], _t["gi_end"])
                    t_decimation, t_tex_size, t_remesh, t_remesh_band, t_remesh_proj = (
                        _t["decimation"], _t["tex_size"], _t["remesh"], _t["remesh_band"], _t["remesh_proj"])

            t_run = gr.Button("Run 2D-Guided Segmentation", variant="primary")

            with gr.Row():
                with gr.Column():
                    t_split_ctrl = _make_split_controls("t")
                    t_split_btn = gr.Button("Split into Parts", variant="secondary")
                with gr.Column():
                    t_parts_model = gr.Model3D(label="Split Parts Output", clear_color=[0.1, 0.1, 0.15, 1])

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
                outputs=[output_model, t_seg_state],
            )

            t_split_btn.click(
                fn=run_split,
                inputs=[t_seg_state] + list(t_split_ctrl.values()),
                outputs=t_parts_model,
            )

    gr.Markdown(
        "---\n"
        "**Tip:** Base models (TRELLIS.2-4B) are loaded once on first run and cached. "
        "Segmentation checkpoints are also cached per path. "
        "Intermediate VXZ files are written to a system temp directory and cleaned up automatically."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
