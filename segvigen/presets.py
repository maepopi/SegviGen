"""segvigen.presets — Sampler and split presets."""

SAMPLER_PRESETS = {
    "fast":     dict(steps=12,  rescale_t=1.0, guidance_strength=7.5, guidance_rescale=0.0,
                     guidance_interval_start=0.0, guidance_interval_end=1.0,
                     decimation_target=50_000,  texture_size=512,
                     remesh=True, remesh_band=1, remesh_project=0),
    "balanced": dict(steps=25,  rescale_t=1.0, guidance_strength=7.5, guidance_rescale=0.0,
                     guidance_interval_start=0.0, guidance_interval_end=1.0,
                     decimation_target=100_000, texture_size=1024,
                     remesh=True, remesh_band=1, remesh_project=0),
    "quality":  dict(steps=50,  rescale_t=1.5, guidance_strength=7.5, guidance_rescale=0.0,
                     guidance_interval_start=0.1, guidance_interval_end=0.9,
                     decimation_target=300_000, texture_size=2048,
                     remesh=True, remesh_band=0, remesh_project=1),
}

SPLIT_PRESETS = {
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
