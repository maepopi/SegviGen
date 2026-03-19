# SegviGen: Repurposing 3D Generative Model for Part Segmentation

---

## 🖥️ This Fork — Gradio Web UI

This fork adds a **Gradio-based web interface** (`app.py`) that wraps all three SegviGen segmentation methods in an interactive browser UI.
It was built with the assistance of **[Claude](https://claude.ai)** (Anthropic).

### What this fork adds

| | |
|---|---|
| `app.py` | Full Gradio app exposing all three segmentation methods |
| `.gitignore` | Excludes checkpoints, caches, and build artifacts |
| README section | This section |

### Features of the Gradio UI

- **Input 3D model viewer** — upload any GLB file and preview it interactively in the browser
- **Three segmentation tabs**, one per method:
  - **Interactive Part Segmentation** — specify 3D voxel click points (x y z, up to 10) to isolate a single part
  - **Full Segmentation** — automatically segment all parts conditioned on a rendered view
  - **Full Segmentation + 2D Guidance Map** — control part granularity by uploading a 2D semantic map (unique color per part)
- **All parameters exposed** per method:
  - Sampler: steps, rescale T, guidance strength (CFG), guidance rescale, guidance interval
  - Export: decimation target, texture size, remesh on/off, remesh band, remesh project
- **Output 3D model viewer** — the segmented GLB renders directly in the browser
- **Automatic checkpoint detection** — checkpoint paths in `ckpt/` are pre-filled at startup
- **Model caching** — TRELLIS.2-4B base models and segmentation checkpoints load once and are reused across runs

### Installation notes for this fork

On top of the standard SegviGen environment, this fork requires:

```sh
# Build o_voxel from the TRELLIS.2 source (includes cumesh + flex_gemm)
pip install path/to/TRELLIS.2/o-voxel --no-build-isolation

# bpy 4.0.0 is unavailable on PyPI; use 4.1.0 instead (API-compatible)
pip install bpy==4.1.0 --extra-index-url https://download.blender.org/pypi/

# Gradio 6.x (already pulled in by TRELLIS.2 setup)
pip install gradio==6.0.1
```

> **Python version:** use Python 3.11. Python 3.12 is not supported — `bpy` 4.x has no wheels for it.

### Running the UI

```sh
conda activate trellis2   # or your equivalent Python 3.11 env
python app.py
# → http://localhost:7860
```

Place your checkpoints in `ckpt/` following the naming convention:

| File | Method |
|------|--------|
| `ckpt/interactive_seg.ckpt` | Interactive Part Segmentation |
| `ckpt/full_seg.ckpt` | Full Segmentation |
| `ckpt/full_seg_w_2d_map.ckpt` | Full Segmentation + 2D Guidance |

Checkpoints are available on [Hugging Face](https://huggingface.co/fenghora/SegviGen).

### Bug fixes included

- `ColorVisuals → TextureVisuals` conversion before voxelization, so GLBs with vertex/flat colors work correctly
- `pipeline.json` resolved via `huggingface_hub` instead of a hardcoded relative path
- Gradio 6.x compatibility (`css=` and `theme=` removed from `gr.Blocks`)

---

![teaser](assets/teaser.png)

## 🏠 [Project Page](https://fenghora.github.io/SegviGen-Page/) | [Paper](https://arxiv.org/abs/2603.16869) | [Online Demo](https://huggingface.co/spaces/fenghora/SegviGen)

***SegviGen*** is a framework for 3D part segmentation that leverages the rich 3D structural and textural knowledge encoded in large-scale 3D generative models. 
It learns to predict part-indicative colors while reconstructing geometry, and unifies three settings in one architecture: **interactive part segmentation**, **full segmentation**, and **2D segmentation map–guided full segmentation** with arbitrary granularity.


## 🌟 Features
- **Repurposed 3D Generative Priors for Data Efficiency**: By reusing the rich structural and textural knowledge encoded in large-scale native 3D generative models, ***SegviGen*** learns 3D part segmentation with minimal task-specific supervision, requiring only **0.32%** training data.
- **Unified and Flexible Segmentation Settings**: Supports **interactive part segmentation**, **full segmentation**, and **2D segmentation map–guided full segmentation** with arbitrary part granularity under a single architecture.
- **State-of-the-Art Accuracy**: Consistently surpasses P3-SAM, delivering a **40%** gain in IoU@1 for single-click interaction on PartObjaverse-Tiny and PartNeXT, and a **15%** improvement in overall IoU for unguided full segmentation averaged across datasets.


## 🔨 Installation

### Prerequisites
- **System**: Linux
- **GPU**: A NVIDIA GPU with at least 24GB of memory is necessary
- **Python**: 3.10

### Installation Steps
1. Create the environment of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
    ```sh
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd TRELLIS.2
    ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
    ```

2. Install the rest of requirements
    ```sh
    pip install mathutils
    pip install transformers==4.57.6 # https://github.com/microsoft/TRELLIS.2/issues/101
    pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
    sudo apt-get install -y libsm6 libxrender1 libxext6
    pip install --upgrade Pillow
    ```


### Pretrained Weights

The checkpoints of **Interactive part-segmentation**, **Full segmentation** and **Full segmentation with 2D guidance** are available on [Hugging Face](https://huggingface.co/fenghora/SegviGen).

## 📒 Usage

- **Interactive part-segmentation**
    ```sh
    python inference_interactive.py \
        --ckpt_path path/to/interactive_seg.ckpt \
        --glb ./data_toolkit/assets/example.glb \
        --input_vxz ./data_toolkit/assets/input.vxz \
        --transforms ./data_toolkit/transforms.json \
        --img ./data_toolkit/assets/img.png \
        --export_glb ./data_toolkit/assets/output.glb \
        --input_vxz_points 388 448 392
    ```

- **Full segmentation**
    ```sh
    python inference_full.py \
        --ckpt_path path/to/full_seg.ckpt \
        --glb ./data_toolkit/assets/example.glb \
        --input_vxz ./data_toolkit/assets/input.vxz \
        --transforms ./data_toolkit/transforms.json \
        --img ./data_toolkit/assets/img.png \
        --export_glb ./data_toolkit/assets/output.glb
    ```

- **Full segmentation with 2D guidance**
    ```sh
    python inference_full.py \
        --ckpt_path path/to/full_seg_w_2d_map.ckpt \
        --glb ./data_toolkit/assets/example.glb \
        --input_vxz ./data_toolkit/assets/input.vxz \
        --img ./data_toolkit/assets/full_seg_w_2d_map/2d_map.png \
        --export_glb ./data_toolkit/assets/output.glb \
        --two_d_map
    ```

## ⚖️ License

This project is licensed under the [MIT License](https://github.com/Nelipot-Lee/SegviGen/blob/main/LICENSE).  
However, please note that the code in **`trellis2`** originates from the [TRELLIS.2](https://github.com/Microsoft/TRELLIS.2) project and remains subject to its original license terms.  
Users must comply with the licensing requirements of [TRELLIS.2](https://github.com/Microsoft/TRELLIS.2) when using or redistributing that portion of the code.

## Citation

```
@article{li2026segvigen,
      title = {SegviGen: Repurposing 3D Generative Model for Part Segmentation}, 
      author = {Lin Li and Haoran Feng and Zehuan Huang and Haohua Chen and Wenbo Nie and Shaohua Hou and Keqing Fan and Pan Hu and Sheng Wang and Buyu Li and Lu Sheng},
      journal = {arXiv preprint arXiv:2603.16869},
      year = {2026}
}
``` 
