#!/usr/bin/env bash
# SegviGen install script — RTX 3060 (12 GB VRAM)
# Tested on Ubuntu/Debian with CUDA 12.x and conda.
#
# What this does:
#   1. Clones TRELLIS.2 (Microsoft) and builds its CUDA extensions
#      (o_voxel, cumesh, flex_gemm, flash-attn, nvdiffrast, nvdiffrec)
#      inside a new conda env "trellis2" (Python 3.10).
#   2. Installs SegviGen-specific packages into that env.
#   3. Creates the ckpt/ directory.
#
# After running, activate the env and launch:
#   conda activate trellis2
#   python app.py
#
# Checkpoints must be downloaded separately from:
#   https://huggingface.co/fenghora/SegviGen
# and placed in ckpt/:
#   ckpt/interactive_seg.ckpt
#   ckpt/full_seg.ckpt
#   ckpt/full_seg_w_2d_map.ckpt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── 1. Clone TRELLIS.2 ───────────────────────────────────────────────────────
if [ ! -d "TRELLIS.2" ]; then
    echo "[1/5] Cloning TRELLIS.2 (this may take a few minutes) …"
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive TRELLIS.2
else
    echo "[1/5] TRELLIS.2 already cloned — skipping."
fi

# ─── 2. Build TRELLIS.2 env + CUDA extensions ────────────────────────────────
# We create the conda env manually so we can pin torch > 2.7.1 (cu128 wheels).
# TRELLIS.2's --new-env installs torch==2.6.0 which we skip by omitting it.
# This step takes 30–60 minutes depending on your CPU.
echo "[2/5] Creating conda env 'trellis2' with Python 3.10 …"
eval "$(conda shell.bash hook)"

# Remove stale env if present so we get a clean slate
conda env remove -n trellis2 -y 2>/dev/null || true

conda create -n trellis2 python=3.10 -y
conda activate trellis2

echo "[2/5] Installing PyTorch > 2.7.1 (cu128) …"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "[2/5] Building TRELLIS.2 CUDA extensions (30–60 min) …"
cd TRELLIS.2
# --new-env is intentionally omitted — env + torch already installed above
bash setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
cd "$SCRIPT_DIR"

# ─── 3. Install SegviGen deps (env already active) ───────────────────────────
echo "[3/5] Installing SegviGen dependencies …"

# Remove any installed trellis2 package so SegviGen's local trellis2/ is used
pip uninstall trellis2 -y 2>/dev/null || true
find "$(python -c "import site; print('\n'.join(site.getsitepackages()))")" \
    -name "trellis2.pth" -delete 2>/dev/null || true

# mathutils 5.1.0 uses PyLong_AsInt (Python 3.12+) and _PyArg_CheckPositional
# (removed in 3.13) — neither compiles cleanly on Python 3.10 without patching.
pip download mathutils==5.1.0 --no-deps -d /tmp/mathutils_src/
cd /tmp && tar -xzf mathutils_src/mathutils-5.1.0.tar.gz && cd mathutils-5.1.0
# Patch 1: PyLong_AsInt → (int)PyLong_AsLong
sed -i 's/PyLong_AsInt(/(int)PyLong_AsLong(/g' \
    src/generic/py_capi_utils.hh src/generic/py_capi_utils.cc
# Patch 2: guard _PyArg_CheckPositional for Python < 3.13
sed -i 's|^int _PyArg_CheckPositional.*|#if PY_VERSION_HEX >= 0x030d0000\n&\n#endif|' \
    src/generic/python_compat.hh
sed -i 's|^/\* Removed in Python 3\.13\. \*/|/* Removed in Python 3.13. */\n#if PY_VERSION_HEX >= 0x030d0000|' \
    src/generic/python_compat.cc
printf '\n#endif /* PY_VERSION_HEX >= 0x030d0000 */\n' >> src/generic/python_compat.cc
pip install . --no-build-isolation
cd "$SCRIPT_DIR"

pip install "transformers==4.57.6"

# bpy: 4.0.0 is the latest available wheel (4.1.0 does not exist on PyPI).
# Note: bpy has no Python 3.12 wheels — requires Python 3.10 or 3.11.
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/

pip install "gradio==6.0.1"
pip install --upgrade Pillow

# ─── 4. System libraries for OpenCV / OpenEXR ────────────────────────────────
echo "[4/5] Installing system libraries (needs sudo) …"
sudo apt-get install -y libsm6 libxrender1 libxext6 libopenexr-dev

# ─── 5. Verify ───────────────────────────────────────────────────────────────
echo "[5/5] Verifying key imports …"
python - <<'EOF'
import o_voxel;  print("o_voxel  OK")
import cumesh;   print("cumesh   OK")
import trimesh;  print("trimesh  OK")
import gradio;   print("gradio   OK")
import bpy;      print("bpy      OK")
EOF

mkdir -p "$SCRIPT_DIR/ckpt"

echo ""
echo "=== Installation complete ==="
echo ""
echo "Download checkpoints from https://huggingface.co/fenghora/SegviGen"
echo "and place them in:  $SCRIPT_DIR/ckpt/"
echo "  interactive_seg.ckpt"
echo "  full_seg.ckpt"
echo "  full_seg_w_2d_map.ckpt"
echo ""
echo "You can download them with:"
echo "  conda activate trellis2"
echo "  pip install huggingface_hub"
echo "  python -c \""
echo "    from huggingface_hub import hf_hub_download"
echo "    import shutil, os"
echo "    ckpt_dir = '$SCRIPT_DIR/ckpt'"
echo "    for f in ['interactive_seg.ckpt', 'full_seg.ckpt', 'full_seg_w_2d_map.ckpt']:"
echo "        path = hf_hub_download('fenghora/SegviGen', f)"
echo "        shutil.copy(path, os.path.join(ckpt_dir, f))"
echo "  \""
echo ""
echo "Then launch:"
echo "  conda activate trellis2"
echo "  cd $SCRIPT_DIR"
echo "  python app.py"
echo "  # → http://localhost:7860"
