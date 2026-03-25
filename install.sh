#!/usr/bin/env bash
# SegviGen install script — RTX 3060 (12 GB VRAM)
# Tested on Ubuntu/Debian with CUDA 12.x and conda.
#
# What this does:
#   1. Clones TRELLIS.2 (Microsoft) and builds its CUDA extensions
#      (o_voxel, cumesh, flex_gemm, flash-attn, nvdiffrast, nvdiffrec)
#      inside a new conda env "segvigen" (Python 3.11).
#   2. Installs SegviGen-specific packages into that env.
#   3. Creates the ckpt/ directory.
#
# After running, activate the env and launch:
#   conda activate segvigen
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
echo "[2/5] Creating conda env 'segvigen' with Python 3.11 …"
eval "$(conda shell.bash hook)"

# Remove stale env if present so we get a clean slate
conda env remove -n segvigen -y 2>/dev/null || true

conda create -n segvigen python=3.11 -y
conda activate segvigen
echo "[2/5] conda env active: $(which python)"

echo "[2/5] Installing PyTorch > 2.7.1 (cu128) …"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "[2/5] Building TRELLIS.2 CUDA extensions (30–60 min) …"
cd TRELLIS.2
# --new-env is intentionally omitted — env + torch already installed above.
bash setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
cd "$SCRIPT_DIR"

# echo "[2/5] Installing flash-attn (Blackwell sm_120 support) …"
# # --no-build-isolation makes the already-installed torch visible to setup.py.
# # MAX_JOBS=4 limits parallel C++ compilation to avoid OOM on the build machine.
# MAX_JOBS=4 pip install flash-attn --no-build-isolation

# ─── 3. Install SegviGen deps (env already active) ───────────────────────────
echo "[3/5] Installing SegviGen dependencies …"

# Remove any installed trellis2 package so SegviGen's local trellis2/ is used
pip uninstall trellis2 -y 2>/dev/null || true
find "$(python -c "import site; print('\n'.join(site.getsitepackages()))")" \
    -name "trellis2.pth" -delete 2>/dev/null || true

# mathutils requires libeigen3-dev to compile — install it now (before the build).
sudo apt-get install -y libeigen3-dev

# mathutils 5.1.0 uses PyLong_AsInt (added in Python 3.12, absent in 3.10/3.11)
# and re-declares _PyArg_CheckPositional (removed in 3.13) — both require patching.
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

# bpy: 4.1.0 is the lowest available wheel on https://download.blender.org/pypi/.
# Requires Python 3.10 or 3.11.
pip install bpy==4.1.0 --extra-index-url https://download.blender.org/pypi/

pip install "gradio==6.0.1"
# TRELLIS.2 setup.sh (--basic) installs pillow-simd which conflicts with Pillow.
# Uninstall it first so our pin takes sole ownership of the PIL namespace.
pip uninstall pillow-simd -y 2>/dev/null || true
# gradio 6.0.1 requires PIL._webp.HAVE_WEBPANIM which was removed in Pillow 11.
# This must be the last pip install so nothing can upgrade it back.
pip install "Pillow>=10.0.0,<11.0.0"

# rembg: replaces the gated briaai/RMBG-2.0 model — no HuggingFace token required.
# Its default model (isnet-general-use) is downloaded automatically on first use.
pip install "rembg[gpu]"

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
import rembg;    print("rembg    OK")
EOF

mkdir -p "$SCRIPT_DIR/ckpt"

# ─── 6. Download checkpoints (skip files that already exist) ─────────────────
CKPT_FILES=("interactive_seg.ckpt" "full_seg.ckpt" "full_seg_w_2d_map.ckpt")
MISSING=()
for f in "${CKPT_FILES[@]}"; do
    [[ ! -f "$SCRIPT_DIR/ckpt/$f" ]] && MISSING+=("$f")
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
    echo "[6/6] All checkpoints already present — skipping download."
else
    echo "[6/6] Downloading ${#MISSING[@]} missing checkpoint(s) from HuggingFace …"
    pip install -q huggingface_hub
    python - "$SCRIPT_DIR/ckpt" "${MISSING[@]}" <<'PYEOF'
import shutil, os, sys
from huggingface_hub import hf_hub_download
ckpt_dir, *files = sys.argv[1:]
for f in files:
    dest = os.path.join(ckpt_dir, f)
    print(f"  Downloading {f} …", flush=True)
    path = hf_hub_download("fenghora/SegviGen", f)
    shutil.copy(path, dest)
    print(f"  Saved → {dest}", flush=True)
PYEOF
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Then launch:"
echo "  conda activate segvigen"
echo "  cd $SCRIPT_DIR"
echo "  python app.py"
echo "  # → http://localhost:7860"
