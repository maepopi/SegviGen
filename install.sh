#!/usr/bin/env bash
# SegviGen install script
# Tested on Ubuntu/Debian with CUDA 12.x / 13.x and conda.
# GPU-agnostic: detects CUDA compute capability at install time.
#
# What this does:
#   1. Clones TRELLIS.2 (Microsoft) and builds its CUDA extensions
#      (o_voxel, cumesh, flex_gemm, flash-attn, nvdiffrast, nvdiffrec)
#      inside a new conda env "segvigen" (Python 3.10).
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
    echo "[1/6] Cloning TRELLIS.2 (this may take a few minutes) …"
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive TRELLIS.2
else
    echo "[1/6] TRELLIS.2 already cloned — skipping."
fi

# ─── 2. Build TRELLIS.2 env + CUDA extensions ────────────────────────────────
# We create the conda env manually so we can pin the torch version to match
# the pixmesh env (torch==2.7.0+cu128 + flash-attn 2.7.4 pre-built wheel).
# This step takes 30–60 minutes depending on your CPU.
echo "[2/6] Creating conda env 'segvigen' with Python 3.10 …"
eval "$(conda shell.bash hook)"

# Remove stale env if present so we get a clean slate
conda env remove -n segvigen -y 2>/dev/null || true

conda create -n segvigen python=3.10 -y
conda activate segvigen
echo "[2/6] conda env active: $(which python)"

echo "[2/6] Installing PyTorch 2.7.0 (cu128 — matches pixmesh env) …"
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128

echo "[2/6] Building TRELLIS.2 CUDA extensions (30–60 min) …"
# Auto-detect all installed GPUs via nvidia-smi and build the arch list so
# the compiled extensions work on any GPU without manual edits.
# Falls back to a broad safe list if nvidia-smi is unavailable (CI / CPU-only).
if command -v nvidia-smi &>/dev/null; then
    TORCH_CUDA_ARCH_LIST="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
        | sort -u \
        | tr '\n' ' ' \
        | sed 's/ *$//'
    )"
    echo "[2/6] Detected GPU arch(es): $TORCH_CUDA_ARCH_LIST"
else
    # Common arches: Turing(7.5), Ampere(8.0/8.6), Ada(8.9), Hopper(9.0), Blackwell(12.0)
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 12.0"
    echo "[2/6] nvidia-smi not found — using fallback arch list: $TORCH_CUDA_ARCH_LIST"
fi
export TORCH_CUDA_ARCH_LIST
cd TRELLIS.2
# --new-env is intentionally omitted — env + torch already installed above.
bash setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
cd "$SCRIPT_DIR"

# Pre-built flash-attn wheel — same wheel used in pixmesh with torch==2.7.0+cu128.
pip install https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl

# ─── Patch flex_gemm to survive absent/broken CUDA context at import time ─────
# flex_gemm calls torch.cuda.get_device_name() at module-level in three places.
# When nvidia_uvm is not loaded (driver mismatch, first boot, CI) this raises a
# RuntimeError before any user code runs.  These patches add try/except guards
# so the server can at least start; actual GPU work still uses CUDA at runtime.
python - << 'PATCH_EOF'
import sys

SITE = next(p for p in sys.path if "site-packages" in p)

# ── utils.py ──────────────────────────────────────────────────────────────────
utils_path = f"{SITE}/flex_gemm/kernels/triton/utils.py"
with open(utils_path) as f:
    src = f.read()

old = "def get_gpu_name():\n    return torch.cuda.get_device_name()"
new = (
    "def get_gpu_name():\n"
    "    try:\n"
    "        return torch.cuda.get_device_name()\n"
    "    except Exception:\n"
    "        return \"\""
)
if old in src:
    src = src.replace(old, new)
    print("flex_gemm/kernels/triton/utils.py  get_gpu_name patched")
else:
    print("flex_gemm/kernels/triton/utils.py  get_gpu_name already patched — skipping")

# get_platform_name uses torch.cuda.is_available() which returns False when
# nvidia_uvm isn't loaded — causing get_autotune_config to fall all the way
# through to the ValueError.  Fall back to /proc/driver/nvidia/version so
# that 'cuda' is returned whenever the nvidia.ko driver is loaded.
old_plat = (
    "def get_platform_name():\n"
    "    if torch.cuda.is_available():\n"
    "        if getattr(torch.version, 'hip', None) is not None:\n"
    "            return 'hip'\n"
    "        return 'cuda'\n"
    "    return 'unknown'"
)
new_plat = (
    "def get_platform_name():\n"
    "    if torch.cuda.is_available():\n"
    "        if getattr(torch.version, 'hip', None) is not None:\n"
    "            return 'hip'\n"
    "        return 'cuda'\n"
    "    # nvidia.ko is loaded even when nvidia_uvm fails (driver mismatch).\n"
    "    # Detect this so autotune still picks the correct 'cuda' config.\n"
    "    import os, shutil\n"
    "    if os.path.exists('/proc/driver/nvidia/version') or shutil.which('nvidia-smi'):\n"
    "        return 'cuda'\n"
    "    return 'unknown'"
)
if old_plat in src:
    src = src.replace(old_plat, new_plat)
    print("flex_gemm/kernels/triton/utils.py  get_platform_name patched")
else:
    print("flex_gemm/kernels/triton/utils.py  get_platform_name already patched — skipping")

with open(utils_path, "w") as f:
    f.write(src)

# ── autotuner.py ──────────────────────────────────────────────────────────────
autotuner_path = f"{SITE}/flex_gemm/utils/autotuner.py"
with open(autotuner_path) as f:
    src = f.read()

old1 = (
    "    if cache is None:\n        return\n\n"
    "    device_name = torch.cuda.get_device_name()"
)
new1 = (
    "    if cache is None:\n        return\n\n"
    "    try:\n        device_name = torch.cuda.get_device_name()\n"
    "    except Exception:\n        return"
)
old2 = (
    "def get_autotune_cache():\n    cache = {}\n"
    "    device_name = torch.cuda.get_device_name()"
)
new2 = (
    "def get_autotune_cache():\n    cache = {}\n"
    "    try:\n        device_name = torch.cuda.get_device_name()\n"
    "    except Exception:\n        return cache"
)
# Benchmark loop: catch per-config RuntimeError (e.g. PassManager::run failed)
# so that one bad Triton config doesn't crash the whole autotuner.
old3 = (
    "                bench_start = time.time()\n"
    "                timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}\n"
    "                bench_end = time.time()\n"
    "                self.bench_time = bench_end - bench_start\n"
    "                self.cache[key] = builtins.min(timings, key=timings.get)"
)
new3 = (
    "                bench_start = time.time()\n"
    "                timings = {}\n"
    "                for _cfg in pruned_configs:\n"
    "                    try:\n"
    "                        timings[_cfg] = self._bench(*args, config=_cfg, **kwargs)\n"
    "                    except RuntimeError:\n"
    "                        pass\n"
    "                if not timings:\n"
    "                    raise RuntimeError(\n"
    "                        'All autotuning configs failed to compile for this kernel. '\n"
    "                        'Try: rm -rf ~/.triton/cache'\n"
    "                    )\n"
    "                bench_end = time.time()\n"
    "                self.bench_time = bench_end - bench_start\n"
    "                self.cache[key] = builtins.min(timings, key=timings.get)"
)

changed = False
for old, new in [(old1, new1), (old2, new2), (old3, new3)]:
    if old in src:
        src = src.replace(old, new)
        changed = True
if changed:
    with open(autotuner_path, "w") as f:
        f.write(src)
    print("flex_gemm/utils/autotuner.py       patched")
else:
    print("flex_gemm/utils/autotuner.py       already patched or changed — skipping")
PATCH_EOF

# Clear any stale Triton kernel cache built with a different Triton version
# or broken configs — it will be rebuilt cleanly on first inference run.
# Also clear the flex_gemm persistent autotune cache: it stores best kernel
# configs keyed by input shape and Triton version; a stale entry from a
# different Triton version causes "All autotuning configs failed" at runtime.
rm -rf ~/.triton/cache
rm -f ~/.flex_gemm/autotune_cache.json

# ─── 3. Install SegviGen deps (env already active) ───────────────────────────
echo "[3/6] Installing SegviGen dependencies …"

# Remove any installed trellis2 package so SegviGen's local trellis2/ is used
pip uninstall trellis2 -y 2>/dev/null || true
# Remove any trellis2.pth files across all site-packages directories.
python -c "
import site, os, glob
for d in site.getsitepackages():
    for f in glob.glob(os.path.join(d, 'trellis2.pth')):
        os.remove(f)
        print(f'Removed {f}')
" 2>/dev/null || true

# Server runtime deps (used by server.py / inference*.py)
pip install fastapi "uvicorn[standard]" python-multipart trimesh tqdm

pip install "transformers==4.57.6"

# Fix PIL BEFORE any package that depends on it.
# TRELLIS.2 setup.sh (--basic) installs pillow-simd which corrupts the PIL
# namespace.  Purge every PIL-related package and reinstall Pillow==11.2.1
# (same version as pixmesh) before bpy / rembg.
pip uninstall pillow pillow-simd Pillow -y 2>/dev/null || true
pip install "Pillow==11.2.1"

# bpy 4.0.0 is the latest wheel on https://download.blender.org/pypi/ and
# bundles its own mathutils — do NOT install mathutils separately.
# Requires Python 3.10 or 3.11.
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/

# rembg: replaces the gated briaai/RMBG-2.0 model — no HuggingFace token required.
# Its default model (isnet-general-use) is downloaded automatically on first use.
pip install "rembg[gpu]"

# ─── 4. System libraries for OpenCV / OpenEXR ────────────────────────────────
echo "[4/6] Installing system libraries (needs sudo) …"
sudo apt-get install -y libsm6 libxrender1 libxext6 libopenexr-dev

# ─── 5. Verify ───────────────────────────────────────────────────────────────
echo "[5/6] Verifying key imports …"
# trimesh / bpy / rembg do not need an active CUDA context.
python - <<'EOF'
import trimesh;  print("trimesh  OK")
import bpy;      print("bpy      OK")
import rembg;    print("rembg    OK")
EOF

# o_voxel → flex_gemm calls torch.cuda.get_device_name() at import time, so
# it requires a live CUDA context.  Warn instead of aborting the install.
python - <<'EOF'
try:
    import o_voxel; print("o_voxel  OK")
    import cumesh;  print("cumesh   OK")
except RuntimeError as e:
    print(f"WARNING: o_voxel/cumesh import skipped during install (no CUDA context): {e}")
    print("  These packages are correctly installed and will work at runtime.")
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

# torch 2.7.0 bundles triton==3.3.0 which fails to compile flex_gemm kernels
# on newer GPU architectures (e.g. Blackwell sm_120).  triton==3.6.0 fixes this.
# The "dependency conflict" warning from pip is harmless — torch works fine with
# triton 3.6.0 at runtime despite the pinned requirement in its metadata.
pip install triton==3.6.0 --no-warn-conflicts

echo ""
echo "=== Installation complete ==="
echo ""
echo "Then launch:"
echo "  conda activate segvigen"
echo "  cd $SCRIPT_DIR"
echo "  python app.py"
echo "  # → http://localhost:7860"
