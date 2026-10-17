#!/usr/bin/env bash
# Track B · B1 — build the Learning to Fold policy-server env (plan §7).
# Userspace only: no sudo, no conda. Log: ~/phi/logs/b1_build.log
# Run on the box:   bash ~/phi/learning-to-fold/scripts/server/b1_build.sh 2>&1 | tee ~/phi/logs/b1_build.log
set -euo pipefail

# Clean environment: no conda, no inherited venv (plan trap 16).
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M VIRTUAL_ENV PYTHONPATH
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
export UV_HTTP_TIMEOUT=300

PIN=ca73828   # the lehome_solution commit the plan was verified against
step() { echo; echo "=== [$(date '+%H:%M:%S')] $*"; }

step "1/4 uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

step "2/4 clone lehome_solution @ $PIN, submodules at its pins"
cd ~/phi
if [ ! -d lehome_solution ]; then
  git clone https://github.com/IliaLarchenko/lehome_solution lehome_solution
fi
cd lehome_solution
echo "upstream main = $(git rev-parse --short origin/main); commits after $PIN: $(git rev-list --count $PIN..origin/main)"
git checkout -q "$PIN"
git submodule update --init --recursive
echo "lehome_solution  $(git rev-parse --short HEAD)"
echo "openpi           $(git -C openpi rev-parse --short=7 HEAD)   (expect c23745b)"
echo "lehome-challenge $(git -C lehome-challenge rev-parse --short=7 HEAD)   (expect 5ea947e)"
test "$(git -C openpi rev-parse --short=7 HEAD)" = c23745b
test "$(git -C lehome-challenge rev-parse --short=7 HEAD)" = 5ea947e

step "3/4 uv sync --python 3.11 --extra dev"
time uv sync --python 3.11 --extra dev
cp uv.lock ~/phi/lehome_uv.lock
echo "resolved lock saved: ~/phi/lehome_uv.lock"

step "4/4 interpreter + key versions"
uv run python - <<'PY'
import sys, importlib.metadata as m
print(sys.executable, sys.version.split()[0])
for p in ["jax", "jaxlib", "jax-cuda12-plugin", "flax", "orbax-checkpoint", "torch",
          "torchcodec", "lerobot", "numpy", "openpi", "websockets", "huggingface-hub"]:
    try:
        print(f"{p:20s} {m.version(p)}")
    except Exception as e:
        print(f"{p:20s} MISSING ({type(e).__name__})")
PY
step "B1 DONE"
