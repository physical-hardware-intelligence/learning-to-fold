#!/usr/bin/env bash
# Track B · B4 — Ilia's policy server with the real-round flags (plan §7, README).
# Listens on ws://0.0.0.0:8000 (hard-coded upstream, serve.py:423). Stop it when the tests end.
# Log: ~/phi/logs/b4_server.log
set -euo pipefail
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M VIRTUAL_ENV PYTHONPATH
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd ~/phi/lehome_solution
echo "=== [$(date '+%H:%M:%S')] GPU before start: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)"
exec uv run python scripts/serve.py \
  --actions_to_execute 5 --actions_to_keep 1 --execute_in_n_steps 5 --num_steps 10 \
  policy:checkpoint --policy.config pi_modified_real_bc --policy.dir outputs/checkpoints/lehome_real
