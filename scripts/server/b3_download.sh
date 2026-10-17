#!/usr/bin/env bash
# Track B · B3 — download the real-round checkpoint + the organizer's real dataset (plan §7).
# Revisions pinned to the Hub commits recorded 2026-09-28. Log: ~/phi/logs/b3_download.log
# Our GPU box can't reach Hugging Face (plan, trap 17): there, download on a laptop and copy (plan, B3).
# This script is for a machine that can reach it.
set -euo pipefail
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M VIRTUAL_ENV PYTHONPATH
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

MODEL_REV=03b26d31bfe6859b438cb793bb911d607226eef6   # IliaLarchenko/lehome_real: 27 files, 10,492,307,528 B
DATA_REV=5a28286fb60db8bf9fab9a552a001645722752cd    # lehome/dataset_challenge_real, 4 subsets: 51 files, 4,479,323,893 B
step() { echo; echo "=== [$(date '+%H:%M:%S')] $*"; }
cd ~/phi/lehome_solution

step "1/3 checkpoint IliaLarchenko/lehome_real @ ${MODEL_REV:0:7}"
time uv run hf download IliaLarchenko/lehome_real --revision "$MODEL_REV" --local-dir outputs/checkpoints/lehome_real

step "2/3 dataset lehome/dataset_challenge_real @ ${DATA_REV:0:7}, 4 garment subsets"
time uv run hf download lehome/dataset_challenge_real --repo-type dataset --revision "$DATA_REV" \
  --local-dir data/lehome_real \
  --include "top_long_merged/*" "top_short_merged/*" "pant_long_merged/*" "pant_short_merged/*"

step "3/3 verify: file count + bytes vs the Hub (hf's own .cache metadata excluded)"
for d in outputs/checkpoints/lehome_real data/lehome_real; do
  n=$(find "$d" -type f -not -path '*/.cache/*' | wc -l)
  b=$(find "$d" -type f -not -path '*/.cache/*' -printf '%s\n' | awk '{s+=$1} END {print s}')
  echo "$d  files=$n  bytes=$b"
done
echo "expect: checkpoint files=27 bytes=10492307528 · dataset files=51 bytes=4479323893"
step "B3 DONE"
