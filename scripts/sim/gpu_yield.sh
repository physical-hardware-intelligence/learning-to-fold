#!/usr/bin/env bash
# Shared-GPU courtesy: while one of our runs is going, step aside if anyone else starts a GPU job.
# Every 20 s: if another user's compute job holds more than 1 GiB on two checks in a row, stop our
# run (SIGTERM, then SIGKILL after 15 s) and write why to <run folder>/yield.log.
# Ends by itself when the run ends (run.txt gets its "end:" line).
# Usage, next to run_sim_eval.sh:   bash gpu_yield.sh ~/phi/logs/sim/<date-time>-<name>
set -u
out=$1
me=$(whoami)
log="$out/yield.log"
ours='scripts/run_eval.py|scripts/serve.py|scripts.eval --policy_type'   # the 3 processes of one evaluation
stamp() { date '+%F %H:%M:%S'; }
echo "[$(stamp)] watching for other users' GPU jobs over 1 GiB" >> "$log"
seen=""
while ! grep -q '^end:' "$out/run.txt" 2>/dev/null; do
  now=""
  while IFS=', ' read -r pid mem; do
    [ -z "$pid" ] && continue
    user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$user" ] && [ "$user" != "$me" ] && [ "${mem:-0}" -gt 1024 ] && now="$now $user:$pid:${mem}MiB"
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
  if [ -n "$now" ] && [ -n "$seen" ]; then
    echo "[$(stamp)] YIELD: other GPU job(s)$now; stopping our run" | tee -a "$log"
    pkill -TERM -u "$me" -f "$ours"; sleep 15; pkill -KILL -u "$me" -f "$ours"
    exit 0
  fi
  seen=$now
  sleep 20
done
echo "[$(stamp)] run ended; watcher off" >> "$log"
