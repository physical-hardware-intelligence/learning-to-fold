#!/usr/bin/env bash
# The folding model for one lab session, on the GPU box. Start it before a session; anyone at the rig
# can then press Prepare model in Phi Studio. It stops itself after IDLE_MIN minutes with no requests
# (every request writes a line to the server log), or when another user starts a GPU job over 1 GiB
# (seen twice, 30 s apart), so it never sits on the shared card for nobody.
#
#   bash fold_session.sh start [IDLE_MIN]   # default 30; prints READY once port 8000 answers
#   bash fold_session.sh stop               # the model off, the GPU free
#   bash fold_session.sh status
#
# The server is b4_serve.sh (its GPU check first, then Ilia's serve.py). It listens on every interface:
# it only answers move requests, but stop it when the session ends.
set -u
here=$(cd "$(dirname "$0")" && pwd)
tmux_bin=~/miniforge3/bin/tmux
session=fold
log=~/phi/logs/b4_server.log
events=~/phi/logs/fold_session.log
me=$(whoami)
stamp() { date '+%F %H:%M:%S'; }
running() { pgrep -u "$me" -f "scripts/serve.py" >/dev/null; }
listening() { ss -ltn 2>/dev/null | grep -q ":8000 "; }
idle_min() { echo $(( ($(date +%s) - $(stat -c %Y "$log")) / 60 )); }

case "${1:-status}" in
start)
  idle=${2:-30}
  if listening; then echo "READY: already running (last request $(idle_min) min ago)."; exit 0; fi
  "$tmux_bin" kill-session -t "$session" 2>/dev/null
  "$tmux_bin" new -d -s "$session" -n server "bash $here/b4_serve.sh 2>&1 | tee $log; exec bash"
  "$tmux_bin" new-window -d -t "$session" -n watch "bash $here/fold_session.sh watch $idle"
  echo "[$(stamp)] start (stops after $idle idle min)" >> "$events"
  echo "Starting the folding model (about a minute)..."
  for _ in $(seq 1 90); do listening && break; grep -q "NO-GO" "$log" 2>/dev/null && break; sleep 2; done
  if listening; then
    echo "READY: the folding model is up. It stops itself after $idle minutes with no requests."
  else
    echo "NOT STARTED. The server log ends with:"; tail -5 "$log"
    "$tmux_bin" kill-session -t "$session" 2>/dev/null; exit 1
  fi ;;
stop)
  pkill -TERM -u "$me" -f "scripts/serve.py"; sleep 3; pkill -KILL -u "$me" -f "scripts/serve.py" 2>/dev/null
  echo "[$(stamp)] stop" >> "$events"
  if listening; then echo "Port 8000 is still open: check the box by hand."; else echo "STOPPED: the model is off and the GPU is free."; fi
  "$tmux_bin" kill-session -t "$session" 2>/dev/null ;;   # last: stop may be running inside the session
status)
  if listening; then echo "RUNNING: last request $(idle_min) min ago."; else echo "OFF."; fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader ;;
watch)
  idle=${2:-30}; seen=""
  for _ in $(seq 1 90); do listening && break; sleep 2; done   # the server takes ~1 min to come up
  while running; do
    if [ "$(idle_min)" -ge "$idle" ]; then
      echo "[$(stamp)] no requests for $idle min" >> "$events"; bash "$here/fold_session.sh" stop >> "$events"; exit 0
    fi
    now=""
    while IFS=', ' read -r pid mem; do
      [ -z "$pid" ] && continue
      user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      [ -n "$user" ] && [ "$user" != "$me" ] && [ "${mem:-0}" -gt 1024 ] && now="$now $user:${mem}MiB"
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
    if [ -n "$now" ] && [ -n "$seen" ]; then
      echo "[$(stamp)] another GPU job:$now" >> "$events"; bash "$here/fold_session.sh" stop >> "$events"; exit 0
    fi
    seen=$now
    sleep 30
  done ;;
*) echo "usage: fold_session.sh start [IDLE_MIN] | stop | status"; exit 2 ;;
esac
