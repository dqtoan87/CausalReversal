#!/usr/bin/env bash
# Supervisor for vr51: restarts missing lanes when the GPU has room, then summarizes. Resumable and safe to rerun.
# Run from Code/:  setsid nohup bash supervise_vr51.sh > ../Result/logs/vr51_supervisor.log 2>&1 < /dev/null &
# Stop:  touch ../Result/vr51/STOP     Status: bash run_vr51.sh status ; tail ../Result/logs/vr51_supervisor.log
set -uo pipefail
cd "$(dirname "$0")"
L=../Result/logs; OUT=../Result/vr51; MAXLANES=${MAXLANES:-10}; MINFREE=${MINFREE:-3000}
lanes() { for s in $(seq 0 500 4500); do echo "image $s $((s + 500))"; done; for s in $(seq 0 1000 4000); do echo "tabular $s $((s + 1000))"; done; }
complete() {  # family start end -> exit 0 if all 3*(end-start) seed fits are stored
  python3 -c "
import json,sys,os
p='$OUT/$1_$2_$3.json'
n=len(json.load(open(p))['reps']) if os.path.exists(p) else 0
sys.exit(0 if n >= 3*($3-$2) else 1)" 2>/dev/null
}
while true; do
  [ -f $OUT/STOP ] && { echo "stop requested $(date)"; exit 0; }
  todo=0; running=$(pgrep -fc "vr51_direct_indeterminate.py --family")
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  while read fam s e; do
    complete $fam $s $e && continue
    todo=$((todo + 1))
    pgrep -f "vr51_direct_indeterminate.py --family $fam --start $s --end $e" > /dev/null && continue
    if [ "$running" -lt "$MAXLANES" ] && [ "$free" -gt "$MINFREE" ]; then
      python3 vr51_direct_indeterminate.py --family $fam --start $s --end $e >> $L/vr51_${fam}_$s.log 2>&1 &
      echo "$(date) started $fam $s-$e (free ${free} MiB)"; running=$((running + 1)); free=$((free - 1500)); sleep 60
    fi
  done < <(lanes)
  if [ "$todo" -eq 0 ]; then
    python3 vr51_direct_indeterminate.py --summarize > $L/vr51_summarize.log 2>&1 && echo "VR51 DONE $(date)" || echo "VR51 SUMMARIZE FAILED $(date)"
    exit 0
  fi
  sleep 600
done
