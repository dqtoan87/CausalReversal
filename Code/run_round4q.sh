#!/usr/bin/env bash
# Hàng đợi vòng 4 (GPU dùng chung chỉ còn ~10 GB). Chạy: setsid nohup bash run_round4q.sh > ../Result/logs/round4q.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
waitpid() { while kill -0 "$1" 2>/dev/null; do sleep 60; done; }
V27=$(pgrep -f "python3 vr27_bridge_eiv.py" | head -1)
V24=$(pgrep -f "python3 vr24_full_bootstrap.py --family")
( [ -n "$V27" ] && waitpid $V27; python3 vr25_psi_sensitivity.py > $L/vr25_psi_sensitivity.log 2>&1; python3 vr26_dose_poisson.py > $L/vr26_dose_poisson.log 2>&1; echo "vr25/26 done $(date)" ) &
Q=$!
for p in $V24; do waitpid $p; done
python3 vr24_full_bootstrap.py --summarize > $L/vr24_summary.log 2>&1
echo "vr24 done $(date)"
for batch in "1000 2000" "3000 4000"; do
  P=()
  for s in $batch; do
    for fam in image tabular; do
      python3 vr19_primary_bootstrap.py --family $fam --start $s --end $((s+1000)) >> $L/vr19b_${fam}_${s}.log 2>&1 &
      P+=($!)
    done
  done
  wait "${P[@]}"
  echo "vr19 batch $batch done $(date)"
done
wait $Q
python3 vr19_primary_bootstrap.py --summarize > $L/vr19_summary.log 2>&1
python3 vr29_bootstrap_audit.py > $L/vr29_bootstrap_audit.log 2>&1
echo "ROUND4 DONE $(date)"
