#!/usr/bin/env bash
# Mở rộng bootstrap chính lên 5,000 lần mỗi họ. Chạy: setsid nohup bash run_round4b.sh > ../Result/logs/round4b.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
P=()
for fam in image tabular; do
  for s in 1000 2000 3000 4000; do
    python3 vr19_primary_bootstrap.py --family $fam --start $s --end $((s+1000)) > $L/vr19b_${fam}_${s}.log 2>&1 &
    P+=($!)
  done
done
wait "${P[@]}"
python3 vr19_primary_bootstrap.py --summarize > $L/vr19_summary.log 2>&1
python3 vr29_bootstrap_audit.py > $L/vr29_bootstrap_audit.log 2>&1
echo "ROUND4B DONE $(date)"
