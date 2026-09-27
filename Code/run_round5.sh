#!/usr/bin/env bash
# Vòng phân tích 5. Chạy: setsid nohup bash run_round5.sh > ../Result/logs/round5.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=()
for r in "-3 330" "330 665" "665 1000"; do set -- $r; python3 vr30_q_bootstrap.py --family tabular --start $1 --end $2 > $L/vr30_tab_$2.log 2>&1 & P+=($!); done
for r in "-3 500" "500 1000"; do set -- $r; python3 vr30_q_bootstrap.py --family image --start $1 --end $2 > $L/vr30_img_$2.log 2>&1 & P+=($!); done
( python3 vr34_semisynth.py > $L/vr34_semisynth.log 2>&1; echo "vr34 done $(date)";
  python3 vr32_dose_nuisance.py > $L/vr32_dose_nuisance.log 2>&1; echo "vr32 done $(date)";
  python3 vr33_bridge_alt.py > $L/vr33_bridge_alt.log 2>&1; echo "vr33 done $(date)" ) &
Q=$!
wait "${P[@]}"
python3 vr30_q_bootstrap.py --summarize > $L/vr30_summary.log 2>&1
echo "vr30 done $(date)"
python3 vr36_subsample.py --family tabular --start 0 --end 150 > $L/vr36_tab_a.log 2>&1 &
A1=$!
python3 vr36_subsample.py --family tabular --start 150 --end 300 > $L/vr36_tab_b.log 2>&1 &
A2=$!
python3 vr36_subsample.py --family image --start 0 --end 300 > $L/vr36_img.log 2>&1
wait $A1 $A2
python3 vr36_subsample.py --summarize > $L/vr36_summary.log 2>&1
echo "vr36 done $(date)"
wait $Q
echo "ROUND5 DONE $(date)"
