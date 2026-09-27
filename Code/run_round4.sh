#!/usr/bin/env bash
# Vòng phân tích 4. Chạy từ Code/:  setsid nohup bash run_round4.sh > ../Result/logs/round4.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python3 vr24_full_bootstrap.py --family tabular --start -3 --end 250 > $L/vr24_tab_a.log 2>&1 &
P1=$!
python3 vr24_full_bootstrap.py --family tabular --start 250 --end 500 > $L/vr24_tab_b.log 2>&1 &
P2=$!
python3 vr24_full_bootstrap.py --family image --start -3 --end 250 > $L/vr24_img_a.log 2>&1 &
P3=$!
python3 vr24_full_bootstrap.py --family image --start 250 --end 500 > $L/vr24_img_b.log 2>&1 &
P4=$!
python3 vr28_psi_stress.py > $L/vr28_psi_stress.log 2>&1 &
P5=$!
( python3 vr25_psi_sensitivity.py > $L/vr25_psi_sensitivity.log 2>&1; python3 vr26_dose_poisson.py > $L/vr26_dose_poisson.log 2>&1 ) &
P6=$!
python3 vr27_bridge_eiv.py > $L/vr27_bridge_eiv.log 2>&1
wait $P1 $P2 $P3 $P4 $P5 $P6
python3 vr24_full_bootstrap.py --summarize > $L/vr24_summary.log 2>&1
echo "ROUND4 DONE $(date)"
