#!/usr/bin/env bash
# Vòng phân tích 3. Chạy từ Code/:  setsid nohup bash run_round3.sh > ../Result/logs/round3.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
( python3 vr19_primary_bootstrap.py --family tabular --start -3 --end 0 && python3 vr19_primary_bootstrap.py --family tabular --start 0 --end 1000 ) > $L/vr19_tabular.log 2>&1 &
P1=$!
( python3 vr19_primary_bootstrap.py --family image --start -3 --end 0 && python3 vr19_primary_bootstrap.py --family image --start 0 --end 1000 ) > $L/vr19_image.log 2>&1 &
P2=$!
( python3 vr21_dose_calibrated.py > $L/vr21_dose_calibrated.log 2>&1; python3 vr20_audit.py > $L/vr20_audit.log 2>&1 ) &
P3=$!
python3 vr22_pointwise_bridge.py > $L/vr22_pointwise_bridge.log 2>&1
wait $P1 $P2 $P3
python3 vr19_primary_bootstrap.py --summarize > $L/vr19_summary.log 2>&1
cd ../paper && python3 build_paper.py > ../Result/logs/build.log 2>&1 && python3 build_supplementary.py >> ../Result/logs/build.log 2>&1 && python3 check_split.py paper.md supplementary.md >> ../Result/logs/build.log 2>&1
echo "ROUND3 DONE $(date)"
