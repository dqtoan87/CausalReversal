#!/usr/bin/env bash
# Phân tích bổ sung vòng 2. Chạy từ Code/:
#   setsid nohup bash run_round2.sh > ../Result/logs/round2.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
L=../Result/logs
python3 vr16_mechanism.py      > $L/vr16_mechanism.log 2>&1 &
P1=$!
python3 vr15_dose_response.py  > $L/vr15_dose_response.log 2>&1 &
P2=$!
python3 vr17_joint_bootstrap.py --B 200 > $L/vr17_joint_bootstrap.log 2>&1
wait $P1 $P2
echo "ROUND2 DONE $(date)"
