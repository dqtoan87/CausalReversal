#!/usr/bin/env bash
# Post-lock claim validation. Chạy từ Code/:
#   setsid nohup bash run_claim_validation.sh > ../Result/logs/claim_validation.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
L=../Result/logs
python3 vr12_claim_validation.py --parts a > $L/vr12_size_matched.log 2>&1 &
CPU=$!
python3 vr4_finetune.py --modes ft --regimes M0,M2 --seeds 3,4 > $L/vr4_extra_seeds.log 2>&1
python3 vr13_headswap.py > $L/vr13_headswap.log 2>&1
wait $CPU
echo "CLAIM VALIDATION DONE $(date)"
