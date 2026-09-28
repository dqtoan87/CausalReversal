#!/usr/bin/env bash
# VILRR pipeline (proposal_v3.md). Chạy từ Code/:
#   setsid nohup bash run_vr_pipeline.sh > ../Result/logs/vr_pipeline.log 2>&1 < /dev/null &
# Bước nào đã có đầu ra JSON thì bỏ qua; vr4 bỏ qua run đã có .npz. Nếu vr2/vr4 đang chạy từ trước
# (đã khởi động riêng), script chờ chúng thay vì chạy lại.
set -uo pipefail
cd "$(dirname "$0")"
L=../Result/logs; RS=../Result
wait_for() { while pgrep -f "python3 $1" > /dev/null; do sleep 30; done; }

[ -f $RS/vr1_reversal_theory.json ] || python3 vr1_reversal_theory.py > $L/vr1_reversal_theory.log 2>&1
[ -f $RS/vr6_pad_boundary.json ]    || python3 vr6_pad_boundary.py    > $L/vr6_pad_boundary.log 2>&1

wait_for vr2_phase_diagram.py
[ -f $RS/vr2_phase_diagram.json ] || python3 vr2_phase_diagram.py > $L/vr2_phase_diagram.log 2>&1 &
wait_for vr4_finetune.py
[ -f $RS/vr4_finetune.json ] || python3 vr4_finetune.py >> $L/vr4_finetune.log 2>&1
wait
python3 vr5_representation.py > $L/vr5_representation.log 2>&1
echo "PIPELINE DONE $(date)"
