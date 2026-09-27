#!/usr/bin/env bash
# Vòng phân tích 6. Chạy: setsid nohup bash run_round6.sh > ../Result/logs/round6.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python3 vr35_marginal_rr.py > $L/vr35_marginal_rr.log 2>&1 &
P1=$!
python3 vr34_semisynth.py mlp > $L/vr34_semisynth_mlp.log 2>&1 &
P2=$!
python3 vr34_semisynth.py gbm > $L/vr34_semisynth_gbm.log 2>&1 &
P3=$!
( python3 vr38_logit_tail.py > $L/vr38_logit_tail.log 2>&1; python3 vr37_overlap_sigma.py > $L/vr37_overlap_sigma.log 2>&1 ) &
P4=$!
wait $P1 $P2 $P3 $P4
echo "ROUND6 DONE $(date)"
