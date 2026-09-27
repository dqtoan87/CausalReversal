#!/usr/bin/env bash
# Chạy lại vr34 sau khi vr33 xong (tránh tràn GPU).
cd "$(dirname "$0")"
until grep -q "vr33 done" ../Result/logs/round5.log; do sleep 60; done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python3 vr34_semisynth.py > ../Result/logs/vr34_semisynth.log 2>&1
echo "ROUND5B DONE $(date)"
