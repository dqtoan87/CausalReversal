#!/usr/bin/env bash
cd "$(dirname "$0")"
until grep -q "ROUND5B DONE" ../Result/logs/round5b.log 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python3 vr32_dose_nuisance.py > ../Result/logs/vr32b_dose_nuisance.log 2>&1
echo "ROUND5C DONE $(date)"
