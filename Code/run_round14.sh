#!/usr/bin/env bash
# Liều chọn mẫu với validation cố định và số ngẫu nhiên chung (vr46).
# Chạy: setsid nohup bash run_round14.sh > ../Result/logs/round14.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
lane() { for i in 1 2 3 4 5; do "$@" && return 0; echo "retry $i: $*"; sleep 120; done; }
for f in tabular image; do
  lane python3 vr46_dose_fixed_val.py --family $f --cells color_variegation:320 size:320 lesion_skin_contrast:320 > $L/vr46_${f}_a.log 2>&1 &
  lane python3 vr46_dose_fixed_val.py --family $f --cells color_variegation:1000 color_variegation:3000 > $L/vr46_${f}_b.log 2>&1 &
done
wait
python3 vr46_dose_fixed_val.py --summarize > $L/vr46_summarize.log 2>&1
echo "ROUND14 DONE $(date)"
