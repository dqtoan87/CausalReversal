#!/usr/bin/env bash
# Vòng phân tích 8. Chạy: setsid nohup bash run_round8.sh > ../Result/logs/round8.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
lane() { for i in 1 2 3; do "$@" && return 0; echo "retry $i: $*"; sleep 60; done; }
lane python3 vr41_marginal_site.py > $L/vr41_marginal_site.log 2>&1 &
lane python3 vr40_support_composition.py > $L/vr40_support_composition.log 2>&1 &
for r in "200 400" "400 600" "600 800" "800 1000"; do set -- $r
  lane python3 vr39_seed_variance.py --family tabular --start $1 --end $2 > $L/vr39_tab_$1.log 2>&1 &
done
for r in "200 600" "600 1000"; do set -- $r
  lane python3 vr39_seed_variance.py --family image --start $1 --end $2 > $L/vr39_img_$1.log 2>&1 &
done
wait
python3 vr39_seed_variance.py --summarize > $L/vr39_summarize.log 2>&1
echo "ROUND8 DONE $(date)"
