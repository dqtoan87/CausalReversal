#!/usr/bin/env bash
# Chia lại nhánh bảng của vr39 thành 4 luồng; tổng hợp sau khi vòng 7 xong.
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
lane() { for i in 1 2 3; do "$@" && return 0; echo "retry $i: $*"; sleep 60; done; }
for r in "0 50" "50 100" "100 150" "150 200"; do
  set -- $r
  lane python3 vr39_seed_variance.py --family tabular --start $1 --end $2 > $L/vr39_tab_$1.log 2>&1 &
done
wait
until grep -q "ROUND7 DONE" $L/round7.log 2>/dev/null; do sleep 60; done
python3 vr39_seed_variance.py --summarize > $L/vr39_summarize.log 2>&1
echo "ROUND7B DONE $(date)"
