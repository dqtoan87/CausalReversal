#!/usr/bin/env bash
# Vòng phân tích 7. Chạy: setsid nohup bash run_round7.sh > ../Result/logs/round7.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
lane() {  # chạy lại tối đa 3 lần nếu lỗi (ví dụ hết bộ nhớ GPU); các script tự tiếp tục từ file dở
  for i in 1 2 3; do "$@" && return 0; echo "retry $i: $*"; sleep 60; done
}
lane python3 vr34_semisynth.py local > $L/vr34_semisynth_local.log 2>&1 &
P1=$!
lane python3 vr39_seed_variance.py --family tabular --start 0 --end 100 > $L/vr39_tab_a.log 2>&1 &
P2=$!
lane python3 vr39_seed_variance.py --family tabular --start 100 --end 200 > $L/vr39_tab_b.log 2>&1 &
P3=$!
lane python3 vr39_seed_variance.py --family image --start 0 --end 100 > $L/vr39_img_a.log 2>&1 &
P4=$!
lane python3 vr39_seed_variance.py --family image --start 100 --end 200 > $L/vr39_img_b.log 2>&1 &
P5=$!
wait $P1 $P2 $P3 $P4 $P5
python3 vr39_seed_variance.py --summarize > $L/vr39_summarize.log 2>&1
echo "ROUND7 DONE $(date)"
