#!/usr/bin/env bash
# Mở rộng bootstrap ba seed chính (vr43) từ 1,000 lên 5,000 lần lặp: huấn luyện hai seed thêm (b+100000, b+200000)
# cho b = 1000..4999 trên đúng mẫu bệnh nhân của vr19, rồi tính lại vr43. KHÔNG chạy lại vr39 --summarize
# (Table S14 giữ phép so sánh trên 1,000 lần lặp đầu).
# Chạy: setsid nohup bash run_round12.sh > ../Result/logs/round12.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
lane() { for i in 1 2 3 4 5; do "$@" && return 0; echo "retry $i: $*"; sleep 120; done; }
for r in "1000 2000" "2000 3000" "3000 4000" "4000 5000"; do set -- $r
  lane python3 vr39_seed_variance.py --family tabular --start $1 --end $2 > $L/vr39x_tab_$1.log 2>&1 &
  lane python3 vr39_seed_variance.py --family image --start $1 --end $2 > $L/vr39x_img_$1.log 2>&1 &
done
wait
python3 vr43_three_seed_primary.py > $L/vr43_5000.log 2>&1
echo "ROUND12 DONE $(date)"
