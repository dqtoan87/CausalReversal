#!/usr/bin/env bash
# Khi các luồng ảnh của run_round12.sh xong, dừng các luồng bảng và chia phần còn lại thành 12 luồng (GPU được giải phóng),
# rồi tổng hợp vr39 (toàn bộ lần lặp) và tính lại vr43.
# Chạy: setsid nohup bash run_round12b.sh > ../Result/logs/round12b.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
L=../Result/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
while pgrep -f "vr39_seed_variance.py --family image" > /dev/null; do sleep 300; done
echo "image lanes done $(date)"
pkill -f run_round12.sh; sleep 5
pkill -f "vr39_seed_variance.py --family tabular"; sleep 20
TAILS=$(python3 - <<'PY'
import glob, json
done = set()
for f in glob.glob("../Result/vr39/tabular_*.json"):
    done |= set(json.load(open(f))["reps"])
rem = [b for b in range(1000, 5000) if not all(f"{b}|{o}" in done for o in (100000, 200000))]
k = 12; n = len(rem); out = []
for i in range(k):
    ch = rem[i * n // k:(i + 1) * n // k]
    if ch: out.append(f"{ch[0]}:{ch[-1] + 1}")
print(" ".join(out))
PY
)
echo "remaining tabular chunks: $TAILS"
lane() { for i in 1 2 3 4 5; do "$@" && return 0; echo "retry $i: $*"; sleep 120; done; }
for r in $TAILS; do s=${r%:*}; e=${r#*:}
  lane python3 vr39_seed_variance.py --family tabular --start $s --end $e > $L/vr39y_tab_$s.log 2>&1 &
done
wait
python3 vr39_seed_variance.py --summarize > $L/vr39_summarize_5000.log 2>&1
python3 vr43_three_seed_primary.py > $L/vr43_5000.log 2>&1
echo "ROUND12B DONE $(date)"
