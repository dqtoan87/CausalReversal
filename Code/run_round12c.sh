#!/usr/bin/env bash
# Chờ run_round12b.sh xong, rồi dựng lại bài và phụ lục từ JSON mới và chạy kiểm tra nhất quán.
# Chạy: setsid nohup bash run_round12c.sh > ../Result/logs/round12c.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
until grep -q "ROUND12B DONE" ../Result/logs/round12b.log 2>/dev/null; do sleep 300; done
cd ../paper
python3 build_paper.py && python3 build_supplementary.py && python3 check_split.py paper.md supplementary.md
echo "ROUND12C DONE $(date)"
