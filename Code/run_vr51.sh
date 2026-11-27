#!/usr/bin/env bash
# VR51 driver: primary bootstrap on the direct logit scale (image) and M2 without indeterminate biopsies (both families).
# Run from Code/:  setsid nohup bash run_vr51.sh > ../Result/logs/vr51_driver.log 2>&1 < /dev/null &
# Resumable: every lane skips replicates already stored in Result/vr51/{family}_{start}_{end}.json.
# Status:   bash run_vr51.sh status
set -uo pipefail
cd "$(dirname "$0")"
L=../Result/logs; mkdir -p $L ../Result/vr51
IMG_LANES=${IMG_LANES:-10}; TAB_LANES=${TAB_LANES:-5}; B=5000
if [ "${1:-}" = "status" ]; then
  python3 - <<'PY'
import json, glob
for fam in ("image", "tabular"):
    n = 0
    for f in glob.glob(f"../Result/vr51/{fam}_*.json"):
        try: n += len(json.load(open(f))["reps"])
        except Exception: pass
    print(fam, "seed fits stored:", n, "of 15000")
PY
  pgrep -af "vr51_direct_indeterminate" | grep -v pgrep; exit 0
fi
pids=()
step=$((B / IMG_LANES))
for i in $(seq 0 $((IMG_LANES - 1))); do
  s=$((i * step)); e=$((s + step))
  python3 vr51_direct_indeterminate.py --family image --start $s --end $e > $L/vr51_image_$s.log 2>&1 & pids+=($!)
done
step=$((B / TAB_LANES))
for i in $(seq 0 $((TAB_LANES - 1))); do
  s=$((i * step)); e=$((s + step))
  python3 vr51_direct_indeterminate.py --family tabular --start $s --end $e > $L/vr51_tabular_$s.log 2>&1 & pids+=($!)
done
python3 vr51_direct_indeterminate.py --points > $L/vr51_points.log 2>&1 & pids+=($!)
echo "lanes started: ${pids[*]}  $(date)"
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "lanes finished, failures=$fail  $(date)"
python3 vr51_direct_indeterminate.py --summarize > $L/vr51_summarize.log 2>&1 && echo "VR51 DONE $(date)" || echo "VR51 SUMMARIZE FAILED $(date)"
