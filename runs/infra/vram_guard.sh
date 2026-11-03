#!/usr/bin/env bash
# Shared VRAM budget guard (source this).
#   budget_start <run_dir> <batch> <floor_mib> <target_min_mib> <target_max_mib>
#   guard_watch  <pid> <run_dir> <floor_mib>    # kills + returns 1 if used < floor after 60s
vram_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
vram_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }

budget_start() {
  local dir="$1" batch="$2" floor="$3" tmin="$4" tmax="$5"
  local free; free=$(vram_free_mib)
  python3 - "$dir/vram_budget.json" "$batch" "$floor" "$tmin" "$tmax" "$free" <<'PY'
import json, sys, time
path, b, floor, tmin, tmax, free = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
json.dump({"batch": b, "floor_mib": floor, "target_min_mib": tmin,
           "target_max_mib": tmax, "free_at_start_mib": free,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(path, "w"), indent=2)
PY
  echo "VRAM_BUDGET batch=$batch floor=${floor}MiB target=${tmin}-${tmax}MiB free_at_start=${free}MiB"
  if [ "${free:-0}" -lt "$tmin" ]; then
    echo "VRAM_GUARD_FAIL raise: free_at_start=${free}MiB < target_min=${tmin}MiB"
    return 1
  fi
  return 0
}

guard_watch() {
  local pid="$1" dir="$2" floor="$3"
  local peak=0 t0 now u state rc=0
  t0=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -z "$state" ] || [ "${state#Z}" != "$state" ]; then
      break   # process gone or zombie: never treat that as "low VRAM"
    fi
    u=$(vram_used_mib); now=$(date +%s)
    [ "${u:-0}" -gt "$peak" ] && peak=$u
    if [ $((now - t0)) -ge 60 ] && [ "${u:-0}" -lt "$floor" ]; then
      kill -9 "$pid" 2>/dev/null
      python3 - "$dir/vram_budget.json" "$peak" "$u" <<'PY'
import json, sys
p, peak, u = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = json.load(open(p)); d.update({"peak_vram_mib": peak, "guarded_used_mib": u, "guard_failed": True})
json.dump(d, open(p, "w"), indent=2)
PY
      echo "VRAM_GUARD_FAIL raise: used=${u}MiB < floor=${floor}MiB after 60s (peak=${peak}MiB)"
      return 1
    fi
    sleep 1
  done
  python3 - "$dir/vram_budget.json" "$peak" <<'PY'
import json, sys
p, peak = sys.argv[1], int(sys.argv[2])
d = json.load(open(p)); d.update({"peak_vram_mib": peak, "peak_vram_gb": round(peak / 1024, 2)})
json.dump(d, open(p, "w"), indent=2)
PY
  echo "VRAM_PEAK used=${peak}MiB"
  return 0
}
