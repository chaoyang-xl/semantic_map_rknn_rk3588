#!/usr/bin/env bash
# Orange Pi 5 Plus / RK3588 semantic mapping performance monitor.
# Usage: sudo ./scripts/monitor_rk3588.sh [--interval SECONDS] [--once]

set -u
export LC_ALL=C

interval=2
run_once=false
no_clear=false

usage() {
  cat <<'EOF'
Usage: monitor_rk3588.sh [--interval SECONDS] [--once] [--no-clear] [--help]

  --interval, -n  Refresh interval in seconds (default: 2)
  --once           Print one sample and exit
  --no-clear       Append output instead of refreshing one dashboard screen
  --help, -h       Show this help

Run with sudo to read the RK3588 NPU debug load file.
EOF
}

while (($#)); do
  case "$1" in
    --interval|-n)
      [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
      interval=$2
      shift 2
      ;;
    --once)
      run_once=true
      shift
      ;;
    --no-clear)
      no_clear=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ $interval =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "interval must be a positive number" >&2
  exit 2
}

declare -A previous_total=()
declare -A previous_idle=()
declare -A cpu_usage=()

sample_cpu() {
  local name user nice system idle iowait irq softirq steal rest
  local cpu total idle_all delta_total delta_idle

  while read -r name user nice system idle iowait irq softirq steal rest; do
    [[ $name =~ ^cpu([0-9]+)$ ]] || continue
    cpu=${BASH_REMATCH[1]}
    iowait=${iowait:-0}
    irq=${irq:-0}
    softirq=${softirq:-0}
    steal=${steal:-0}
    idle_all=$((idle + iowait))
    total=$((user + nice + system + idle_all + irq + softirq + steal))

    if [[ -n ${previous_total[$cpu]+x} ]]; then
      delta_total=$((total - previous_total[$cpu]))
      delta_idle=$((idle_all - previous_idle[$cpu]))
      if ((delta_total > 0)); then
        cpu_usage[$cpu]=$(((100 * (delta_total - delta_idle) + delta_total / 2) / delta_total))
      else
        cpu_usage[$cpu]=0
      fi
    else
      cpu_usage[$cpu]=0
    fi
    previous_total[$cpu]=$total
    previous_idle[$cpu]=$idle_all
  done < /proc/stat
}

cpu_frequency_khz() {
  local cpu=$1 path
  path="/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq"
  [[ -r $path ]] && cat "$path" || printf '0\n'
}

print_cpu() {
  local cpu type freq governor cpu_count cpu_path
  cpu_count=$(nproc)
  echo "[CPU - interval utilization]"
  for ((cpu = 0; cpu < cpu_count; cpu++)); do
    cpu_path="/sys/devices/system/cpu/cpu${cpu}"
    if ((cpu_count == 8)); then
      ((cpu < 4)) && type="A55" || type="A76"
    else
      type="CPU"
    fi
    freq=$(cpu_frequency_khz "$cpu")
    governor="-"
    [[ -r "$cpu_path/cpufreq/scaling_governor" ]] && \
      governor=$(<"$cpu_path/cpufreq/scaling_governor")
    printf "  cpu%-2d %-3s  %4d MHz  %3d%%  %-11s\n" \
      "$cpu" "$type" "$((freq / 1000))" "${cpu_usage[$cpu]:-0}" "$governor"
  done
  printf "  load average: %s\n" "$(cut -d' ' -f1-3 /proc/loadavg)"
}

print_npu() {
  local load_file=/sys/kernel/debug/rknpu/load
  echo "[NPU - RK3588 three-core load]"
  if [[ -r $load_file ]]; then
    sed 's/^/  /' "$load_file"
  elif [[ -e $load_file ]]; then
    echo "  permission denied (run this script with sudo)"
  else
    echo "  unavailable (rknpu driver/debugfs is not exposed)"
  fi
}

print_gpu() {
  local device="" candidate freq load
  for candidate in /sys/class/devfreq/*; do
    [[ -e $candidate ]] || continue
    if [[ ${candidate##*/} == *gpu* ]]; then
      device=$candidate
      break
    fi
  done

  echo "[GPU - Mali-G610]"
  if [[ -n $device && -r $device/cur_freq ]]; then
    freq=$(<"$device/cur_freq")
    load="n/a"
    [[ -r $device/load ]] && load=$(<"$device/load")
    printf "  device: %s  freq: %d MHz  load: %s\n" \
      "${device##*/}" "$((freq / 1000000))" "$load"
  else
    echo "  unavailable"
  fi
}

print_temperatures() {
  local zone type temp
  echo "[Temperature]"
  for zone in /sys/class/thermal/thermal_zone*; do
    [[ -r $zone/type && -r $zone/temp ]] || continue
    type=$(<"$zone/type")
    temp=$(<"$zone/temp")
    awk -v label="$type" -v value="$temp" \
      'BEGIN { printf "  %-24s %5.1f C\n", label, value / 1000 }'
  done
}

print_memory_and_disk() {
  echo "[Memory / storage]"
  free -h | awk '
    /^Mem:/  {printf "  RAM  total:%-6s used:%-6s available:%-6s\n", $2, $3, $7}
    /^Swap:/ {printf "  Swap total:%-6s used:%-6s free:%-6s\n", $2, $3, $4}'
  df -h / | awk 'NR == 2 {
    printf "  Root total:%-6s used:%-6s available:%-6s usage:%s\n", $2, $3, $4, $5
  }'
}

print_processes() {
  echo "[Top processes]"
  printf "  %-7s %-22s %6s %6s %8s\n" PID COMMAND CPU% MEM% RSS
  ps -eo pid=,comm=,%cpu=,%mem=,rss= --sort=-%cpu | \
    awk -v self="$$" '$1 != self && $2 != "ps" && $2 != "awk" && shown < 6 {
      printf "  %-7s %-22.22s %6s %6s %7.1fM\n", $1, $2, $3, $4, $5/1024
      shown++
    }'
}

# Establish a baseline so the first screen is an interval measurement rather
# than the misleading average CPU utilization since boot.
sample_cpu
sleep 0.2

while true; do
  sample_cpu
  if ! $no_clear && ! $run_once; then
    printf '\033[H\033[2J'
  fi

  echo "============================================================"
  printf " RK3588 semantic mapping monitor   %s   uptime: %s\n" \
    "$(date '+%F %T')" "$(uptime -p 2>/dev/null || echo unknown)"
  echo "============================================================"
  print_cpu
  echo "------------------------------------------------------------"
  print_npu
  echo "------------------------------------------------------------"
  print_gpu
  echo "------------------------------------------------------------"
  print_temperatures
  echo "------------------------------------------------------------"
  print_memory_and_disk
  echo "------------------------------------------------------------"
  print_processes
  echo "============================================================"

  $run_once && break
  sleep "$interval"
done
