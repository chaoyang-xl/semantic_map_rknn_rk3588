#!/usr/bin/env bash
# Orange Pi 5 Plus / RK3588 semantic mapping performance monitor.
# Usage: sudo ./scripts/monitor_rk3588.sh [--interval SECONDS] [--once]

set -u
export LC_ALL=C

interval=2
run_once=false
no_clear=false
dashboard_active=false

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
  local cpu freq cpu_count
  cpu_count=$(nproc)
  echo "[CPU - interval utilization]"
  for ((cpu = 0; cpu < cpu_count; cpu++)); do
    freq=$(cpu_frequency_khz "$cpu")
    ((cpu % 4 == 0)) && printf "  "
    printf "c%d:%3d%%/%4dM  " \
      "$cpu" "${cpu_usage[$cpu]:-0}" "$((freq / 1000))"
    if ((cpu % 4 == 3 || cpu == cpu_count - 1)); then
      printf "\n"
    fi
  done
  printf "  load average: %s\n" "$(cut -d" " -f1-3 /proc/loadavg)"
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
  local zone type temp max_temp=-1 max_type="unavailable"
  echo "[Temperature]"
  for zone in /sys/class/thermal/thermal_zone*; do
    [[ -r $zone/type && -r $zone/temp ]] || continue
    type=$(<"$zone/type")
    temp=$(<"$zone/temp")
    if ((temp > max_temp)); then
      max_temp=$temp
      max_type=$type
    fi
  done
  if ((max_temp >= 0)); then
    awk -v label="$max_type" -v value="$max_temp" \
      "BEGIN { printf \"  hottest: %-20s %5.1f C\\n\", label, value / 1000 }"
  else
    echo "  unavailable"
  fi
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
    awk -v self="$$" '$1 != self && $2 != "ps" && $2 != "awk" && shown < 4 {
      printf "  %-7s %-22.22s %6s %6s %7.1fM\n", $1, $2, $3, $4, $5/1024
      shown++
    }'
}

# Establish a baseline so the first screen is an interval measurement rather
# than the misleading average CPU utilization since boot.
sample_cpu
sleep 0.2

restore_terminal() {
  if $dashboard_active; then
    printf "\033[?25h\033[?1049l"
    dashboard_active=false
  fi
}

if ! $no_clear && ! $run_once && [[ -t 1 ]]; then
  dashboard_active=true
  trap restore_terminal EXIT
  trap "restore_terminal; exit 0" INT TERM HUP
  printf "\033[?1049h\033[?25l"
elif [[ ! -t 1 ]]; then
  no_clear=true
fi

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
