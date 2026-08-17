#!/bin/bash
#SBATCH --job-name=obs2_validate_l4
#SBATCH --partition=long
#SBATCH --nodelist=ganymede
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=logs/obs2_validate_l4_%j.out
#SBATCH --error=logs/obs2_validate_l4_%j.err

# ==============================================================================
# Obs2 motivation hardware spot-validation - L4 fill-in cells only.
#
# Fills the two operating points the predictive model EXTRAPOLATED to (never
# measured in calibration), which underlie the counter-intuitive Fig.2 choices:
#
#   EXP2  L4 short-prefill frequency threshold:
#         il=128, ol=32, TP=1, rate=1, freq in {1410,1620,1755}
#         (calibration already has 2040@r1=251ms PASS; this brackets the threshold)
#
#   EXP4  L4 decode at the selected 1200 MHz under the decode-heavy load:
#         il=32, ol=1024, TP=4, rate in {1,4}, freq in {1200,1410}
#         (calibration only had 1620@r1 for this cell)
#
# Reuses the exact phase2_characterization_l4.sh invocations (clocks via
# nvidia-smi -ac, vLLM OpenAI server, gpu_monitor.py NVML energy, vllm bench
# serve). Emits rows in master_results.csv schema -> concatenate before refit.
# ==============================================================================

set -uo pipefail

# ---- config (matches phase2_characterization_l4.sh) ----
MODEL="mistralai/Mistral-7B-v0.1"
PORT=8000
GPU_TYPE="L4"
SUPPORTED_MEM_FREQ="6251"          # L4 supported memory clock
MAX_MODEL_LEN=9216
MONITOR_SCRIPT="gpu_monitor.py"
MONITOR_INTERVAL=0.05
NUM_PROMPTS=200
NUM_WARMUPS=10
RUNS=3
VLLM_LOG="vllm_server_obs2val.log"

OUTPUT_DIR="Obs2_Validation_L4"
mkdir -p "$OUTPUT_DIR" logs
MASTER_CSV="$OUTPUT_DIR/obs2_validation_results.csv"
if [ ! -f "$MASTER_CSV" ]; then
  echo "step,gpu_type,gpu_freq_mhz,mem_freq_mhz,tp_degree,input_len,output_len,request_rate,run,num_prompts,benchmark_duration_s,request_throughput_rps,output_token_throughput_tps,total_token_throughput_tps,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_itl_ms,median_itl_ms,p99_itl_ms,avg_power_w,min_power_w,max_power_w,energy_j,avg_gpu_util_pct,avg_mem_util_pct,monitor_samples" > "$MASTER_CSV"
fi

# Fill-in cells: "step:tp:il:ol:rate:freq"
CELLS=(
  "exp2_thresh:1:128:32:1:1410"
  "exp2_thresh:1:128:32:1:1620"
  "exp2_thresh:1:128:32:1:1755"
  "exp4_decode:4:32:1024:1:1200"
  "exp4_decode:4:32:1024:4:1200"
  "exp4_decode:4:32:1024:1:1410"
  "exp4_decode:4:32:1024:4:1410"
)

# ==============================================================================
# helpers (copied from phase2_characterization_l4.sh)
# ==============================================================================
SERVER_PID=""
MONITOR_PIDS=()

cleanup_monitors() {
  if [ ${#MONITOR_PIDS[@]} -eq 0 ]; then return; fi
  for PID in "${MONITOR_PIDS[@]}"; do kill -TERM "$PID" 2>/dev/null || true; done
  sleep 1
  for PID in "${MONITOR_PIDS[@]}"; do kill -KILL "$PID" 2>/dev/null || true; done
  MONITOR_PIDS=(); sync; sleep 0.5
}

stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true
  fi
  pkill -TERM -f "vllm.entrypoints" 2>/dev/null || true
  pkill -TERM -f "vllm[._-]worker"  2>/dev/null || true
  sleep 5
  pkill -KILL -f "vllm.entrypoints" 2>/dev/null || true
  pkill -KILL -f "vllm[._-]worker"  2>/dev/null || true
  local w=0
  while lsof -i:$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    sleep 2; w=$((w+2))
    if [ $w -ge 30 ]; then lsof -i:$PORT -sTCP:LISTEN -t | xargs -r kill -KILL 2>/dev/null || true; sleep 3; break; fi
  done
  SERVER_PID=""
}

reset_clocks() { sudo nvidia-smi -rac 2>/dev/null || true; }

cleanup_all() { echo "--- cleanup ---"; cleanup_monitors; stop_server; reset_clocks; }
trap cleanup_all EXIT INT TERM

set_gpu_freq() {
  local freq=$1 tp=$2
  echo "  Setting GPU clocks to ${SUPPORTED_MEM_FREQ},${freq} on $tp GPU(s)..."
  for ((g=0; g<tp; g++)); do sudo nvidia-smi -i $g -ac "${SUPPORTED_MEM_FREQ},${freq}"; done
  sleep 2
}

start_server() {
  local tp=$1
  stop_server
  local gpu_list=""
  for ((g=0; g<tp; g++)); do [ -n "$gpu_list" ] && gpu_list+=","; gpu_list+="$g"; done
  export CUDA_VISIBLE_DEVICES="$gpu_list"
  echo "  Starting vLLM: TP=$tp GPUs=$gpu_list"
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host localhost --port "$PORT" \
    --tensor-parallel-size "$tp" --max-model-len "$MAX_MODEL_LEN" \
    --disable-log-requests > "$VLLM_LOG" 2>&1 &
  SERVER_PID=$!
  # health check
  local waited=0
  until curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
    sleep 5; waited=$((waited+5))
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "  ERROR: server died"; tail -30 "$VLLM_LOG"; return 1; fi
    if [ $waited -ge 600 ]; then echo "  ERROR: server not ready after 600s"; tail -30 "$VLLM_LOG"; return 1; fi
  done
  echo "  server ready after ${waited}s"
}

start_monitors() {
  local prefix=$1 tp=$2
  MONITOR_PIDS=()
  for ((g=0; g<tp; g++)); do
    $(which python3) "$MONITOR_SCRIPT" --monitor --interval "$MONITOR_INTERVAL" --output "${prefix}_gpu${g}.csv" --gpu-id $g &
    MONITOR_PIDS+=($!)
  done
  sleep 2
}

parse_bench_output() {
  local f=$1
  local dur=$(grep "Benchmark duration" "$f" | awk '{print $NF}')
  local rt=$(grep "Request throughput" "$f" | awk '{print $NF}')
  local ott=$(grep "Output token throughput" "$f" | head -1 | awk '{print $NF}')
  local ttt=$(grep "Total token throughput" "$f" | awk '{print $NF}')
  local mttft=$(grep "Mean TTFT" "$f" | awk '{print $NF}')
  local medttft=$(grep "Median TTFT" "$f" | awk '{print $NF}')
  local p99ttft=$(grep "P99 TTFT" "$f" | awk '{print $NF}')
  local mtpot=$(grep "Mean TPOT" "$f" | awk '{print $NF}')
  local medtpot=$(grep "Median TPOT" "$f" | awk '{print $NF}')
  local p99tpot=$(grep "P99 TPOT" "$f" | awk '{print $NF}')
  local mitl=$(grep "Mean ITL" "$f" | awk '{print $NF}')
  local meditl=$(grep "Median ITL" "$f" | awk '{print $NF}')
  local p99itl=$(grep "P99 ITL" "$f" | awk '{print $NF}')
  echo "${dur:-NA},${rt:-NA},${ott:-NA},${ttt:-NA},${mttft:-NA},${medttft:-NA},${p99ttft:-NA},${mtpot:-NA},${medtpot:-NA},${p99tpot:-NA},${mitl:-NA},${meditl:-NA},${p99itl:-NA}"
}

compute_power_energy() {
  local prefix=$1 tp=$2 start_ts=$3 end_ts=$4
  local tap=0 tmin=999999 tmax=0 te=0 tgu=0 tmu=0 tn=0
  for ((g=0; g<tp; g++)); do
    local lf="${prefix}_gpu${g}.csv"
    if [ ! -f "$lf" ]; then echo "0,0,0,0,0,0,0"; return; fi
    # monitor CSV cols: timestamp,datetime,power_w,total_energy_mj,gpu_freq,mem_freq,temp,gpu_util,mem_util
    local r
    r=$(awk -F, -v s="$start_ts" -v e="$end_ts" '
      BEGIN{c=0;sp=0;mn=999999;mx=0;fe=-1;le=-1;sg=0;sm=0}
      NR>1 && $1>=s && $1<=e {p=$3+0;en=$4+0;gu=$8+0;mu=$9+0;sp+=p;if(p<mn)mn=p;if(p>mx)mx=p;if(fe==-1)fe=en;le=en;sg+=gu;sm+=mu;c++}
      END{if(c>0){printf "%.2f,%.2f,%.2f,%.3f,%.1f,%.1f,%d",sp/c,mn,mx,(le-fe)/1000.0,sg/c,sm/c,c}else{print "0,0,0,0,0,0,0"}}' "$lf")
    local ap=$(echo "$r"|cut -d, -f1) mp=$(echo "$r"|cut -d, -f2) xp=$(echo "$r"|cut -d, -f3) en=$(echo "$r"|cut -d, -f4) gu=$(echo "$r"|cut -d, -f5) mu=$(echo "$r"|cut -d, -f6) n=$(echo "$r"|cut -d, -f7)
    tap=$(awk "BEGIN{print $tap+$ap}"); te=$(awk "BEGIN{print $te+$en}")
    tgu=$(awk "BEGIN{print $tgu+$gu}"); tmu=$(awk "BEGIN{print $tmu+$mu}"); tn=$((tn+n))
    awk "BEGIN{exit !($mp<$tmin)}" && tmin=$mp
    awk "BEGIN{exit !($xp>$tmax)}" && tmax=$xp
  done
  if [ "$tp" -gt 0 ]; then tgu=$(awk "BEGIN{printf \"%.1f\",$tgu/$tp}"); tmu=$(awk "BEGIN{printf \"%.1f\",$tmu/$tp}"); fi
  echo "${tap},${tmin},${tmax},${te},${tgu},${tmu},${tn}"
}

max_conc_for() {
  local rate=$1 il=$2 tp=$3
  local mc=32
  if   [ "$rate" -le 5 ]; then mc=32
  elif [ "$rate" -le 20 ]; then mc=128
  else mc=256; fi
  local budget=$((8*1024*tp))
  local per=$(awk "BEGIN{printf \"%.0f\",0.128*$il}")
  if [ "$per" -gt 0 ]; then local cap=$((budget/per)); [ "$cap" -lt "$mc" ] && mc=$cap; [ "$mc" -lt 4 ] && mc=4; fi
  echo "$mc"
}

# ==============================================================================
# main
# ==============================================================================
for cell in "${CELLS[@]}"; do
  IFS=: read -r step tp il ol rate freq <<< "$cell"
  echo "=========================================================="
  echo "CELL step=$step tp=$tp il=$il ol=$ol rate=$rate freq=$freq"
  set_gpu_freq "$freq" "$tp"
  if ! start_server "$tp"; then echo "  skipping cell (server failed)"; continue; fi
  mc=$(max_conc_for "$rate" "$il" "$tp")
  for ((run=1; run<=RUNS; run++)); do
    label="${step}_f${freq}_tp${tp}_i${il}_o${ol}_r${rate}_run${run}"
    bench_file="$OUTPUT_DIR/bench_${label}.txt"
    mon_prefix="$OUTPUT_DIR/mon_${label}"
    echo "  [$label] mc=$mc"
    start_monitors "$mon_prefix" "$tp"
    start_ts=$(python3 -c "import time;print(f'{time.time():.6f}')")
    vllm bench serve --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
      --num-prompts "$NUM_PROMPTS" --request-rate "$rate" --dataset-name random \
      --random-input-len "$il" --random-output-len "$ol" \
      --max-concurrency "$mc" --num-warmups "$NUM_WARMUPS" > "$bench_file" 2>&1
    end_ts=$(python3 -c "import time;print(f'{time.time():.6f}')")
    cleanup_monitors
    bm=$(parse_bench_output "$bench_file")
    pm=$(compute_power_energy "$mon_prefix" "$tp" "$start_ts" "$end_ts")
    echo "${step},${GPU_TYPE},${freq},${SUPPORTED_MEM_FREQ},${tp},${il},${ol},${rate},${run},${NUM_PROMPTS},${bm},${pm}" >> "$MASTER_CSV"
    echo "  [$label] done"
    sleep 3
  done
  stop_server
done
reset_clocks
echo "ALL DONE -> $MASTER_CSV"
