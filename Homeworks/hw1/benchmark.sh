#!/bin/bash

# Check usage
if [ -z "$1" ]; then
    echo "Usage: ./benchmark_compare.sh <path/to/matrix.mtx>"
    exit 1
fi

MATRIX=$1
THREADS_LIST="1 2 4 8 16"
RUNS=5  

echo "========================================================"
echo "BENCHMARK STARTING: $MATRIX"
echo "System: 4 Sockets, 15 Cores/Socket (Total 60 Threads)"
echo "Runs per config: $RUNS"
echo "========================================================"

export OMP_PLACES=cores

# --- FUNCTION TO RUN MEASUREMENTS ---
run_measurements() {
    local T=$1
    local STRAT=$2
    local CMD=$3
    
    echo "--------------------------------------------------------"
    echo "Threads: $T | Strategy: $STRAT"

    local TOTAL_GFLOPS=0
    local TOTAL_TIME=0
    local TOTAL_PREPROCESS=0

    for (( i=1; i<=RUNS; i++ ))
    do
        OUTPUT=$($CMD 2>&1)
        
        # Değerleri al
        GFLOPS=$(echo "$OUTPUT" | grep -o "gflops=[0-9.]*" | cut -d= -f2)
        TIME=$(echo "$OUTPUT" | grep -o "time_sec=[0-9.]*" | cut -d= -f2)
        CHECKSUM=$(echo "$OUTPUT" | grep -o "checksum=[0-9.]*" | cut -d= -f2)
        
        # Preprocessing süresini al
        PRE_TIME=$(echo "$OUTPUT" | grep -o "Pre Process: n=[0-9.]*" | cut -d= -f2)
        
        # EĞER PRE_TIME BOŞSA (Baseline modu), 0 OLARAK AYARLA
        if [ -z "$PRE_TIME" ]; then
            PRE_TIME="0"
        fi
        
        if [ -z "$GFLOPS" ]; then
            echo "  [Run $i] ERROR: Could not parse output!"
            continue
        fi
        
        # Her turu ekrana bas (PreProc dahil)
        echo "  [Run $i] Time: $TIME s | GFLOPS: $GFLOPS | Checksum: $CHECKSUM | PreProc: $PRE_TIME s"

        # Toplamlara ekle
        TOTAL_GFLOPS=$(echo "$TOTAL_GFLOPS + $GFLOPS" | bc)
        TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
        TOTAL_PREPROCESS=$(echo "$TOTAL_PREPROCESS + $PRE_TIME" | bc)
    done

    # Ortalamaları hesapla
    AVG_GFLOPS=$(echo "scale=4; $TOTAL_GFLOPS / $RUNS" | bc)
    AVG_TIME=$(echo "scale=6; $TOTAL_TIME / $RUNS" | bc)
    AVG_PRE=$(echo "scale=6; $TOTAL_PREPROCESS / $RUNS" | bc)
    
    # Sonuç satırı (Avg PreProc dahil)
    echo ">>> RESULT ($EXE_NAME, T=$T, $STRAT): Avg Time = $AVG_TIME s | Avg GFLOPS = $AVG_GFLOPS | Avg PreProc = $AVG_PRE s"
}

# --- MAIN LOOP ---

for EXE_NAME in "spmv_base" "spmv_opt"
do
    if [ "$EXE_NAME" == "spmv_base" ]; then
        echo -e "\n########################################################"
        echo ">>> MODE: BASELINE (No Preprocessing)"
        echo "########################################################"
    else
        echo -e "\n########################################################"
        echo ">>> MODE: OPTIMIZED (RCM / Degree Sort)"
        echo "########################################################"
    fi

    for T in $THREADS_LIST
    do
        # 1. STANDARD LOGIC (1-8 Compact)
        if [ "$T" -le 8 ]; then
            export OMP_PROC_BIND=close
            export OMP_NUM_THREADS=$T
            CMD="numactl --cpunodebind=0 --membind=0 ./$EXE_NAME $MATRIX"
            run_measurements $T "COMPACT (Socket 0)" "$CMD"
            
        elif [ "$T" -eq 16 ]; then
            # 16 THREAD - DOUBLE TEST
            
            # CASE A: 4 Sockets (Max Bandwidth)
            export OMP_PROC_BIND=spread
            export OMP_NUM_THREADS=$T
            CMD="numactl --interleave=all ./$EXE_NAME $MATRIX"
            run_measurements $T "SCATTER (4 Sockets: 4x4)" "$CMD"
            
            # CASE B: 2 Sockets (Low Latency)
            export OMP_PROC_BIND=spread 
            export OMP_NUM_THREADS=$T
            CMD="numactl --cpunodebind=0,1 --interleave=0,1 ./$EXE_NAME $MATRIX"
            run_measurements $T "BALANCED (2 Sockets: 8x8)" "$CMD"
        fi
    done
done