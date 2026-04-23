#!/bin/bash
# H200 Sensitivity Sweep: Llama-3.1-405B-Instruct TP8
# 6 conv counts × 2 memory configs = 12 simulations
# Configs: 2TB CPU (160 GB/s), 32TB CXL (128 GB/s)

set -e

mkdir -p output/8H200

COMMON_ARGS="--fp 16 --block-size 256 --enable-prefix-caching --enable-prefix-sharing --enable-attn-prediction --max-num-batched-tokens 131072 --max-batch 2 --log-interval 1.0 --log-level WARNING"

for N in 50 100 150 200 250 300; do
    NUM_REQ=$((N * 10))

    echo "=============================="
    echo "[$N convs] 2TB CPU DRAM ($NUM_REQ requests)"
    echo "=============================="
    python main.py \
        --cluster-config 'cluster_config/h200_2tb_cpu_bw160.json' \
        --prefix-storage CPU \
        $COMMON_ARGS \
        --dataset "dataset/synthetic_${N}conv_10turns.jsonl" \
        --output "output/8H200/sensitivity_2tb_${N}conv" \
        --num-req "$NUM_REQ"
    echo "[$N convs] 2TB exit code: $?"
    echo ""

    echo "=============================="
    echo "[$N convs] 32TB CXL ($NUM_REQ requests)"
    echo "=============================="
    python main.py \
        --cluster-config 'cluster_config/h200_32tb_cxl_bw128.json' \
        --prefix-storage CXL \
        $COMMON_ARGS \
        --dataset "dataset/synthetic_${N}conv_10turns.jsonl" \
        --output "output/8H200/sensitivity_cxl32tb_${N}conv" \
        --num-req "$NUM_REQ"
    echo "[$N convs] CXL-32TB exit code: $?"
    echo ""
done

echo "=============================="
echo "All 12 simulations complete."
echo "=============================="
