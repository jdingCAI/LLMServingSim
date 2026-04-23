# LLMServingSim — H200 Sensitivity Sweep

Minimal reproducibility fork of
[`casys-kaist/LLMServingSim`](https://github.com/casys-kaist/LLMServingSim)
containing only what is needed to reproduce one experiment:

**8× H200 (TP8) × Llama-3.1-405B-Instruct, 2 TB CPU (160 GB/s) vs 32 TB CXL (128 GB/s),
6 conversation counts × 2 memory configurations = 12 simulations.**

The full upstream simulator — evaluation suite, other hardware/model profiles,
alternate experiments, documentation — has been stripped.

## Prerequisites

- Linux, CUDA-capable host (PyTorch loads CUDA libs at import time)
- System protobuf (`protoc` at `/usr/bin/protoc`; `compile.sh` avoids Anaconda's)
- Python 3.8+ with PyTorch, scikit-learn, pandas, numpy, matplotlib, pyyaml
- `cmake`, `g++`, `make` for building the ASTRA-Sim C++ backend
- Submodules fetched: `git submodule update --init --recursive`

## 1. Build

```
./compile.sh
```

This pip-installs Chakra (from `astra-sim/extern/graph_frontend/chakra`) and
builds the ASTRA-Sim analytical backend at
`astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra`.

## 2. Generate datasets

The 6 workload JSONL files are not checked in (they exceed GitHub's size limit).
The generator is fully deterministic — the same command produces byte-identical
output on any machine.

```
cd dataset
for N in 50 100 150 200 250 300; do
  python generate_synthetic_multi_turn.py \
    --num-conversations $N --num-turns 10 \
    --output synthetic_${N}conv_10turns.jsonl
done
cd ..
```

Produces ~2.6 GB total under `dataset/synthetic_{N}conv_10turns.jsonl`.

## 3. Run the sweep

```
mkdir -p output/8H200
bash run_h200_sensitivity_sweep.sh > output/8H200/h200_sensitivity_sweep_log.txt 2>&1
```

12 simulations are emitted to `output/8H200/sensitivity_{2tb,cxl32tb}_{N}conv`
for N ∈ {50, 100, 150, 200, 250, 300}.

## 4. Generate plots and report

```
python analyze_h200_sensitivity_sweep.py
```

Produces the following under `output/8H200/`:

- `h200_sensitivity_mean_ttft.png`
- `h200_sensitivity_prefix_hit.png`
- `h200_sensitivity_ttft_and_hit_combined.png` (paper-format, 3.3"×1.2")

