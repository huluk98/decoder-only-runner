#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NPROC_PER_NODE=8 \
  SPARSITY_GPU_IDS=0,1,2,3,4,5,6,7 \
  bash run_linear_sparsity_revision_from_base.sh /PATH/TO/MY/DECODER_SLM_CHECKPOINT

Optional:
  OUTPUT_ROOT=outputs/decoder_pruning_full_matrix
  OUTPUT_JSON=$OUTPUT_ROOT/all_sparsity_results.json
  DRY_RUN=1

This writes a decoder_pruning_full_matrix JSON with 20 planned rows:
  2 dense baselines
  14 one-shot pruning rows
  4 progressive magnitude rows
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

INPUT_CHECKPOINT="$1"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/decoder_pruning_full_matrix}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_ROOT/all_sparsity_results.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SPARSITY_GPU_IDS="${SPARSITY_GPU_IDS:-0,1,2,3,4,5,6,7}"
TRAINING_DATASET="${TRAINING_DATASET:-data/scenic/SCENIC_full_training_dataset.json}"
CONTRASTIVE_TRAINING_DATASET="${CONTRASTIVE_TRAINING_DATASET:-data/scenic/SCENIC_full_anchor_positive_negative.json}"
BENCHMARK_DATASET="${BENCHMARK_DATASET:-data/benchmarks/iot_instruction_benchmark_200.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export SPARSITY_GPU_IDS

args=(
  -m decoder_only.full_matrix
  "$INPUT_CHECKPOINT"
  --output-root "$OUTPUT_ROOT"
  --output-json "$OUTPUT_JSON"
  --training-dataset "$TRAINING_DATASET"
  --contrastive-training-dataset "$CONTRASTIVE_TRAINING_DATASET"
  --benchmark "$BENCHMARK_DATASET"
  --nproc-per-node "$NPROC_PER_NODE"
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES"
  --sparsity-gpu-ids "$SPARSITY_GPU_IDS"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

"$PYTHON_BIN" "${args[@]}"
