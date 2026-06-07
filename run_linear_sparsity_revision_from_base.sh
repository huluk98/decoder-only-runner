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
  MODEL_KIND=hf
  LOCAL_ONLY=1
  CHECKPOINT_DIAGNOSE=1
  CHECKPOINT_LOAD_MODEL=1
  LOG_DIR=$OUTPUT_ROOT/logs
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
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
CHECKPOINT_DIAGNOSE="${CHECKPOINT_DIAGNOSE:-1}"
CHECKPOINT_LOAD_MODEL="${CHECKPOINT_LOAD_MODEL:-1}"
if [[ -z "${PYTHON_BIN:-}" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
MODEL_KIND="${MODEL_KIND:-auto}"
LOCAL_ONLY="${LOCAL_ONLY:-1}"

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export SPARSITY_GPU_IDS
export DECODER_ONLY_MODEL_KIND="$MODEL_KIND"
export DECODER_ONLY_LOG_DIR="$LOG_DIR"
export DECODER_ONLY_VERBOSE_COMMANDS="${DECODER_ONLY_VERBOSE_COMMANDS:-1}"
if [[ "$LOCAL_ONLY" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

if [[ "$CHECKPOINT_DIAGNOSE" == "1" ]]; then
  mkdir -p "$LOG_DIR"
  diagnose_log="$LOG_DIR/checkpoint_preflight.log"
  echo "Running checkpoint preflight..."
  echo "Log: $diagnose_log"
  set +e
  diagnose_args=("$INPUT_CHECKPOINT" --model-kind "$MODEL_KIND" --local-only "$LOCAL_ONLY")
  if [[ "$CHECKPOINT_LOAD_MODEL" == "1" ]]; then
    diagnose_args+=(--load-model)
  fi
  "$PYTHON_BIN" -m decoder_only.diagnose "${diagnose_args[@]}" >"$diagnose_log" 2>&1
  diagnose_status=$?
  set -e
  cat "$diagnose_log"
  if [[ "$diagnose_status" -ne 0 ]]; then
    echo "Checkpoint preflight failed. Fix the checkpoint folder before running the matrix." >&2
    exit "$diagnose_status"
  fi
fi

echo "Model loader kind: $MODEL_KIND"
echo "Local-only Hugging Face loading: $LOCAL_ONLY"
echo "Logs: $LOG_DIR"

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
