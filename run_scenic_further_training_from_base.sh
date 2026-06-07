#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NPROC_PER_NODE=8 \
  bash run_scenic_further_training_from_base.sh /PATH/TO/BASE_DECODER_SLM

Runs two SCENIC further-training jobs from the base decoder model:
  1. contrastive_sft: 5 epochs on data/scenic/SCENIC_full_anchor_positive_negative.json
  2. regular_sft:    5 epochs on data/scenic/SCENIC_full_training_dataset.json

Useful overrides:
  OUTPUT_ROOT=outputs/scenic_further_training
  CONTRASTIVE_EPOCHS=5
  REGULAR_EPOCHS=5
  REGULAR_START=base        # base or contrastive
  BATCH_SIZE=1
  GRADIENT_ACCUMULATION_STEPS=16
  LEARNING_RATE=2e-5
  MIXED_PRECISION=bf16
  MODEL_KIND=hf          # auto, hf, or custom
  LOCAL_ONLY=1
  CHECKPOINT_DIAGNOSE=1
  LOG_DIR=$OUTPUT_ROOT/logs
  DRY_RUN=1
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

BASE_MODEL="$1"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/scenic_further_training}"
CONTRASTIVE_DATASET="${CONTRASTIVE_DATASET:-data/scenic/SCENIC_full_anchor_positive_negative.json}"
REGULAR_DATASET="${REGULAR_DATASET:-data/scenic/SCENIC_full_training_dataset.json}"
CONTRASTIVE_OUTPUT="${CONTRASTIVE_OUTPUT:-$OUTPUT_ROOT/contrastive_sft_5epoch}"
REGULAR_OUTPUT="${REGULAR_OUTPUT:-$OUTPUT_ROOT/regular_sft_5epoch}"
CONTRASTIVE_EPOCHS="${CONTRASTIVE_EPOCHS:-5}"
REGULAR_EPOCHS="${REGULAR_EPOCHS:-5}"
REGULAR_START="${REGULAR_START:-base}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-256}"
CONTRASTIVE_LOSS_WEIGHT="${CONTRASTIVE_LOSS_WEIGHT:-0.1}"
CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-0.5}"
NEGATIVE_FIELD="${NEGATIVE_FIELD:-negative}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
CHECKPOINT_DIAGNOSE="${CHECKPOINT_DIAGNOSE:-1}"
if [[ -z "${PYTHON_BIN:-}" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
MODEL_KIND="${MODEL_KIND:-auto}"
LOCAL_ONLY="${LOCAL_ONLY:-1}"

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export DECODER_ONLY_MODEL_KIND="$MODEL_KIND"
if [[ "$LOCAL_ONLY" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required path: $1" >&2
    exit 1
  fi
}

run_python_module() {
  local stage_name="$1"
  shift
  mkdir -p "$LOG_DIR"
  local log_file="$LOG_DIR/${stage_name}.log"
  local command=()
  if [[ "$NPROC_PER_NODE" == "1" ]]; then
    command=("$PYTHON_BIN" -m decoder_only.train "$@")
  else
    command=("$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" -m decoder_only.train "$@")
  fi

  echo "Command:"
  print_command "${command[@]}"
  echo "Log: $log_file"
  set +e
  "${command[@]}" >"$log_file" 2>&1
  local status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "Stage failed: $stage_name" >&2
    echo "Exit code: $status" >&2
    echo "Last 120 log lines from $log_file:" >&2
    tail -n 120 "$log_file" >&2 || true
    exit "$status"
  fi
  echo "Stage completed: $stage_name"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

print_training_command() {
  if [[ "$NPROC_PER_NODE" == "1" ]]; then
    print_command "$PYTHON_BIN" -m decoder_only.train "$@"
  else
    print_command "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" -m decoder_only.train "$@"
  fi
}

require_path "$BASE_MODEL"
require_path "$CONTRASTIVE_DATASET"
require_path "$REGULAR_DATASET"

if [[ "$CHECKPOINT_DIAGNOSE" == "1" ]]; then
  mkdir -p "$LOG_DIR"
  diagnose_log="$LOG_DIR/checkpoint_preflight.log"
  echo "Running checkpoint preflight..."
  echo "Log: $diagnose_log"
  set +e
  "$PYTHON_BIN" -m decoder_only.diagnose "$BASE_MODEL" --model-kind "$MODEL_KIND" --local-only "$LOCAL_ONLY" >"$diagnose_log" 2>&1
  diagnose_status=$?
  set -e
  cat "$diagnose_log"
  if [[ "$diagnose_status" -ne 0 ]]; then
    echo "Checkpoint preflight failed. Fix the checkpoint folder before training." >&2
    exit "$diagnose_status"
  fi
fi

if [[ "$REGULAR_START" != "base" && "$REGULAR_START" != "contrastive" ]]; then
  echo "REGULAR_START must be either 'base' or 'contrastive'." >&2
  exit 2
fi

REGULAR_MODEL="$BASE_MODEL"
if [[ "$REGULAR_START" == "contrastive" ]]; then
  REGULAR_MODEL="$CONTRASTIVE_OUTPUT/checkpoint-final"
fi

contrastive_args=(
  --training-mode contrastive
  --model-path "$BASE_MODEL"
  --train-data "$CONTRASTIVE_DATASET"
  --output-dir "$CONTRASTIVE_OUTPUT"
  --epochs "$CONTRASTIVE_EPOCHS"
  --block-size "$BLOCK_SIZE"
  --max-source-length "$MAX_SOURCE_LENGTH"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --mixed-precision "$MIXED_PRECISION"
  --contrastive-loss-weight "$CONTRASTIVE_LOSS_WEIGHT"
  --contrastive-margin "$CONTRASTIVE_MARGIN"
  --negative-field "$NEGATIVE_FIELD"
  --gradient-checkpointing
)

regular_args=(
  --training-mode sft
  --model-path "$REGULAR_MODEL"
  --train-data "$REGULAR_DATASET"
  --output-dir "$REGULAR_OUTPUT"
  --epochs "$REGULAR_EPOCHS"
  --block-size "$BLOCK_SIZE"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --mixed-precision "$MIXED_PRECISION"
  --gradient-checkpointing
)

echo "SCENIC contrastive dataset: $CONTRASTIVE_DATASET"
echo "SCENIC regular SFT dataset: $REGULAR_DATASET"
echo "Base model: $BASE_MODEL"
echo "Contrastive output: $CONTRASTIVE_OUTPUT/checkpoint-final"
echo "Regular SFT output: $REGULAR_OUTPUT/checkpoint-final"
echo "Regular SFT starts from: $REGULAR_START"
echo "Model loader kind: $MODEL_KIND"
echo "Local-only Hugging Face loading: $LOCAL_ONLY"
echo "Logs: $LOG_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run; commands that would run:"
  print_training_command "${contrastive_args[@]}"
  print_training_command "${regular_args[@]}"
  exit 0
fi

echo "Starting contrastive SCENIC training for $CONTRASTIVE_EPOCHS epochs..."
run_python_module "contrastive_sft" "${contrastive_args[@]}"

if [[ "$REGULAR_START" == "contrastive" ]]; then
  require_path "$REGULAR_MODEL"
fi

echo "Starting regular SCENIC SFT for $REGULAR_EPOCHS epochs..."
run_python_module "regular_sft" "${regular_args[@]}"

echo "Done."
echo "Contrastive checkpoint: $CONTRASTIVE_OUTPUT/checkpoint-final"
echo "Regular SFT checkpoint: $REGULAR_OUTPUT/checkpoint-final"
