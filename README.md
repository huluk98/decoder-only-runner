# Decoder-Only SLM Further Training

## Run First

Use these after activating the `decoder-only-runner` environment. For `Decoder-Chinese-SLM`
checkpoints, keep `MODEL_KIND=hf`; loading is local/offline by default.

Dual SCENIC further training only:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MODEL_KIND=hf \
bash run_scenic_further_training_from_base.sh /PATH/TO/MY/BASE_DECODER_SLM
```

Full SCENIC training plus 20-row pruning matrix:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
SPARSITY_GPU_IDS=0,1,2,3,4,5,6,7 \
MODEL_KIND=hf \
bash run_linear_sparsity_revision_from_base.sh /PATH/TO/MY/DECODER_SLM_CHECKPOINT
```

If a run fails, first check the preflight and stage logs:

```bash
decoder-diagnose /PATH/TO/MY/DECODER_SLM_CHECKPOINT --model-kind hf
tail -n 120 outputs/scenic_further_training/logs/checkpoint_preflight.log
tail -n 120 outputs/scenic_further_training/logs/contrastive_sft.log
tail -n 120 outputs/scenic_further_training/logs/regular_sft.log
tail -n 120 outputs/decoder_pruning_full_matrix/logs/dense_regular_sft.log
```

That checkpoint path must be the final/full model folder, not only the tokenizer folder. For
`Decoder-Chinese-SLM`, it should contain `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json`, and `model.safetensors` or
`pytorch_model.bin` in the same directory. If CPU preflight loading is too slow, set
`CHECKPOINT_LOAD_MODEL=0` after the folder layout has been confirmed.

Small public-safe repository for continuing training and sampling from a trained decoder-only
language model.

It supports two common checkpoint layouts:

- Hugging Face causal language model directories with `config.json`, tokenizer files, and model weights.
- Simple custom PyTorch decoder-only checkpoints with a compatible `config.json` and `model.pt` / `checkpoint.pt`.

Large checkpoint, dataset, and output files are ignored by git so this repo can stay public without
accidentally publishing your trained SLM or private data.

## Setup

For local CPU/MPS testing:

```bash
conda env create -f environment.yml
conda activate decoder-only-runner
pip install -e .
```

If you prefer plain pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For a Linux CUDA server such as an 8x NVIDIA H20 box, install CUDA PyTorch instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cu124.txt
pip install -e .
```

## Add A Checkpoint

Place the trained model under `checkpoints/`, for example:

```text
checkpoints/my-model/
  config.json
  tokenizer.json
  model.safetensors
```

For a custom PyTorch checkpoint, use:

```text
checkpoints/my-model/
  config.json
  model.pt
  vocab.json
```

The custom `config.json` should include at least:

```json
{
  "vocab_size": 256,
  "block_size": 256,
  "n_layer": 6,
  "n_head": 6,
  "n_embd": 384
}
```

If `vocab_size` is `256` and no tokenizer is provided, the runner uses a UTF-8 byte tokenizer.

## Add Further-Training Data

The repo includes the final SCENIC artifacts needed for the decoder SLM runs:

```text
data/scenic/SCENIC_full_training_dataset.json
data/scenic/SCENIC_full_anchor_positive_negative.json
data/benchmarks/iot_instruction_benchmark_200.json
```

Supported additional formats:

- `.txt`, `.md`, `.text`: entire file is used as training text.
- `.jsonl`: each line can contain `text`, or `prompt` and `completion`.

Example:

```jsonl
{"text": "A complete training example goes here."}
{"prompt": "Question: ...\nAnswer:", "completion": " ..."}
```

## One-Command SCENIC Further Training

Run both 5-epoch SCENIC training jobs from a base decoder checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MODEL_KIND=hf \
bash run_scenic_further_training_from_base.sh /PATH/TO/MY/BASE_DECODER_SLM
```

This writes:

```text
outputs/scenic_further_training/contrastive_sft_5epoch/checkpoint-final/
outputs/scenic_further_training/regular_sft_5epoch/checkpoint-final/
```

Defaults:

- contrastive triplet SFT: 5 epochs on `data/scenic/SCENIC_full_anchor_positive_negative.json`
- regular SFT: 5 epochs on `data/scenic/SCENIC_full_training_dataset.json`
- regular SFT starts from the base model. To chain regular SFT after contrastive SFT, set `REGULAR_START=contrastive`.
- `MODEL_KIND=hf` is the right setting for `Decoder-Chinese-SLM` checkpoints. They are local
  Hugging Face-style Llama causal LM checkpoints, even after repairing the tokenizer with
  `PreTrainedTokenizerFast`.
- `LOCAL_ONLY=1` is the default, so model/tokenizer loading uses local files only.
- `MODEL_KIND=custom` is only for the older `model.pt` custom `DecoderOnlyTransformer` format.

Planner check without training:

```bash
DRY_RUN=1 bash run_scenic_further_training_from_base.sh /PATH/TO/MY/BASE_DECODER_SLM
```

## Continue Training

Single-process test run:

```bash
decoder-train \
  --model-path checkpoints/my-model \
  --train-data data/train.jsonl \
  --output-dir outputs/my-model-further-trained \
  --block-size 1024 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 3e-5 \
  --mixed-precision no
```

8x H20 GPU run:

```bash
accelerate launch --config_file configs/accelerate_8xh20.yaml \
  -m decoder_only.train \
  --model-path checkpoints/my-decoder-slm \
  --train-data data/train.jsonl \
  --validation-data data/valid.jsonl \
  --output-dir outputs/my-decoder-slm-further-trained \
  --block-size 2048 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-5 \
  --mixed-precision bf16 \
  --gradient-checkpointing \
  --save-every 500
```

The final checkpoint is written to:

```text
outputs/my-decoder-slm-further-trained/checkpoint-final/
```

## Generate Sparsity Results JSON

To scan a local decoder SLM checkpoint and create a JSON file shaped like
`all_sparsity_results-2.json`:

```bash
./generate_sparsity_results.sh checkpoints/my-decoder-slm all_sparsity_results.json
```

Optional labels can be supplied as environment variables:

```bash
MODEL_TRAINING=regular_sft RUN_LABEL=magnitude_0p5 TARGET_SPARSITY=0.5 \
  ./generate_sparsity_results.sh checkpoints/my-decoder-slm all_sparsity_results.json
```

The script measures checkpoint sparsity only. Training and benchmark EM fields are emitted as
`null` unless they are produced by a separate evaluation workflow. The sparsity calculation follows
the encoder-only implementation: it measures prunable `nn.Linear` weights, excludes output heads by
default, and records encoder-style `pruning_config` fields.

## Run Full 20-Row Pruning Matrix

To start from one decoder checkpoint, create dense SFT baselines, run the one-shot pruning matrix,
run progressive magnitude pruning with recovery, and write the final JSON:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
SPARSITY_GPU_IDS=0,1,2,3,4,5,6,7 \
MODEL_KIND=hf \
bash run_linear_sparsity_revision_from_base.sh /PATH/TO/MY/DECODER_SLM_CHECKPOINT
```

The script uses the bundled SCENIC regular, SCENIC contrastive, and IoT benchmark JSON files by
default.

Expected final JSON rows:

- dense baselines: 2
- regular original one-shot: 7
- contrastive original one-shot: 7
- regular progressive magnitude: 2
- contrastive progressive magnitude: 2
- total JSON rows: 20

For a planner/schema check without training or pruning:

```bash
DRY_RUN=1 bash run_linear_sparsity_revision_from_base.sh /PATH/TO/MY/DECODER_SLM_CHECKPOINT
```

## Run Generation

```bash
python -m decoder_only.generate \
  --model-path outputs/my-decoder-slm-further-trained/checkpoint-final \
  --prompt "Once upon a time" \
  --max-new-tokens 80
```

Useful options:

```bash
python -m decoder_only.generate --help
```

## Repository Notes

- `checkpoints/` is intentionally ignored except for `.gitkeep`.
- `data/` and `outputs/` are intentionally ignored except for lightweight placeholders.
- Use GitHub Releases, Hugging Face Hub, or another artifact store for large public weights.
- Keep tokenizer files with the checkpoint whenever the model was trained with a tokenizer other than UTF-8 bytes.
