# Decoder-Only SLM Further Training

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

Put data under `data/`. Supported formats:

- `.txt`, `.md`, `.text`: entire file is used as training text.
- `.jsonl`: each line can contain `text`, or `prompt` and `completion`.

Example:

```jsonl
{"text": "A complete training example goes here."}
{"prompt": "Question: ...\nAnswer:", "completion": " ..."}
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
