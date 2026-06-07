# Decoder-Only Model Runner

Small runnable environment for loading and sampling from a trained decoder-only language model.

It supports two common checkpoint layouts:

- Hugging Face causal language model directories with `config.json`, tokenizer files, and model weights.
- Simple custom PyTorch decoder-only checkpoints with a compatible `config.json` and `model.pt` / `checkpoint.pt`.

Large checkpoint files are ignored by git so this repo can stay public without accidentally publishing trained weights.

## Setup

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

## Run Generation

```bash
python -m decoder_only.generate \
  --model-path checkpoints/my-model \
  --prompt "Once upon a time" \
  --max-new-tokens 80
```

Useful options:

```bash
python -m decoder_only.generate --help
```

## Repository Notes

- `checkpoints/` is intentionally ignored except for `.gitkeep`.
- Use GitHub Releases, Hugging Face Hub, or another artifact store for large public weights.
- Keep tokenizer files with the checkpoint whenever the model was trained with a tokenizer other than UTF-8 bytes.
