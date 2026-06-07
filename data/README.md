# Training Data

This repo tracks only the final SCENIC artifacts needed for the one-command decoder SLM runs:

- `data/scenic/SCENIC_full_training_dataset.json`
- `data/scenic/SCENIC_full_anchor_positive_negative.json`
- `data/benchmarks/iot_instruction_benchmark_200.json`

Other local datasets remain ignored by git.

Supported formats:

- `.txt`, `.md`, `.text`: the whole file is treated as training text.
- `.jsonl`: each line can contain a `text` field, or `prompt` and `completion` fields.
- `.json`: list/object payloads with `text`, `prompt`/`response`, `prompt`/`completion`, or SCENIC-style `anchor`/`positive`/`negative`/`response` fields.

Example JSONL:

```jsonl
{"text": "A complete training example goes here."}
{"prompt": "Question: ...\nAnswer:", "completion": " ..."}
```
