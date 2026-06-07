# Training Data

Put local further-training data here. This directory is ignored by git except for this note.

Supported formats:

- `.txt`, `.md`, `.text`: the whole file is treated as training text.
- `.jsonl`: each line can contain a `text` field, or `prompt` and `completion` fields.
- `.json`: list/object payloads with `text`, `prompt`/`response`, `prompt`/`completion`, or SCENIC-style `anchor`/`positive`/`response` fields.

Example JSONL:

```jsonl
{"text": "A complete training example goes here."}
{"prompt": "Question: ...\nAnswer:", "completion": " ..."}
```
