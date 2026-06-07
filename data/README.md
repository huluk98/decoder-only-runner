# Training Data

Put local further-training data here. This directory is ignored by git except for this note.

Supported formats:

- `.txt`, `.md`, `.text`: the whole file is treated as training text.
- `.jsonl`: each line can contain a `text` field, or `prompt` and `completion` fields.

Example JSONL:

```jsonl
{"text": "A complete training example goes here."}
{"prompt": "Question: ...\nAnswer:", "completion": " ..."}
```
