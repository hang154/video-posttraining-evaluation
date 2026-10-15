# BASE vs LoRA comparison clips

This directory contains the generated clips used for the preserved model-before
and model-after comparison.

- `ltx/base/`: 50 BASE clips (`step_000000_*`)
- `ltx/lora_tuned/`: 50 LoRA-tuned clips (`step_000120_*`)
- `hunyuan/base/`: 50 BASE clips (`step_000000_*`)
- `hunyuan/lora_tuned/`: 50 LoRA-tuned clips (`step_000004_*`)

Matching numeric suffixes refer to the same prompt index within each model.
The clips are short generated samples, not licensed source media. Check
`SHA256SUMS` after download for byte-level integrity.
