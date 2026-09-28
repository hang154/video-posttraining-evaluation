# Video Post-training Evaluation

Evaluation-first video adaptation work covering LTX-Video 13B,
HunyuanVideo-1.5, real-video manifests, sealed holdouts, and blinded review.

## Synthetic controlled benchmark

### LTX-Video 13B

The 120-step LoRA improved the CLIP prompt/action proxy
(`0.2330 → 0.2566`) but regressed identity and temporal metrics. This is
published as a negative result rather than a cherry-picked demo.

### HunyuanVideo-1.5

A bounded four-step LoRA endpoint improved all five automated proxy metrics in
the preserved 50-prompt evaluation, including optical-flow warp MAE
(`15.0097 → 12.5380`). It remains a proxy result on synthetic character data.

## Real-video extension

The scene-level dataset split contains 16 train, 4 validation, and 4 sealed
holdout records. Media hashes do not cross split boundaries. Public manifests
preserve source URL, license, and hash metadata; source media is not
redistributed here.

Human review is **pending**. Blank blinded reviewer sheets and the review
protocol are included, but no preference rate is claimed.

## Validate

```bash
python scripts/validate_public_evidence.py
python -m py_compile scripts/*.py evaluation/*.py
```

## Limits

Automated embeddings and optical-flow proxies do not replace independent
human review. Dataset size and adaptation budgets are deliberately disclosed.

