# Video post-training on H200

## Dataset and protocol

- Synthetic character identity: `TCROBOT`.
- Scene-disjoint split: 24 train, 4 validation, 4 sealed holdout clips.
- LTX evaluation: 50 fixed prompts and seed 42, untouched step-0 versus
  120-step LoRA.
- Hunyuan evaluation: the same 50 prompt concepts, untouched baseline versus
  a bounded 4-step LoRA endpoint.
- Human blind-review sheets are generated with randomized A/B ordering and
  intentionally blank preference fields.

## LTX-Video 13B result

The single-GPU 120-step LoRA run saved 40/80/120-step checkpoints and 50+50
before/after videos. A separate official two-process distributed run completed
with global batch 2, peak memory 118.54GB, 0.36 step/s, and exit code 0.

After the smoke gate, the official two-process launcher also completed the full
120-step run on GPU2+GPU3: global batch 2, 1.36 steps/s, 2.71 samples/s, 118.54GB
peak memory, and 1.5 minutes reported training time. It emitted checkpoints at
steps 40/80/120; retention kept step 80, step 120, and the step-120 ComfyUI
export.

| Metric | untouched base | LoRA | delta |
|---|---:|---:|---:|
| DINO identity cosine | 0.2976 | 0.2671 | -0.0305 |
| DINO temporal cosine | 0.9651 | 0.9077 | -0.0574 |
| CLIP prompt/action proxy | 0.2330 | 0.2566 | +0.0236 |
| optical-flow warp MAE | 5.7382 | 5.8016 | +0.0634 |
| brightness drift | 0.4272 | 0.4809 | +0.0537 |

This is a regression case, not a success-only showcase: prompt alignment
improved while identity and temporal stability declined.

## HunyuanVideo-1.5

The official trainer required a real manifest dataloader and a true untouched
baseline path; both are preserved as vendor patches. A one-prompt smoke test
completed baseline generation, one real LoRA update (loss `0.1865`, gradient
norm `0.6562`), checkpoint save, and same-prompt final generation.

The 50-prompt before/after result was generated in two 25-prompt shards with a
single shared 4-step LoRA checkpoint. The fixed 100-video evaluation completed
as follows.

| Metric | untouched base | LoRA | delta |
|---|---:|---:|---:|
| DINO identity cosine | 0.1532 | 0.1831 | +0.0299 |
| DINO temporal cosine | 0.9103 | 0.9198 | +0.0095 |
| CLIP prompt/action proxy | 0.2712 | 0.2845 | +0.0133 |
| optical-flow warp MAE | 15.0097 | 12.5380 | -2.4717 |
| brightness drift | 0.8004 | 0.7782 | -0.0222 |

All five automated metrics improved for this bounded endpoint. This remains a
proxy evaluation on a synthetic character dataset, not a human preference
result.

## Disclosed compatibility repairs

- Corrected absolute dataset paths that silently misplaced LTX condition
  tensors.
- Unwrapped Accelerate DDP before enabling LTX gradient checkpointing.
- Initialized the non-main-rank LTX checkpoint return path.
- Disabled incompatible optional FlashAttention, Apex fused RMSNorm, and
  Triton compilation; used eager PyTorch SDPA.
- Used public `google/siglip-so400m-patch14-384` because the officially
  documented FLUX.1-Redux dependency was gated for this account.
- Disabled Hunyuan gradient checkpointing after an upstream tracing/assert
  conflict; the LoRA step fits on one 141GB H200.
