#!/usr/bin/env python3
"""Evaluate fixed-prompt before/after video suites without cherry-picking."""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


def sample_frames(path: Path, count: int = 5) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, max(0, total - 1), count).round().astype(int).tolist())
    frames = []
    for index in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if index in indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(a, dim=-1) @ torch.nn.functional.normalize(b, dim=-1).T


def optical_metrics(frames: list[np.ndarray]) -> tuple[float, float]:
    errors, brightness = [], []
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    for first, second in zip(gray, gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = first.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        remap = np.stack([xx + flow[..., 0], yy + flow[..., 1]], -1).astype(np.float32)
        warped = cv2.remap(first, remap, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        errors.append(float(np.abs(warped.astype(np.float32) - second).mean()))
        brightness.append(abs(float(first.mean()) - float(second.mean())))
    return float(np.mean(errors)), float(np.mean(brightness))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-step", type=int, default=0)
    parser.add_argument("--post-step", type=int, default=120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prompts.suffix in {".yaml", ".yml"}:
        prompts = yaml.safe_load(args.prompts.read_text())["validation"]["prompts"]
    else:
        prompts = json.loads(args.prompts.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()

    dino_name = "facebook/dinov2-small"
    clip_name = "openai/clip-vit-base-patch32"
    dino_processor = AutoImageProcessor.from_pretrained(dino_name)
    dino = AutoModel.from_pretrained(dino_name).eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).eval().to(device)
    action_labels = sorted({action for action in ("cooking", "cycling", "juggling", "walking") if any(action in prompt.lower() for prompt in prompts)})
    with torch.inference_mode():
        action_inputs = clip_processor(text=[f"a video of {action}" for action in action_labels], return_tensors="pt", padding=True).to(device)
        action_text_embeddings = clip_model.get_text_features(**action_inputs)

    reference_frames = []
    for video in sorted(args.references.glob("*.mp4"))[:8]:
        reference_frames.extend(sample_frames(video, 2))
    with torch.inference_mode():
        batch = dino_processor(images=[Image.fromarray(x) for x in reference_frames], return_tensors="pt").to(device)
        reference_embeddings = dino(**batch).last_hidden_state[:, 0]

    rows = []
    for phase, step in (("baseline", args.baseline_step), ("posttrain", args.post_step)):
        for index, prompt in enumerate(prompts):
            path = args.samples / f"step_{step:06d}_{index}.mp4"
            if not path.exists():
                path = args.samples / f"step_{step:06d}_prompt_{index:02d}.mp4"
            if not path.exists():
                raise FileNotFoundError(path)
            frames = sample_frames(path)
            images = [Image.fromarray(x) for x in frames]
            with torch.inference_mode():
                dino_inputs = dino_processor(images=images, return_tensors="pt").to(device)
                frame_embeddings = dino(**dino_inputs).last_hidden_state[:, 0]
                identity = cosine(frame_embeddings, reference_embeddings).max(dim=1).values.mean().item()
                adjacent = cosine(frame_embeddings[:-1], frame_embeddings[1:]).diag().mean().item()
                clip_inputs = clip_processor(text=[prompt], images=images, return_tensors="pt", padding=True).to(device)
                clip_out = clip_model(**clip_inputs)
                adherence = cosine(clip_out.image_embeds, clip_out.text_embeds).mean().item()
                mean_image = clip_out.image_embeds.mean(dim=0, keepdim=True)
                predicted_action = action_labels[int(cosine(mean_image, action_text_embeddings).argmax())]
                expected_action = next((action for action in action_labels if action in prompt.lower()), "")
            warp, brightness = optical_metrics(frames)
            rows.append({
                "phase": phase,
                "step": step,
                "prompt_index": index,
                "prompt": prompt,
                "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frames_sampled": len(frames),
                "identity_dino_cosine": identity,
                "temporal_dino_cosine": adjacent,
                "prompt_action_clip_cosine": adherence,
                "action_expected": expected_action,
                "action_predicted": predicted_action,
                "action_success": int(predicted_action == expected_action),
                "optical_warp_mae": warp,
                "brightness_drift": brightness,
            })

    fields = list(rows[0])
    with (args.output / "per_video_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    numeric = ["identity_dino_cosine", "temporal_dino_cosine", "prompt_action_clip_cosine", "action_success", "optical_warp_mae", "brightness_drift"]
    for phase in ("baseline", "posttrain"):
        subset = [row for row in rows if row["phase"] == phase]
        summary[phase] = {key: float(np.mean([row[key] for row in subset])) for key in numeric}
        summary[phase]["n"] = len(subset)
    summary["delta_post_minus_base"] = {key: summary["posttrain"][key] - summary["baseline"][key] for key in numeric}
    summary["models"] = {"identity": dino_name, "adherence": clip_name}
    summary["elapsed_sec"] = time.time() - started
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    randomizer = random.Random(20260928)
    blind_rows = []
    for index, prompt in enumerate(prompts):
        pair = [row for row in rows if row["prompt_index"] == index]
        randomizer.shuffle(pair)
        blind_rows.append({
            "pair_id": f"pair-{index:03d}",
            "prompt": prompt,
            "candidate_a": pair[0]["file"],
            "candidate_b": pair[1]["file"],
            "preferred": "",
            "identity_notes": "",
            "action_notes": "",
            "temporal_notes": "",
        })
    with (args.output / "blind_review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blind_rows[0]))
        writer.writeheader()
        writer.writerows(blind_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
