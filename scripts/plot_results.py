#!/usr/bin/env python3
"""Plot the real-video sealed-holdout proxies from the preserved summary."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "results/real_video_summary.json").read_text())
metrics = [
    ("Reference visual similarity", "identity_dino_cosine", True),
    ("Adjacent-frame consistency", "temporal_dino_cosine", True),
    ("Prompt/action CLIP", "prompt_action_clip_cosine", True),
    ("Optical warp MAE", "optical_warp_mae", False),
    ("Brightness drift", "brightness_drift", False),
]
rows = []
for i, (label, key, higher) in enumerate(metrics):
    base, tuned = data["baseline"][key], data["posttrain"][key]
    maximum = max(base, tuned) or 1
    y = 90 + i * 70
    bw, tw = 240 * base / maximum, 240 * tuned / maximum
    direction = "higher is better" if higher else "lower is better"
    rows.append(f'<text x="20" y="{y+16}" font-size="14">{label}</text><text x="20" y="{y+35}" font-size="11" fill="#64748b">{direction}</text>')
    rows.append(f'<rect x="240" y="{y}" width="{bw:.1f}" height="20" fill="#64748b"/><text x="{250+bw:.1f}" y="{y+15}" font-size="12">{base:.4f}</text>')
    rows.append(f'<rect x="240" y="{y+25}" width="{tw:.1f}" height="20" fill="#7c3aed"/><text x="{250+tw:.1f}" y="{y+40}" font-size="12">{tuned:.4f}</text>')
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470"><rect width="760" height="470" fill="#f8fafc"/>
<text x="20" y="38" font-size="24" font-weight="700">Real-video sealed holdout (12 generations)</text>
<rect x="500" y="22" width="16" height="16" fill="#64748b"/><text x="522" y="35" font-size="13">Base</text><rect x="580" y="22" width="16" height="16" fill="#7c3aed"/><text x="602" y="35" font-size="13">Tuned</text>{''.join(rows)}</svg>'''
(ROOT / "plots/real_video_holdout.svg").write_text(svg)
