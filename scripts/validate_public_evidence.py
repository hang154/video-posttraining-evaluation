import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
parts = {}
for name in ("train", "validation", "sealed_holdout"):
    records = json.loads((root / f"dataset/{name}.json").read_text())
    parts[name] = {r.get("sha256") or r.get("media_sha256") for r in records}
assert not (parts["train"] & parts["validation"])
assert not (parts["train"] & parts["sealed_holdout"])
assert not (parts["validation"] & parts["sealed_holdout"])
status = json.loads((root / "results/human_review_status.json").read_text())
assert status.get("status") == "PENDING_HUMAN_REVIEW"
print("video public evidence validation: PASS")

