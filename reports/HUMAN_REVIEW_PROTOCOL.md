# Blinded human review protocol

The review pack contains 12 prompt-matched A/B pairs. The mapping from A/B to
baseline/post-training is stored separately in `private_randomization_key.json`.

Three reviewers independently watch both complete clips, without seeing model
labels or automatic metrics. For each pair they record A, B, or tie and score
identity stability, temporal quality, and action adherence from 1–5. Reviewers
must not coordinate. After all 36 rows are complete,
`aggregate_blind_reviews.py` resolves the hidden labels and writes the result.

Until then, `human_review_status.json` remains `PENDING_HUMAN_REVIEW`; no human
preference percentage may be reported or inferred from automatic metrics.
