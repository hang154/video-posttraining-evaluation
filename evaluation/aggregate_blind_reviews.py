#!/usr/bin/env python3
"""Aggregate three completed blinded CSVs without exposing the key beforehand."""
import csv
import json
from collections import Counter
from pathlib import Path

PACK = Path(__file__).resolve().parents[1] / "results"
key={row['pair_id']:row for row in json.loads((PACK/'private_randomization_key.json').read_text())}
reviews=[]
for index in range(1,4):
    path=PACK/f'reviewer_{index:02d}.csv'
    rows=list(csv.DictReader(path.open()))
    if len(rows)!=len(key) or any(row['preferred_A_B_tie'] not in ('A','B','tie') for row in rows):
        raise SystemExit(f'PENDING: complete every preference in {path.name}')
    reviews.extend(rows)
counts=Counter()
for row in reviews:
    choice=row['preferred_A_B_tie']
    counts['tie' if choice=='tie' else key[row['pair_id']][choice]] += 1
result={'reviewers':3,'ratings':len(reviews),'preference_counts':dict(counts),
        'tuned_preference_rate_excluding_ties':counts['posttrain']/max(1,counts['posttrain']+counts['baseline'])}
(PACK/'human_review_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
