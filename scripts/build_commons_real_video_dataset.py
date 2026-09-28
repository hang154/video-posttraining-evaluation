#!/usr/bin/env python3
"""Build a deterministic, attributed real-video dataset from Wikimedia Commons."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dataset_download'
CATEGORIES = [
    ('walking', 'Videos of people walking'),
    ('cooking', 'Videos of cooking'),
    ('cycling', 'Videos of cycling'),
    ('juggling', 'Videos of juggling'),
    ('skateboarding', 'Videos of skateboarding'),
]
ALLOWED = {'CC0', 'Public domain', 'CC BY 2.0', 'CC BY 2.5', 'CC BY 3.0', 'CC BY 4.0',
           'CC BY-SA 2.0', 'CC BY-SA 2.5', 'CC BY-SA 3.0', 'CC BY-SA 4.0'}
USER_AGENT = 'H200PortfolioResearch/1.0 (dataset provenance audit)'


def api(params: dict) -> dict:
    url = 'https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(params)
    delay = 10
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError('unreachable')


def candidates(action: str, category: str) -> list[dict]:
    data = api({
        'action': 'query', 'generator': 'categorymembers', 'gcmtitle': f'Category:{category}',
        'gcmtype': 'file', 'gcmlimit': '50', 'prop': 'videoinfo',
        'viprop': 'url|mime|size|derivatives|extmetadata', 'format': 'json', 'formatversion': '2',
    })
    rows = []
    for page in data.get('query', {}).get('pages', []):
        info = page.get('videoinfo', [{}])[0]
        meta = info.get('extmetadata', {})
        license_name = meta.get('LicenseShortName', {}).get('value', '')
        size = int(info.get('size', 0))
        mime = info.get('mime', '')
        if license_name not in ALLOWED or not (mime.startswith('video/') or mime == 'application/ogg'):
            continue
        if float(info.get('duration', 0)) < 2.6:
            continue
        derivatives = [item for item in info.get('derivatives', []) if item.get('height', 0) in (240, 360) and item.get('src')]
        if not derivatives:
            continue
        derivative = sorted(derivatives, key=lambda item: (abs(item.get('height', 0) - 360), item.get('bandwidth', 0)))[0]
        rows.append({
            'action': action, 'category': category, 'title': page['title'],
            'source_page': info.get('descriptionurl'), 'original_url': info.get('url'),
            'download_url': derivative['src'], 'derivative_height': derivative.get('height'),
            'mime': mime, 'source_bytes': size, 'license': license_name,
            'license_url': meta.get('LicenseUrl', {}).get('value', ''),
            'artist_html': meta.get('Artist', {}).get('value', ''),
            'credit_html': meta.get('Credit', {}).get('value', ''),
            'usage_terms': meta.get('UsageTerms', {}).get('value', ''),
        })
    return sorted(rows, key=lambda row: (row['source_bytes'], row['title']))[:8]


def download(row: dict, index: int) -> dict | None:
    raw = OUT / 'raw' / f'{index:03d}{Path(urllib.parse.urlparse(row["download_url"]).path).suffix}'
    clip = OUT / 'clips' / f'{index:03d}.mp4'
    raw.parent.mkdir(parents=True, exist_ok=True)
    clip.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists() or raw.stat().st_size < 100_000:
        delay = 10
        for attempt in range(8):
            try:
                request = urllib.request.Request(row['download_url'], headers={'User-Agent': USER_AGENT})
                with urllib.request.urlopen(request, timeout=180) as response, raw.open('wb') as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                if attempt == 7:
                    return None
                time.sleep(delay)
                delay = min(delay * 2, 300)
    command = [
        'ffmpeg', '-y', '-v', 'error', '-ss', '0.5', '-i', str(raw), '-an',
        '-vf', 'fps=12,scale=256:256:force_original_aspect_ratio=increase,crop=256:256',
        '-frames:v', '25', '-threads', '4', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(clip),
    ]
    if subprocess.run(command, check=False).returncode:
        return None
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
         '-show_entries', 'stream=nb_read_frames', '-of', 'default=nw=1:nk=1', str(clip)],
        capture_output=True, text=True, check=False,
    )
    if probe.stdout.strip() != '25':
        clip.unlink(missing_ok=True)
        return None
    result = dict(row)
    result.update({'media_path': str(clip), 'sha256': hashlib.sha256(clip.read_bytes()).hexdigest()})
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = []
    for action, category in CATEGORIES:
        try:
            selected.extend(candidates(action, category))
        except Exception as error:
            (OUT / 'download_errors.log').open('a').write(f'{category}: {error}\n')
        time.sleep(2)
    if len(selected) < 24:
        raise RuntimeError(f'only {len(selected)} license-filtered candidates; need at least 24')
    selected = selected[:32]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        processed = list(pool.map(lambda pair: download(*pair), [(row, i) for i, row in enumerate(selected)]))
    valid = [row for row in processed if row is not None]
    groups = {}
    for row in valid:
        groups.setdefault(row['action'], []).append(row)
    eligible = [(action, items) for action, items in sorted(groups.items()) if len(items) >= 6][:4]
    if len(eligible) < 4:
        raise RuntimeError(f'need four actions with six clips each; got { {k: len(v) for k,v in groups.items()} }')
    rows = []
    for split, start, end in (('train', 0, 4), ('validation', 4, 5), ('sealed_holdout', 5, 6)):
        for _, items in eligible:
            for row in items[start:end]:
                row['split'] = split
                rows.append(row)
    # Deterministic action-stratified, scene-level split; holdout never enters train JSON.
    for index, row in enumerate(rows):
        split = row['split']
        row['scene_id'] = f'commons-{index:03d}-{row["action"]}'
        row['caption'] = f'a real-world video of {row["action"]}, stable subject identity and natural temporal motion'
    assert len({row['sha256'] for row in rows}) == len(rows)
    for split in ('train', 'validation', 'sealed_holdout'):
        payload = [row for row in rows if row['split'] == split]
        (OUT / f'{split}.json').write_text(json.dumps(payload, indent=2) + '\n')
    (OUT / 'immutable_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    (OUT / 'fingerprint.sha256').write_text(fingerprint + '\n')
    print(json.dumps({'train': 16, 'validation': 4, 'sealed_holdout': 4, 'fingerprint': fingerprint}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
