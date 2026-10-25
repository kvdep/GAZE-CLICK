#!/usr/bin/env python3
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

SEARCH_TERMS = [
    'gaze-estimation',
    'gaze estimation',
    'gaze360',
    'mpiigaze',
    'mpiifacegaze',
    'xgaze',
    'eth-xgaze',
    'l2cs',
    'gazetr',
    'puregaze',
    'unigaze',
    'omnigaze',
    'eyetag',
    'emc-gaze',
    'eye-tracking',
    'eye tracking',
    'eye-gaze',
    'gaze-tracking',
    'gaze tracking',
    'gazecapture',
    'crossgaze',
    'mcgaze',
    'ptgaze',
    'appearance gaze',
    'point of gaze',
]

EXCLUDE_PATTERNS = [
    'irish', 'gazeta', 'the-gazette', 'stargazer', 'flower', 'iris_species',
    'wav2vec2', 'speech', 'whisper', 'audio', 'voice', 'sound', 'music',
    't5', 'bart', 'bert', 'roberta', 'electra', 'gpt', 'llama', 'mistral',
    'translation', 'summarization', 'ner', 'sentiment', 'text-generation',
    'diffusers', 'stable-diffusion', 'lora', 'konosuba', 'arknights', 'anime',
    'pokemon', 'dreambooth'
]

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return None

def fetch_text(url):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return None

def main():
    print('=== Hugging Face Gaze Estimation Model Parser ===')
    collected_models = {}

    for term in SEARCH_TERMS:
        q = urllib.parse.quote(term)
        url = f'https://huggingface.co/api/models?search={q}&limit=100&full=true'
        data = fetch_json(url)
        if not data:
            continue
        for item in data:
            mid = item.get('id')
            if not mid:
                continue
            mid_lower = mid.lower()
            if any(p in mid_lower for p in EXCLUDE_PATTERNS):
                continue
            tags = [t.lower() for t in item.get('tags', [])]
            if any(any(p in t for p in EXCLUDE_PATTERNS) for t in tags):
                continue
            if mid not in collected_models:
                collected_models[mid] = item

    print(f'Retrieved {len(collected_models)} candidates from initial search.')

    # Deep filter and metadata extraction
    parsed_records = []
    for mid, item in collected_models.items():
        mid_lower = mid.lower()
        # Verify gaze / eye relevance
        is_gaze = any(k in mid_lower for k in ['gaze', 'l2cs', 'ptgaze', 'eye-track', 'eyetrack', 'puregaze', 'gazetr', 'unigaze', 'omnigaze'])
        tags = [t.lower() for t in item.get('tags', [])]
        if not is_gaze and not any('gaze' in t or 'eye-tracking' in t for t in tags):
            continue

        detail_url = f'https://huggingface.co/api/models/{mid}'
        detail = fetch_json(detail_url)
        if not detail:
            detail = item

        readme_url = f'https://huggingface.co/{mid}/raw/main/README.md'
        readme = fetch_text(readme_url) or ''

        # Extract file formats and size
        siblings = detail.get('siblings', [])
        files = [s.get('rfilename', '') for s in siblings]
        has_onnx = any(f.endswith('.onnx') for f in files)
        has_pt = any(f.endswith(('.pt', '.pth', '.pkl', '.bin')) for f in files)
        has_safetensors = any(f.endswith('.safetensors') for f in files)
        has_tflite = any(f.endswith('.tflite') for f in files)

        created_at = detail.get('createdAt') or detail.get('lastModified') or '2020-01-01T00:00:00.000Z'
        year = int(created_at[:4])

        # License
        license_str = detail.get('cardData', {}).get('license') if isinstance(detail.get('cardData'), dict) else None
        if not license_str:
            for t in detail.get('tags', []):
                if t.startswith('license:'):
                    license_str = t.split(':', 1)[1]
                    break
        if not license_str:
            license_str = 'Not specified'

        record = {
            'id': mid,
            'year': year,
            'createdAt': created_at,
            'downloads': detail.get('downloads', 0),
            'likes': detail.get('likes', 0),
            'license': license_str,
            'files': files,
            'formats': {
                'onnx': has_onnx,
                'pt': has_pt,
                'safetensors': has_safetensors,
                'tflite': has_tflite
            },
            'pipeline_tag': detail.get('pipeline_tag'),
            'readme_snippet': readme[:1500] if readme else ''
        }
        parsed_records.append(record)

    # Sort: year descending, downloads descending
    parsed_records.sort(key=lambda r: (r['year'], r['downloads']), reverse=True)

    out_file = 'scripts/huggingface_gaze_models.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_records, f, indent=2, ensure_ascii=False)

    print(f'Successfully parsed and saved {len(parsed_records)} verified gaze models to {out_file}.')
    print('\n=== Models summary (chronological order) ===')
    for r in parsed_records:
        fmts = [k for k, v in r['formats'].items() if v]
        fmt_str = ', '.join(fmts) if fmts else 'code/data'
        print(f"[{r['year']}] {r['id']:40s} | Downloads: {r['downloads']:5d} | License: {r['license']:15s} | Formats: {fmt_str}")

if __name__ == '__main__':
    main()
