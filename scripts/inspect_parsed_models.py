import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('scripts/huggingface_gaze_models.json', 'r', encoding='utf-8') as f:
    models = json.load(f)

print(f"Total parsed models: {len(models)}")
for m in models:
    mid = m['id']
    yr = m['year']
    dl = m['downloads']
    lic = m['license']
    n_files = len(m['files'])
    fmts = [k for k, v in m['formats'].items() if v]
    print("=" * 70)
    print(f"Model ID: {mid} | Year: {yr} | Downloads: {dl} | License: {lic}")
    print(f"Formats: {fmts} | Files ({n_files}): {m['files'][:8]}")
    if m['readme_snippet']:
        snippet = " // ".join([l.strip() for l in m['readme_snippet'].splitlines() if l.strip()][:3])
        print(f"README: {snippet[:200]}")
