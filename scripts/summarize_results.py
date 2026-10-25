import json

with open('scripts/github_arxiv_gaze_models.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total entries in github_arxiv_gaze_models.json: {len(data)}")

with_weights = [d for d in data if d.get('weight_links') and len(d['weight_links']) > 0]
print(f"With weight links: {len(with_weights)}")

for d in with_weights:
    print(f"\n[{d['year']}] {d['repo']} (Stars: {d['stars']})")
    print(f"  Description: {d['description'][:80]}")
    print(f"  Backbones: {d['backbones']}")
    print(f"  Datasets: {d['datasets']}")
    print(f"  Weight links ({len(d['weight_links'])}):")
    for w in d['weight_links']:
        print(f"    - {w}")
