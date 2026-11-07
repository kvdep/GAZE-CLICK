import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('mvp/scratch/autoregressive_arxiv_dump.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total: {len(data)}")
for k, v in data.items():
    print(f"ID: {k}")
    print(f"  Title: {v['title']}")
    print(f"  Date: {v['published'][:10]} | Authors: {', '.join(v['authors'][:3])}")
    print(f"  Summary: {v['summary'][:220]}...")
    print('-'*70)
