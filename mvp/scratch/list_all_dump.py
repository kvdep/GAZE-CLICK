import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('mvp/scratch/autoregressive_arxiv_dump.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for idx, (k, v) in enumerate(data.items()):
    print(f"[{idx+1}] {v['title']} ({v['published'][:10]})")
    print(f"    URL: {k}")
    print(f"    Authors: {', '.join(v['authors'][:4])}")
    print(f"    Summary: {v['summary'][:200]}...")
    print()
