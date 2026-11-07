import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import sys
import time
import json

sys.stdout.reconfigure(encoding='utf-8')

queries = [
    'all:"video gaze estimation"',
    'all:"temporal gaze estimation"',
    'all:"EVE dataset" AND all:"gaze"',
    'all:"continuous gaze estimation"',
    'all:"spatio-temporal" AND all:"gaze estimation"'
]

results = []
seen_ids = set()

for q in queries:
    url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            arxiv_id = entry.find('atom:id', ns).text.strip()
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', ns).text.strip()[:10]
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            results.append({'title': title, 'id': arxiv_id, 'date': published, 'summary': summary})
        time.sleep(3.2)
    except Exception as e:
        print(f'Error fetching {q}: {e}')

with open('mvp/scratch/arxiv_video_gaze_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f'Fetched {len(results)} papers.')
for i, r in enumerate(results[:15]):
    print(f'{i+1}. [{r["date"]}] {r["title"]}')
    print(f'   ID: {r["id"]}')
