import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time

queries = [
    'all:autoregressive AND all:"gaze"',
    'all:autoregressive AND all:"eye tracking"',
    'all:autoregressive AND all:"scanpath"',
    'all:"autoregressive gaze"',
    'all:"gaze trajectory" AND all:transformer',
    'all:"scanpath prediction" AND all:transformer',
    'all:"recurrent gaze" AND all:differential',
    'all:"causal video" AND all:gaze'
]

results = {}
ns = {'atom': 'http://www.w3.org/2005/Atom'}

for q in queries:
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&start=0&max_results=15"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        entries = root.findall('atom:entry', ns)
        print(f"Query '{q}': {len(entries)} entries found")
        for e in entries:
            id_val = e.find('atom:id', ns).text.strip()
            title = e.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = e.find('atom:summary', ns).text.strip().replace('\n', ' ')
            published = e.find('atom:published', ns).text.strip()
            authors = [a.find('atom:name', ns).text.strip() for a in e.findall('atom:author', ns)]
            if id_val not in results:
                results[id_val] = {
                    'title': title,
                    'published': published,
                    'authors': authors,
                    'summary': summary,
                    'url': id_val
                }
        time.sleep(1.0)
    except Exception as ex:
        print(f"Error querying {q}: {ex}")

with open('mvp/scratch/autoregressive_arxiv_dump.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Total unique papers found: {len(results)}")
