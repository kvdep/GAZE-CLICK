import urllib.request
import xml.etree.ElementTree as ET
import sys

sys.stdout.reconfigure(encoding='utf-8')

queries = [
    'all:autoregressive AND all:"gaze estimation"',
    'all:autoregressive AND all:"eye tracking"',
    'all:"differential gaze" AND all:video',
    'all:"trajectory token" AND all:gaze'
]

for q in queries:
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&start=0&max_results=6"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        print(f"=== Query: {q} (Found: {len(entries)}) ===")
        for e in entries:
            title = e.find('atom:title', ns).text.strip().replace('\n', ' ')
            id_val = e.find('atom:id', ns).text.strip()
            summary = e.find('atom:summary', ns).text.strip().replace('\n', ' ')[:250]
            print(f"- Title: {title}\n  ID: {id_val}\n  Summary: {summary}...\n")
    except Exception as ex:
        print(f"Error querying {q}: {ex}")
