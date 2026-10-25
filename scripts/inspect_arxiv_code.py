import xml.etree.ElementTree as ET
import urllib.request
import re

HEADERS = {'User-Agent': 'Mozilla/5.0'}
url = 'http://export.arxiv.org/api/query?search_query=all:%22gaze+estimation%22+AND+cat:cs.CV&start=0&max_results=150&sortBy=submittedDate&sortOrder=descending'
req = urllib.request.Request(url, headers=HEADERS)
with urllib.request.urlopen(req, timeout=30) as resp:
    xml_data = resp.read().decode('utf-8')

root = ET.fromstring(xml_data)
ns = {'atom': 'http://www.w3.org/2005/Atom'}
entries = root.findall('atom:entry', ns)

print(f"Total entries: {len(entries)}")

papers_with_code = []
for entry in entries:
    title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
    summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
    published = entry.find('atom:published', ns).text[:10]
    arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
    
    code_matches = re.findall(r'https?://[^\s\)]+(?:github\.com|huggingface\.co|gitlab\.com)[^\s\)\.\,]+', summary)
    if 'code' in summary.lower() or 'available at' in summary.lower() or code_matches:
        papers_with_code.append({
            'arxiv_id': arxiv_id,
            'title': title,
            'published': published,
            'year': published[:4],
            'urls': code_matches,
            'summary_snippet': summary[:250]
        })

print(f"Found {len(papers_with_code)} papers from 2024-2026 mentioning code/urls:")
for p in papers_with_code:
    print(f"[{p['published']}] ({p['arxiv_id']}) {p['title']}")
    if p['urls']:
        print(f"   URLs: {p['urls']}")
