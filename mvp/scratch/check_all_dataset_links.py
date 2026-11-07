import urllib.request
import urllib.error
import ssl
import re
import json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
}

def test_url(name, url):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=12) as response:
            code = response.getcode()
            final_url = response.geturl()
            return {'status': 'OK', 'code': code, 'final_url': final_url, 'error': None}
    except urllib.error.HTTPError as e:
        return {'status': 'FAIL', 'code': e.code, 'final_url': url, 'error': f'HTTPError {e.code}'}
    except Exception as e:
        return {'status': 'FAIL', 'code': None, 'final_url': url, 'error': str(e)}

# Read all links in docs/datasets/catalog.md
with open('docs/datasets/catalog.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract links from the table
# format: | № | **Name** | Year | 1. Access | ... | [Title](URL) |
table_matches = re.findall(r'\|\s*\d+\s*\|\s*\*\*([^*]+)\*\*\s*\|[^|]+\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|[^|]+\|\s*\[([^\]]+)\]\(([^)]+)\)', text)

print(f"Total datasets found in table: {len(table_matches)}")
results = []

for match in table_matches:
    name = match[0].strip()
    access = match[1].strip()
    eyes = match[2].strip()
    data_type = match[3].strip()
    fmt = match[4].strip()
    link_title = match[5].strip()
    url = match[6].strip()
    
    is_open = 'Открытый' in access
    
    res = test_url(name, url)
    results.append({
        'name': name,
        'access': access,
        'is_open': is_open,
        'url': url,
        'link_title': link_title,
        'result': res
    })
    print(f"[{res['status']} {res['code']}] {name} ({access}): {url} -> {res['error'] or res['final_url']}")

with open('mvp/scratch/link_check_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
