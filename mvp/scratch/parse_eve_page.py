import urllib.request
import ssl
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request('https://ait.ethz.ch/eve', headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')

    print('HTML length:', len(html))

    links = set(re.findall(r'href=["\'](https?://[^"\'>]+)["\']', html))
    print("\n=== ALL EXTERNAL LINKS ===")
    for l in sorted(links):
        print('Link:', l)

    # Search for google forms or dataset mentions
    print("\n=== DATASET / DOWNLOAD CONTEXT ===")
    clean_text = re.sub(r'<[^>]+>', ' ', html)
    clean_text = ' '.join(clean_text.split())
    for match in re.finditer(r'(download|dataset|form|request|access|terms)', clean_text, re.IGNORECASE):
        idx = match.start()
        print('--- SNIPPET ---')
        print(clean_text[max(0, idx-80):min(len(clean_text), idx+200)])
except Exception as e:
    print('Error:', e)
