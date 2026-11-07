import urllib.request
import ssl
import re

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://journals.pan.pl/dlibra/publication/157892/edition/138498/content"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        html = r.read().decode('utf-8', errors='ignore')
        print("Page fetched! Length:", len(html))
        # Find all pdf links or external links
        links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html)
        for l in links:
            if 'pdf' in l.lower() or 'kaggle' in l.lower() or 'github' in l.lower() or 'download' in l.lower():
                print("Interesting link:", l)
except Exception as e:
    print("Error:", e)
