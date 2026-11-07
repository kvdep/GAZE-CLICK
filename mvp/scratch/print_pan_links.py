import urllib.request
import ssl
import re

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://journals.pan.pl/dlibra/publication/157892/edition/138498/content"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
    html = r.read().decode('utf-8', errors='ignore')

# print all text or links
links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html)
for l in set(links):
    print("LINK:", l)
