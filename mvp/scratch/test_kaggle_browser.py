import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request(
    "https://www.kaggle.com/datasets/michachwesiuk/hybridgaze",
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
)

try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Status code:", r.getcode())
        content = r.read().decode('utf-8', errors='ignore')
        print("Content preview:", content[:500])
        print("Title:", [l for l in content.split('\n') if '<title>' in l])
except urllib.error.HTTPError as e:
    print("HTTPError:", e.code)
    try:
        body = e.read().decode('utf-8', errors='ignore')
        print("Error body:", body[:500])
    except:
        pass
except Exception as e:
    print("Error:", e)
