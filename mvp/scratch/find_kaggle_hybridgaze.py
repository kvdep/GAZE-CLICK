import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

candidates = [
    "https://www.kaggle.com/datasets/michalchwesiuk/hybridgaze",
    "https://www.kaggle.com/datasets/mchwesiuk/hybridgaze",
    "https://www.kaggle.com/datasets/piotrpopis/hybridgaze",
    "https://www.kaggle.com/datasets/sqoshi/hybridgaze",
    "https://www.kaggle.com/datasets/cypriumcuprum/hybridgaze",
    "https://www.kaggle.com/datasets/hybridgaze/hybridgaze",
    "https://www.kaggle.com/d/michalchwesiuk/hybridgaze",
    "https://www.kaggle.com/d/piotrpopis/hybridgaze"
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for url in candidates:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
            print(f"FOUND: {url} -> {resp.getcode()} (final: {resp.geturl()})")
    except Exception as e:
        print(f"FAILED: {url} -> {e}")
