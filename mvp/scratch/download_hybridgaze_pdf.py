import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://journals.pan.pl/Content/138498"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
        data = r.read()
        print("Downloaded bytes:", len(data), "ContentType:", r.headers.get('Content-Type'))
        with open("mvp/scratch/hybridgaze_paper.pdf", "wb") as f:
            f.write(data)
        print("Saved to mvp/scratch/hybridgaze_paper.pdf")
except Exception as e:
    print("Error:", e)
