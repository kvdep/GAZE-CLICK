import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Search Hugging Face for gaze estimation datasets
hf_url = "https://huggingface.co/api/datasets?search=gaze&limit=50"
req = urllib.request.Request(hf_url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        data = json.loads(r.read().decode('utf-8'))
        print(f"HuggingFace gaze datasets found: {len(data)}")
        for item in data:
            print(f"HF: {item['id']} (downloads: {item.get('downloads', 0)}, likes: {item.get('likes', 0)})")
except Exception as e:
    print("HF search error:", e)
