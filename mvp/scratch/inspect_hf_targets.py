import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

targets = [
    "Morning5/Gaze360",
    "goatman/synthetic_gaze_prediction",
    "goatman/metahuman-gaze-prediction",
    "Tony2202/mono_gaze"
]

for t in targets:
    url = f"https://huggingface.co/api/datasets/{t}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=8) as r:
            info = json.loads(r.read().decode())
            print(f"Dataset {t}:")
            print("  description:", info.get('description', '')[:200].strip())
            print("  files:", [s['rfilename'] for s in info.get('siblings', [])[:10]])
            print("  storage:", info.get('usedStorage', 0) / (1024*1024), "MB")
    except Exception as e:
        print(f"Error for {t}:", e)
