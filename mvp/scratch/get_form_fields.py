import urllib.request
import ssl
import re
import json

url = 'https://docs.google.com/forms/d/e/1FAIpQLSfZtMVpNbWV9yHX5toXVzVpDpOENy-SB7XfMIx5V6u7sITuNg/viewform?usp=sf_link'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

with urllib.request.urlopen(req, context=ctx) as r:
    html = r.read().decode('utf-8')

# FB_PUBLIC_LOAD_DATA_ contains all Google form fields
m = re.search(r'var FB_PUBLIC_LOAD_DATA_ = (.*?);</script>', html, re.DOTALL)
if m:
    data = json.loads(m.group(1))
    form_title = data[1][8] if len(data[1]) > 8 else "No title"
    form_desc = data[1][0]
    print(f"Form Title: {form_title}")
    print(f"Form Description: {form_desc[:300]}...\n")
    fields = data[1][1]
    for idx, f in enumerate(fields):
        title = f[1]
        desc = f[2]
        field_type = f[3]
        print(f"Field {idx+1}: {title}")
        if desc:
            print(f"   Desc: {desc}")
else:
    print("Could not find FB_PUBLIC_LOAD_DATA_")
