import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = 'https://api.github.com/repos/swook/EVE/issues?state=all'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        issues = json.loads(resp.read().decode('utf-8'))
    print(f"GitHub Issues count: {len(issues)}")
    for issue in issues:
        num = issue.get("number")
        title = issue.get("title")
        state = issue.get("state")
        body = issue.get("body", "") or ""
        print(f"Issue #{num}: {title} [{state}]")
        for line in body.splitlines()[:5]:
            print(f"   {line}")
        print("-" * 50)
except Exception as e:
    print("Error:", e)
