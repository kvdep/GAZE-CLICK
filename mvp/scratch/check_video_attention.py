import subprocess
import re

res = subprocess.run(["curl.exe", "-s", "-L", "https://raw.githubusercontent.com/ejcgt/attention-target-detection/master/README.md"], capture_output=True, text=True, encoding='utf-8', errors='ignore')
print(res.stdout[:2000])
