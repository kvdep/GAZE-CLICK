import re

with open("mvp/scratch/hybridgaze_paper.pdf", "rb") as f:
    data = f.read()

urls = re.findall(b'https?://[a-zA-Z0-9./_~%-]+', data)
for u in sorted(list(set(urls))):
    print(u.decode('latin1', errors='ignore'))
