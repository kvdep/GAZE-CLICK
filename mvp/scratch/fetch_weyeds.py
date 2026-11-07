import subprocess

res = subprocess.run(["curl.exe", "-s", "-L", "https://robbinslab.github.io/weyeds/"], capture_output=True, text=True, encoding='utf-8', errors='ignore')
print(res.stdout[:2000])

with open("mvp/scratch/weyeds_page.html", "w", encoding="utf-8") as f:
    f.write(res.stdout)
