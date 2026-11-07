import subprocess

res = subprocess.run(["curl.exe", "-s", "-L", "https://www.idiap.ch/en/dataset/childplay-gaze"], capture_output=True, text=True, encoding='utf-8', errors='ignore')
print(res.stdout[:3000])
