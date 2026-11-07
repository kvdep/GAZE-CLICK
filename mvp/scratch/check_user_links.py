import subprocess
import json

urls = [
    ("HybridGaze Kaggle (user link)", "https://www.kaggle.com/datasets/michachwesiuk/hybridgaze"),
    ("VRGaze HuggingFace (user link)", "https://huggingface.co/datasets/HungSteven/VRGaze"),
    ("GazeFollow HuggingFace (user link)", "https://huggingface.co/datasets/vikhyatk/gazefollow"),
    ("DrivFace UCI 378 (user link)", "https://archive.ics.uci.edu/dataset/378/drivface"),
    ("DREYEVE-mini Kaggle (user link)", "https://www.kaggle.com/datasets/hughiephan/dreyeve-mini"),
    ("EGTEA Gaze+ Motus (user link)", "https://www.patientcompanion.ca/motus/datasets/egtea-gaze-plus"),
    ("GazeBase Figshare File (user link)", "https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257?file=27039812"),
    ("SynthesEyes Cambridge DOI (user link)", "https://doi.org/10.17863/CAM.57942")
]

results = []
for label, url in urls:
    res = subprocess.run(["curl.exe", "-s", "-I", "-L", "-k", "-m", "10", url], capture_output=True, text=True, errors='ignore')
    status_line = res.stdout.split('\n')[0] if res.stdout else "No output"
    lines = [l for l in res.stdout.split('\n') if l.startswith('HTTP/')]
    last_status = lines[-1] if lines else status_line
    print(f"[{last_status.strip()}] {label}: {url}")
    results.append({"label": label, "url": url, "status": last_status.strip(), "headers": res.stdout})

with open("mvp/scratch/user_links_check.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
