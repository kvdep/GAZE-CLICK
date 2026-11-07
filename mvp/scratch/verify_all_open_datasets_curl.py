import subprocess
import json
import re

open_datasets = [
    {
        "id": 1,
        "name": "VideoAttentionTarget",
        "current_url": "https://github.com/ejcgt/attention-target-detection",
        "description": "Официальный GitHub репозиторий датасета с кодом и ссылками на скачивание"
    },
    {
        "id": 2,
        "name": "ChildPlay Gaze",
        "current_url": "https://github.com/fkryan/childplay",
        "alt_urls": [
            "https://www.idiap.ch/en/dataset/childplay-gaze",
            "https://zenodo.org/records/8254823",
            "https://github.com/idiap/geomgaze"
        ]
    },
    {
        "id": 3,
        "name": "WEyeDS",
        "current_url": "https://dl.acm.org/",
        "alt_urls": [
            "https://robbinslab.github.io/weyeds/",
            "https://github.com/robbinslab/weyeds-processing"
        ]
    },
    {
        "id": 4,
        "name": "Katrychuk VOG",
        "current_url": "https://zenodo.org/records/13931668",
        "alt_urls": [
            "https://zenodo.org/records/13931669",
            "https://doi.org/10.5281/zenodo.13931669"
        ]
    },
    {
        "id": 5,
        "name": "LPW",
        "current_url": "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-1718",
        "alt_urls": [
            "https://doi.org/10.18419/darus-3373",
            "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-3373",
            "https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-estimation/labelled-pupils-in-the-wild-lpw"
        ]
    },
    {
        "id": 6,
        "name": "HybridGaze",
        "current_url": "https://www.kaggle.com/datasets/grzegorzchwesiuk/hybridgaze",
        "alt_urls": [
            "https://doi.org/10.24425/ijet.2026.157892",
            "https://journals.pan.pl/dlibra/publication/157892/edition/138498/content"
        ]
    },
    {
        "id": 7,
        "name": "RT-GENE",
        "current_url": "https://zenodo.org/records/2529036",
        "alt_urls": [
            "https://github.com/Tobias-Fischer/rt_gene"
        ]
    },
    {
        "id": 8,
        "name": "VRGaze / GazeShift",
        "current_url": "https://huggingface.co/datasets/gazeshift/VRGaze",
        "alt_urls": [
            "https://huggingface.co/datasets/gazeshift/VRGaze"
        ]
    },
    {
        "id": 9,
        "name": "MPIIGaze",
        "current_url": "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz",
        "alt_urls": [
            "https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild"
        ]
    },
    {
        "id": 10,
        "name": "MPIIFaceGaze",
        "current_url": "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip",
        "alt_urls": [
            "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip",
            "https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild"
        ]
    },
    {
        "id": 11,
        "name": "Columbia Gaze",
        "current_url": "https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/",
        "alt_urls": [
            "https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/"
        ]
    },
    {
        "id": 12,
        "name": "GazeFollow",
        "current_url": "https://gazefollow.csail.mit.edu/",
        "alt_urls": [
            "https://github.com/recasens/Gaze-Following",
            "https://github.com/ejcgt/attention-target-detection"
        ]
    },
    {
        "id": 13,
        "name": "DrivFace",
        "current_url": "https://archive.ics.uci.edu/dataset/358/drivface",
        "alt_urls": [
            "https://archive.ics.uci.edu/dataset/358/drivface"
        ]
    },
    {
        "id": 14,
        "name": "DR(eye)VE",
        "current_url": "https://imagelab.ing.unimore.it/dreve/",
        "alt_urls": [
            "https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8",
            "https://github.com/ndrplz/dreyeve"
        ]
    },
    {
        "id": 15,
        "name": "EGTEA Gaze+",
        "current_url": "https://cbs.ic.gatech.edu/fpv/",
        "alt_urls": [
            "https://github.com/amitsou/EGTEA_Gaze_Plus_Downloader"
        ]
    },
    {
        "id": 16,
        "name": "GazeBase",
        "current_url": "https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257",
        "alt_urls": [
            "https://doi.org/10.6084/m9.figshare.12912257"
        ]
    },
    {
        "id": 17,
        "name": "GazeBaseVR",
        "current_url": "https://figshare.com/articles/dataset/GazeBaseVR/19643442",
        "alt_urls": [
            "https://doi.org/10.6084/m9.figshare.19643442"
        ]
    },
    {
        "id": 18,
        "name": "SynthesEyes",
        "current_url": "https://www.repository.cam.ac.uk/handle/1810/252736",
        "alt_urls": [
            "https://www.repository.cam.ac.uk/items/b7817912-50b5-423b-882e-978fb39a49df"
        ]
    },
    {
        "id": 19,
        "name": "UnityEyes",
        "current_url": "https://github.com/errollw/UnityEyes",
        "alt_urls": [
            "https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/"
        ]
    }
]

def check_url_curl(url):
    try:
        res = subprocess.run(
            ["curl.exe", "-I", "-L", "-k", "-s", "-m", "10", url],
            capture_output=True,
            text=True
        )
        output = res.stdout
        # Extract all HTTP status lines
        status_lines = re.findall(r'HTTP/[12\.]+\s+(\d+)', output)
        if status_lines:
            last_code = int(status_lines[-1])
            return {"status": "OK" if last_code in [200, 202, 301, 302] else "FAIL", "code": last_code, "history": status_lines}
        return {"status": "FAIL", "code": None, "error": "No HTTP status returned (timeout or DNS failure)"}
    except Exception as e:
        return {"status": "FAIL", "code": None, "error": str(e)}

report = []
for ds in open_datasets:
    print(f"\n--- Checking {ds['name']} ---")
    cur_res = check_url_curl(ds["current_url"])
    print(f"Current link: {ds['current_url']} -> {cur_res}")
    
    verified_link = ds["current_url"] if cur_res["status"] == "OK" and cur_res["code"] in [200, 202] else None
    
    tested_alts = []
    if "alt_urls" in ds:
        for alt in ds["alt_urls"]:
            alt_res = check_url_curl(alt)
            print(f"  Alt link: {alt} -> {alt_res}")
            tested_alts.append({"url": alt, "res": alt_res})
            if verified_link is None and alt_res["status"] == "OK" and alt_res["code"] in [200, 202]:
                verified_link = alt
                
    report.append({
        "id": ds["id"],
        "name": ds["name"],
        "current_url": ds["current_url"],
        "current_res": cur_res,
        "verified_link": verified_link,
        "alts": tested_alts
    })

with open("mvp/scratch/open_datasets_verification_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\nVerification report complete!")
