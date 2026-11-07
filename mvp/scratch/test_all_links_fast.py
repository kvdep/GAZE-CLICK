import urllib.request
import urllib.error
import ssl
import json
from concurrent.futures import ThreadPoolExecutor

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
}

test_urls = {
    # 1. VideoAttentionTarget
    "VideoAttentionTarget": "https://github.com/ejcgt/attention-target-detection",
    
    # 2. ChildPlay Gaze
    "ChildPlay_old": "https://github.com/fkryan/childplay",
    "ChildPlay_idiap": "https://www.idiap.ch/en/dataset/childplay-gaze",
    "ChildPlay_zenodo": "https://zenodo.org/records/8254823",
    "ChildPlay_github": "https://github.com/idiap/geomgaze",
    
    # 3. WEyeDS
    "WEyeDS_old": "https://dl.acm.org/",
    "WEyeDS_project": "https://robbinslab.github.io/weyeds/",
    "WEyeDS_github": "https://github.com/robbinslab/weyeds-processing",
    
    # 4. Katrychuk VOG
    "Katrychuk_VOG": "https://zenodo.org/records/13931668",
    
    # 5. LPW
    "LPW_darus": "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-1718",
    "LPW_perceptualui": "https://perceptualui.org/research/datasets/LPW/",
    
    # 6. HybridGaze
    "HybridGaze_old_kaggle": "https://www.kaggle.com/datasets/grzegorzchwesiuk/hybridgaze",
    "HybridGaze_paper_doi": "https://doi.org/10.24425/ijet.2026.157892",
    "HybridGaze_pan": "https://journals.pan.pl/dlibra/publication/157892/edition/140228/content",
    
    # 7. RT-GENE
    "RT-GENE_zenodo": "https://zenodo.org/records/2529036",
    "RT-GENE_github": "https://github.com/Tobias-Fischer/rt_gene",
    
    # 8. VRGaze / GazeShift
    "VRGaze_huggingface": "https://huggingface.co/datasets/gazeshift/VRGaze",
    
    # 9. MPIIGaze
    "MPIIGaze_tar": "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz",
    "MPIIGaze_page": "https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild",
    
    # 10. MPIIFaceGaze
    "MPIIFaceGaze_zip": "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip",
    "MPIIFaceGaze_page": "https://perceptualui.org/research/datasets/MPIIFaceGaze/",
    
    # 11. Columbia Gaze
    "Columbia_Gaze": "https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/",
    
    # 12. GazeFollow
    "GazeFollow_https": "https://gazefollow.csail.mit.edu/",
    "GazeFollow_http": "http://gazefollow.csail.mit.edu/",
    "GazeFollow_github": "https://github.com/ejcgt/attention-target-detection",
    
    # 13. DrivFace
    "DrivFace_uci": "https://archive.ics.uci.edu/dataset/358/drivface",
    
    # 14. DR(eye)VE
    "DR(eye)VE_old": "https://imagelab.ing.unimore.it/dreve/",
    "DR(eye)VE_aimagelab": "https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8",
    "DR(eye)VE_github": "https://github.com/ndrplz/dreyeve",
    
    # 15. EGTEA Gaze+
    "EGTEA_https": "https://cbs.ic.gatech.edu/fpv/",
    "EGTEA_http": "http://cbs.ic.gatech.edu/fpv/",
    "EGTEA_github": "https://github.com/yinb/EGTEA-Gaze-Plus",
    
    # 16. GazeBase
    "GazeBase": "https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257",
    
    # 17. GazeBaseVR
    "GazeBaseVR": "https://figshare.com/articles/dataset/GazeBaseVR/19643442",
    
    # 18. SynthesEyes
    "SynthesEyes": "https://www.repository.cam.ac.uk/handle/1810/252736",
    
    # 19. UnityEyes
    "UnityEyes_old_github": "https://github.com/errollw/UnityEyes",
    "UnityEyes_cambridge": "https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/",
    "UnityEyes_tadas_github": "https://github.com/TadasBaltrusaitis/UnityEyes"
}

def check(item):
    k, u = item
    req = urllib.request.Request(u, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=7) as resp:
            code = resp.getcode()
            final_url = resp.geturl()
            res = {"status": "OK", "code": code, "final_url": final_url, "error": None}
    except urllib.error.HTTPError as e:
        res = {"status": "FAIL", "code": e.code, "final_url": u, "error": f"HTTP {e.code}"}
    except Exception as e:
        res = {"status": "FAIL", "code": None, "final_url": u, "error": str(e)}
    print(f"[{res['status']} {res['code']}] {k}: {u} -> {res.get('final_url', '')} (error: {res['error']})", flush=True)
    return k, u, res

results = {}
with ThreadPoolExecutor(max_workers=10) as ex:
    for k, u, r in ex.map(check, test_urls.items()):
        results[k] = {"url": u, "result": r}

with open("mvp/scratch/fast_link_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nFinished checking all URLs successfully!", flush=True)
