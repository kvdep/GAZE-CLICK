import urllib.request
import ssl
import json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

candidates = [
    # HybridGaze
    ("HybridGaze Paper PAN", "https://journals.pan.pl/dlibra/publication/157892/edition/140228/content"),
    ("HybridGaze DOI", "https://doi.org/10.24425/ijet.2026.157892"),
    ("HybridGaze sqoshi github", "https://github.com/sqoshi"),
    
    # ChildPlay
    ("ChildPlay Idiap", "https://www.idiap.ch/en/dataset/childplay-gaze"),
    ("ChildPlay Zenodo", "https://zenodo.org/records/8254823"),
    ("ChildPlay GitHub", "https://github.com/idiap/geomgaze"),
    
    # WEyeDS
    ("WEyeDS Project Site", "https://robbinslab.github.io/weyeds/"),
    ("WEyeDS GitHub Processing", "https://github.com/robbinslab/weyeds-processing"),
    
    # LPW
    ("LPW DaRUS", "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-1718"),
    ("LPW PerceptualUI", "https://perceptualui.org/research/datasets/LPW/"),
    
    # GazeFollow
    ("GazeFollow HTTPS", "https://gazefollow.csail.mit.edu/"),
    ("GazeFollow HTTP", "http://gazefollow.csail.mit.edu/"),
    ("GazeFollow GitHub", "https://github.com/ejcgt/attention-target-detection"),
    
    # DR(eye)VE
    ("DREYEVE AImageLab", "https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8"),
    ("DREYEVE GitHub", "https://github.com/ndrplz/dreyeve"),
    
    # EGTEA Gaze+
    ("EGTEA Gaze+ HTTP", "http://cbs.ic.gatech.edu/fpv/"),
    ("EGTEA Gaze+ HTTPS", "https://cbs.ic.gatech.edu/fpv/"),
    ("EGTEA Gaze+ GitHub", "https://github.com/yinb/EGTEA-Gaze-Plus"),
    
    # UnityEyes
    ("UnityEyes Cambridge", "https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/"),
    ("UnityEyes GitHub Mirror", "https://github.com/TadasBaltrusaitis/UnityEyes"),
    ("UnityEyes2 GitHub", "https://github.com/errollw/UnityEyes2"),
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for label, url in candidates:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=8) as r:
            print(f"[OK {r.getcode()}] {label}: {url} -> {r.geturl()}")
    except Exception as e:
        print(f"[FAIL] {label}: {url} -> {e}")
