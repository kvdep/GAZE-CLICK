import sys, os
sys.path.insert(0, os.path.abspath("mvp"))
import torch
from adapters.factory import create_adapter

device = torch.device("cpu")
models = ["yolo26n", "l2cs", "l2cs_diversity", "gazetr", "puregaze", "mediapipe"]

for m in models:
    try:
        ad = create_adapter(m, device)
        print(f"[OK] {m}: {ad.name} initialized successfully.")
    except Exception as e:
        print(f"[FAIL] {m}: {type(e).__name__}: {e}")

try:
    ad = create_adapter("idiap_resnet18", device)
    print(f"[OK] idiap_resnet18: {ad.name} initialized successfully.")
except Exception as e:
    print(f"[FAIL] idiap_resnet18: {type(e).__name__}: {e}")
