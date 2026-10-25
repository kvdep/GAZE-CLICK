import os, torch
from typing import Optional

from adapters.base import BaseGazeAdapter
from adapters.l2cs_adapter import L2CSAdapter
from adapters.gazetr_adapter import GazeTRAdapter
from adapters.puregaze_adapter import PureGazeAdapter
from adapters.idiap_adapter import IdiapResNet18Adapter
from adapters.yolo_adapter import YOLOAdapter
from adapters.mediapipe_adapter import MediaPipeAdapter

def create_adapter(model_name: str, device: torch.device, checkpoint_path: Optional[str] = None) -> BaseGazeAdapter:
    """
    Factory function instantiating the model-specific Gaze Adapter.
    Encapsulates preprocessing, architecture, inference, and coordinate conversions.
    """
    key = model_name.strip().lower()

    if key in ("l2cs", "l2cs-net", "l2cs_pyfeat"):
        adapter = L2CSAdapter(name="l2cs", device=device, variant="pyfeat")
    elif key in ("l2cs_diversity", "l2cs-diversity", "l2csnet_diversity"):
        adapter = L2CSAdapter(name="l2cs_diversity", device=device, variant="diversity")
    elif key in ("gazetr", "gaze_tr"):
        adapter = GazeTRAdapter(name="gazetr", device=device)
    elif key in ("puregaze", "pure_gaze"):
        adapter = PureGazeAdapter(name="puregaze", device=device)
    elif key in ("idiap", "idiap_resnet18", "resnet18-gaze360", "gaze360_resnet18"):
        adapter = IdiapResNet18Adapter(name="idiap_resnet18", device=device)
    elif key in ("yolo", "yolo26n", "yolo26n_gaze_net"):
        adapter = YOLOAdapter(name="yolo26n", device=device)
    elif key in ("mediapipe", "iris"):
        adapter = MediaPipeAdapter(name="mediapipe", device=device)
    else:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Supported models: "
            "['l2cs', 'l2cs_diversity', 'gazetr', 'puregaze', 'idiap_resnet18', 'yolo26n', 'mediapipe']"
        )

    adapter.load_weights(checkpoint_path=checkpoint_path)
    return adapter
