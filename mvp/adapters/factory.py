import os, sys, torch
_mvp_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if _mvp_dir not in sys.path:
    sys.path.insert(0, _mvp_dir)

from typing import Optional

try:
    from adapters.base import BaseGazeAdapter
    from adapters.l2cs_adapter import L2CSAdapter
    from adapters.gazetr_adapter import GazeTRAdapter
    from adapters.puregaze_adapter import PureGazeAdapter
    from adapters.idiap_adapter import IdiapResNet18Adapter
    from adapters.yolo_adapter import YOLOAdapter
    from adapters.mediapipe_adapter import MediaPipeAdapter
    from adapters.ptgaze_eth_adapter import PTGazeETHAdapter
    from adapters.unigaze_adapter import UniGazeAdapter
    from adapters.yakhyo_adapter import YakhyoAdapter
    from adapters.ptgaze_mpii_adapter import PTGazeMPIIAdapter
    from adapters.qualcomm_adapter import QualcommEyeGazeAdapter
except ImportError:
    from mvp.adapters.base import BaseGazeAdapter
    from mvp.adapters.l2cs_adapter import L2CSAdapter
    from mvp.adapters.gazetr_adapter import GazeTRAdapter
    from mvp.adapters.puregaze_adapter import PureGazeAdapter
    from mvp.adapters.idiap_adapter import IdiapResNet18Adapter
    from mvp.adapters.yolo_adapter import YOLOAdapter
    from mvp.adapters.mediapipe_adapter import MediaPipeAdapter
    from mvp.adapters.ptgaze_eth_adapter import PTGazeETHAdapter
    from mvp.adapters.unigaze_adapter import UniGazeAdapter
    from mvp.adapters.yakhyo_adapter import YakhyoAdapter
    from mvp.adapters.ptgaze_mpii_adapter import PTGazeMPIIAdapter
    from mvp.adapters.qualcomm_adapter import QualcommEyeGazeAdapter

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
    elif key in ("ptgaze_eth_xgaze", "ptgaze_eth", "eth_xgaze_resnet18"):
        adapter = PTGazeETHAdapter(name="ptgaze_eth_xgaze", device=device)
    elif key in ("unigaze", "unigaze_b16", "unigaze_joint"):
        adapter = UniGazeAdapter(name="unigaze", device=device)
    elif key in ("yakhyo_resnet50", "yakhyo_r50", "yakhyo"):
        adapter = YakhyoAdapter(name="yakhyo_resnet50", device=device)
    elif key in ("yakhyo_resnet18", "yakhyo_r18"):
        adapter = YakhyoAdapter(name="yakhyo_resnet18", device=device)
    elif key in ("ptgaze_mpiifacegaze", "ptgaze_mpii", "mpiifacegaze_resnet"):
        adapter = PTGazeMPIIAdapter(name="ptgaze_mpiifacegaze", device=device)
    elif key in ("qualcomm_eyegaze", "qualcomm", "eyegaze_onnx"):
        adapter = QualcommEyeGazeAdapter(name="qualcomm_eyegaze", device=device)
    else:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Supported models: "
            "['puregaze', 'gazetr', 'unigaze', 'ptgaze_eth_xgaze', 'mediapipe', 'yolo26n', "
            "'yakhyo_resnet50', 'yakhyo_resnet18', 'ptgaze_mpiifacegaze', 'idiap_resnet18', 'l2cs', 'l2cs_diversity']"
        )

    adapter.load_weights(checkpoint_path=checkpoint_path)
    return adapter
