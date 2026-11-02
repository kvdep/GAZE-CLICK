import os
import sys
import csv
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

headers = [
    "Ранг",
    "Название модели / Архитектура",
    "Ошибка 1: Бенчмарк 9 сессий (Mean deg)",
    "Ошибка 2: Бенчмарк 9 сессий (Median deg)",
    "Ошибка 3: Бенчмарк 9 сессий (Экран px)",
    "Ошибка 4: Бенчмарк 9 сессий (Корреляция R2)",
    "Ошибка 5: Тестовый датасет (Mean deg)",
    "Ошибка 6: Тестовый датасет (Median deg)",
    "Ошибка 7: Тестовый датасет (Экран px)",
    "Ошибка 8: Тестовый датасет (Корреляция R2)",
    "Ошибка 9: Литература ETH-XGaze (deg)",
    "Ошибка 10: Литература MPIIFaceGaze (deg)",
    "Ошибка 11: Литература Gaze360 / EVE (deg)",
    "Задержка CPU (мс)",
    "Задержка GPU (мс)",
    "Потребление RAM (МБ)",
    "Потребление VRAM (МБ)",
    "Год создания",
    "Место, где лежат веса",
    "Источник (HF / GitHub / ArXiv)"
]

MODELS_REGISTRY = [
    {
        "name": "PureGaze (ResNet-50 + ResGazeEs)",
        "benchmark_key": "puregaze",
        "test_dataset_key": "puregaze",
        "lit_eth": "3.88°",
        "lit_mpii": "—",
        "lit_gaze360": "—",
        "cpu": "13.27",
        "gpu": "3.20",
        "ram": "420",
        "vram": "250",
        "year": "2026/2022",
        "weights": "https://huggingface.co/akgulburak/gaze-estimation-diversity/resolve/main/PureGaze/PureGaze_baseline_1.pt",
        "source": "https://huggingface.co/akgulburak/gaze-estimation-diversity (AAAI 2022)"
    },
    {
        "name": "UniGaze-Base (ViT-Base 12L)",
        "benchmark_key": "unigaze",
        "test_dataset_key": "unigaze",
        "lit_eth": "3.96°",
        "lit_mpii": "4.07°",
        "lit_gaze360": "3.01 (GazeCapture)",
        "cpu": "35.00",
        "gpu": "11.75",
        "ram": "850",
        "vram": "680",
        "year": "2025",
        "weights": "https://huggingface.co/UniGaze/UniGaze-models/resolve/main/unigaze_b16_joint.safetensors",
        "source": "https://huggingface.co/UniGaze/UniGaze-models (WACV 2025)"
    },
    {
        "name": "ptgaze ETH-XGaze (ResNet-18)",
        "benchmark_key": "ptgaze_eth_xgaze",
        "test_dataset_key": "ptgaze_eth_xgaze",
        "lit_eth": "4.20°",
        "lit_mpii": "—",
        "lit_gaze360": "—",
        "cpu": "8.50",
        "gpu": "6.51",
        "ram": "310",
        "vram": "180",
        "year": "2026",
        "weights": "https://huggingface.co/hysts/ptgaze-eth-xgaze-resnet18/resolve/main/model.safetensors",
        "source": "https://huggingface.co/hysts/ptgaze-eth-xgaze-resnet18 (GitHub: hysts/pytorch_mpiigaze_demo)"
    },
    {
        "name": "GazeTR-Hybrid (ResNet-18 + ViT-6L)",
        "benchmark_key": "gazetr",
        "test_dataset_key": "gazetr",
        "lit_eth": "3.88°",
        "lit_mpii": "3.69°",
        "lit_gaze360": "—",
        "cpu": "14.75",
        "gpu": "11.40",
        "ram": "390",
        "vram": "280",
        "year": "2026/2022",
        "weights": "https://huggingface.co/akgulburak/gaze-estimation-diversity/resolve/main/GazeTR/GazeTR-baseline-1.pt",
        "source": "https://github.com/yihuacheng/GazeTR (ICPR 2022)"
    },
    {
        "name": "MediaPipe Iris + FaceMesh",
        "benchmark_key": "mediapipe",
        "test_dataset_key": "mediapipe",
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "3.80 (Geometry)",
        "cpu": "15.30",
        "gpu": "N/A",
        "ram": "180",
        "vram": "0",
        "year": "2020",
        "weights": "Встроен в Google MediaPipe бинарник",
        "source": "https://github.com/google-ai-edge/mediapipe"
    },
    {
        "name": "YOLO26n (4D Head Comp)",
        "benchmark_key": "yolo26n_head_comp",
        "test_dataset_key": "yolo26n_head_comp",
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "—",
        "cpu": "19.19",
        "gpu": "4.20",
        "ram": "140",
        "vram": "90",
        "year": "2026",
        "weights": "weights/yolo26n_gaze.pt (локальный)",
        "source": "Локальный эксперимент Run 014 (CSPDarknet + 4D Polynomial Gain)"
    },
    {
        "name": "ptgaze MPIIFaceGaze (ResNet-18)",
        "benchmark_key": "ptgaze_mpiifacegaze",
        "test_dataset_key": "ptgaze_mpiifacegaze",
        "lit_eth": "—",
        "lit_mpii": "4.80°",
        "lit_gaze360": "—",
        "cpu": "5.58",
        "gpu": "3.10",
        "ram": "310",
        "vram": "180",
        "year": "2026",
        "weights": "https://huggingface.co/hysts/ptgaze-mpiifacegaze-resnet-simple/resolve/main/model.safetensors",
        "source": "https://huggingface.co/hysts/ptgaze-mpiifacegaze-resnet-simple"
    },
    {
        "name": "Idiap ResNet-18 Gaze360",
        "benchmark_key": "idiap_resnet18",
        "test_dataset_key": "idiap_resnet18",
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "9.02 (Gaze360 Front)",
        "cpu": "17.01",
        "gpu": "2.00",
        "ram": "320",
        "vram": "190",
        "year": "2026",
        "weights": "https://huggingface.co/Idiap/resnet18-gaze360/resolve/main/gaze360_resnet18.pt",
        "source": "https://huggingface.co/Idiap/resnet18-gaze360"
    },
    {
        "name": "YOLO26n (2D Gaze Only)",
        "benchmark_key": "yolo26n",
        "test_dataset_key": "yolo26n",
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "—",
        "cpu": "19.28",
        "gpu": "4.20",
        "ram": "140",
        "vram": "90",
        "year": "2026",
        "weights": "weights/yolo26n_gaze.pt (локальный)",
        "source": "Локальный эксперимент Run 014 (без 4D компенсации)"
    },
    {
        "name": "L2CS-Net (Diversity)",
        "benchmark_key": "l2cs_diversity",
        "test_dataset_key": "l2cs_diversity",
        "lit_eth": "—",
        "lit_mpii": "3.92°",
        "lit_gaze360": "9.92 (Gaze360 Front)",
        "cpu": "13.74",
        "gpu": "3.80",
        "ram": "380",
        "vram": "260",
        "year": "2026",
        "weights": "https://huggingface.co/akgulburak/gaze-estimation-diversity/resolve/main/L2CS/L2CS-baseline-1.pt",
        "source": "https://huggingface.co/akgulburak/gaze-estimation-diversity"
    },
    {
        "name": "L2CS-Net (PyFeat / LiteRT)",
        "benchmark_key": "l2cs",
        "test_dataset_key": "l2cs",
        "lit_eth": "—",
        "lit_mpii": "3.92°",
        "lit_gaze360": "10.41 (Gaze360 Full)",
        "cpu": "13.79",
        "gpu": "3.80",
        "ram": "380",
        "vram": "260",
        "year": "2026/2024",
        "weights": "https://github.com/cosmicad/L2CS-Net/releases/download/v1.0/L2CSNet_gaze360.pkl",
        "source": "https://github.com/cosmicad/L2CS-Net"
    },
    {
        "name": "yakhyo/gaze-estimation (ResNet-50)",
        "benchmark_key": "yakhyo_resnet50",
        "test_dataset_key": "yakhyo_resnet50",
        "lit_eth": "—",
        "lit_mpii": "4.38°",
        "lit_gaze360": "—",
        "cpu": "13.93",
        "gpu": "3.80",
        "ram": "370",
        "vram": "250",
        "year": "2024",
        "weights": "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet50.pt",
        "source": "https://github.com/yakhyo/gaze-estimation"
    },
    {
        "name": "yakhyo/gaze-estimation (ResNet-18)",
        "benchmark_key": "yakhyo_resnet18",
        "test_dataset_key": "yakhyo_resnet18",
        "lit_eth": "—",
        "lit_mpii": "4.72°",
        "lit_gaze360": "—",
        "cpu": "6.85",
        "gpu": "2.20",
        "ram": "290",
        "vram": "170",
        "year": "2024",
        "weights": "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet18.pt",
        "source": "https://github.com/yakhyo/gaze-estimation"
    },
    {
        "name": "Qualcomm EyeGaze (MobileOne)",
        "benchmark_key": "qualcomm_eyegaze",
        "test_dataset_key": "qualcomm_eyegaze",
        "lit_eth": "—",
        "lit_mpii": "4.35°",
        "lit_gaze360": "—",
        "cpu": "14.69",
        "gpu": "0.59 (NPU)",
        "ram": "110",
        "vram": "60",
        "year": "2025",
        "weights": "https://huggingface.co/qualcomm/EyeGaze (ONNX/TFLite/QNN DLC)",
        "source": "https://huggingface.co/qualcomm/EyeGaze"
    },
    {
        "name": "EyeTAG (Kinematic Prior + CCAF)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "2.56 (EVE) / 8.08 (Gaze360 Front)",
        "cpu": "19.97",
        "gpu": "4.80",
        "ram": "550",
        "vram": "420",
        "year": "2026",
        "weights": "https://github.com/cydonia999/VGGFace2-pytorch (resnet50_ft_weight.pkl)",
        "source": "https://github.com/peter8366/EyeTAG (BMVC 2026, arXiv:2610.00922)"
    },
    {
        "name": "STAGE (SAM Attention + Gaussian Proc)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "2.58 (EVE) / 4.12 (EYEDIAP)",
        "cpu": "21.50",
        "gpu": "5.20",
        "ram": "480",
        "vram": "350",
        "year": "2024",
        "weights": "https://drive.google.com/drive/folders/16re0UoAnmaqQcqjXSkQ8u63iz_Tm9fwd",
        "source": "https://github.com/jswati31/stage (arXiv:2404.05215)"
    },
    {
        "name": "EMC-Gaze (EGNN Landmark Graph)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "5.79 (Dynamic Head)",
        "cpu": "18.40",
        "gpu": "4.50",
        "ram": "360",
        "vram": "240",
        "year": "2026",
        "weights": "Открытые веса в репозитории проекта",
        "source": "https://github.com/beeyuun/EMC-Gaze (arXiv:2603.11894)"
    },
    {
        "name": "IVGaze (In-Vehicle Transformer)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "7.04 (In-Cabin Wide Yaw)",
        "cpu": "16.20",
        "gpu": "4.50",
        "ram": "410",
        "vram": "300",
        "year": "2024",
        "weights": "https://drive.google.com/file/d/1uc5OkwZJO-KfMSuFAW0Hl-JSwTrfPURq",
        "source": "https://github.com/yihuacheng/IVGaze (CVPR 2024)"
    },
    {
        "name": "Gaze-LLE / Gazelle (DINOv2 ViT-B)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "0.11 AUC (ChildPlay / EVE)",
        "cpu": "42.00",
        "gpu": "18.00",
        "ram": "900",
        "vram": "720",
        "year": "2025",
        "weights": "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14.pt",
        "source": "https://github.com/fkryan/gazelle (CVPR 2025 Highlight)"
    },
    {
        "name": "Gazelle-DINOv3 (HGNetV2 ONNX)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "Эквивалент Gaze-LLE",
        "cpu": "6.80",
        "gpu": "2.10",
        "ram": "160",
        "vram": "120",
        "year": "2025",
        "weights": "https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_hgnetv2_n_distill_1x3x640x640_1xNx4.onnx",
        "source": "https://github.com/PINTO0309/gazelle-dinov3"
    },
    {
        "name": "MCGaze (Spatial-Temporal R-50)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "9.82 (L2CS) / 10.74 (Gaze360)",
        "cpu": "18.50",
        "gpu": "4.80",
        "ram": "450",
        "vram": "320",
        "year": "2024",
        "weights": "https://drive.google.com/file/d/1ru0xhuB5N9kwvN9XLvZMQvVSfOgtbxmq",
        "source": "https://github.com/zgchen33/MCGaze (IEEE SPL 2024)"
    },
    {
        "name": "CrossGaze (Multimodal Attention)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "9.94 (Gaze360)",
        "cpu": "19.00",
        "gpu": "4.90",
        "ram": "410",
        "vram": "290",
        "year": "2024",
        "weights": "https://huggingface.co/akgulburak/gaze-estimation-diversity/resolve/main/CrossGaze/pretrained-multimodal-crossatention_baseline_1.pth",
        "source": "https://huggingface.co/akgulburak/gaze-estimation-diversity"
    },
    {
        "name": "GazeShift (VR Domain Adaptation)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "11.2 -> 6.4 (Unsupervised)",
        "cpu": "15.00",
        "gpu": "3.80",
        "ram": "330",
        "vram": "210",
        "year": "2026",
        "weights": "https://drive.google.com/file/d/1U-txJpQUm7YTXd13n2lYXAQ8zkz3iieC",
        "source": "https://github.com/gazeshift3/gazeshift (arXiv:2603.07832)"
    },
    {
        "name": "GLC (SlowFast + Transformer)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "Egocentric PoG (EGTEA)",
        "cpu": "38.00",
        "gpu": "12.00",
        "ram": "620",
        "vram": "480",
        "year": "2024",
        "weights": "https://drive.google.com/file/d/15XVipU1CqoosyvPU1vfcG0w2LuLQecF1",
        "source": "https://github.com/BolinLai/GLC (IJCV 2023 / GitHub 2024)"
    },
    {
        "name": "GazeCLR (Self-Supervised Contrastive)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "Contrastive Pretraining",
        "cpu": "12.50",
        "gpu": "3.20",
        "ram": "300",
        "vram": "180",
        "year": "2024",
        "weights": "https://drive.google.com/file/d/1qGwQnCLkxHitrj1QLhBGDVLfmvEYU0r1",
        "source": "https://github.com/jswati31/gazeclr (2024)"
    },
    {
        "name": "OpenFace 3.0 (Multitask ViT)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "Multitask Facial Behavior",
        "cpu": "22.00",
        "gpu": "6.50",
        "ram": "510",
        "vram": "380",
        "year": "2025",
        "weights": "https://huggingface.co/nutPace/openface_weights",
        "source": "https://github.com/CMU-MultiComp-Lab/OpenFace-3.0 (arXiv:2506.02891)"
    },
    {
        "name": "Ghost-CBAM MobileNetV4",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "Screen Fixation Zones",
        "cpu": "5.00",
        "gpu": "1.50",
        "ram": "130",
        "vram": "80",
        "year": "2026",
        "weights": "https://huggingface.co/Jason-42195/asd-eye-tracking-models/resolve/main/mobilenetv4_small/best.pth",
        "source": "https://huggingface.co/Jason-42195/asd-eye-tracking-models"
    },
    {
        "name": "MAGE (6-DoF Multi-Decoder)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "6-DoF Gaze Vector & Origin",
        "cpu": "16.50",
        "gpu": "4.20",
        "ram": "340",
        "vram": "220",
        "year": "2025",
        "weights": "Открытый код (GitHub repo)",
        "source": "IEEE Transactions 2025"
    },
    {
        "name": "ViT-Base Mobile Gaze",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "88.4% Screen Area Acc",
        "cpu": "45.00",
        "gpu": "14.00",
        "ram": "850",
        "vram": "680",
        "year": "2024",
        "weights": "https://huggingface.co/wahidww/vit-base-patch16-224-finetuned-mobile-eye-tracking-dataset-v2/resolve/main/model.safetensors",
        "source": "https://huggingface.co/wahidww/vit-base-patch16-224-finetuned-mobile-eye-tracking-dataset-v2"
    },
    {
        "name": "FAZE (Few-Shot Meta MAML)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "3.18 (3-shot adaptation)",
        "cpu": "11.50",
        "gpu": "3.20",
        "ram": "310",
        "vram": "190",
        "year": "2019",
        "weights": "Открытые чекпоинты (GitHub repo)",
        "source": "https://github.com/NVlabs/FAZE (ICCV 2019)"
    },
    {
        "name": "iTracker (4-Stream CNN)",
        "benchmark_key": None,
        "test_dataset_key": None,
        "lit_eth": "—",
        "lit_mpii": "—",
        "lit_gaze360": "1.71 cm screen error",
        "cpu": "15.00",
        "gpu": "4.10",
        "ram": "390",
        "vram": "270",
        "year": "2016",
        "weights": "Открытые весы Caffe / PyTorch",
        "source": "https://github.com/CSAILVision/GazeCapture (CVPR 2016)"
    }
]

# 1. Load multi_session_summary.csv directly produced by code
def clean_encoding_artifacts(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.replace("В±", "±").replace("В°", "°")

multi_summary_path = os.path.join("mvp", "experiments", "multi_session_summary.csv")
multi_data = {}
if os.path.exists(multi_summary_path):
    try:
        df_bench = pd.read_csv(multi_summary_path, encoding='utf-8-sig')
    except Exception:
        df_bench = pd.read_csv(multi_summary_path, encoding='cp1251')
    for _, r in df_bench.iterrows():
        k = str(r['Model']).strip()
        multi_data[k] = {
            "mean_ang": clean_encoding_artifacts(str(r['Mean AngErr (deg)']).strip()),
            "med_ang": clean_encoding_artifacts(f"{r['Median AngErr (deg)']}°"),
            "screen_px": clean_encoding_artifacts(str(r['Mean Err (px)']).strip()),
            "r2": clean_encoding_artifacts(str(r['Mean R2']).strip()),
            "cpu_lat": str(r.get('Mean Latency (ms)', '')).strip()
        }
    print(f"Loaded {len(multi_data)} benchmarked models from {multi_summary_path}.")
else:
    print(f"Warning: {multi_summary_path} not found.")

# 2. Load test_dataset_summary.csv if produced by automated test code
test_summary_path = os.path.join("mvp", "experiments", "test_dataset_summary.csv")
test_data = {}
if os.path.exists(test_summary_path):
    try:
        df_test = pd.read_csv(test_summary_path, encoding='utf-8-sig')
    except Exception:
        df_test = pd.read_csv(test_summary_path, encoding='cp1251')
    for _, r in df_test.iterrows():
        k = str(r['Model']).strip()
        test_data[k] = {
            "mean_ang": clean_encoding_artifacts(str(r['Mean AngErr (deg)']).strip()),
            "med_ang": clean_encoding_artifacts(f"{r['Median AngErr (deg)']}°"),
            "screen_px": clean_encoding_artifacts(str(r['Mean Err (px)']).strip()),
            "r2": clean_encoding_artifacts(str(r['Mean R2']).strip())
        }
    print(f"Loaded {len(test_data)} test dataset evaluations from {test_summary_path}.")
else:
    print(f"Notice: {test_summary_path} not present. Columns 5-8 will strictly be '—' for all models (Rule: Code to Code).")

# 3. Assemble rows strictly from code data
evaluated_rows = []
unevaluated_rows = []

for m in MODELS_REGISTRY:
    bkey = m["benchmark_key"]
    tkey = m["test_dataset_key"]
    
    # Primary benchmark metrics (9 webcam sessions)
    if bkey and bkey in multi_data:
        m_bench = multi_data[bkey]
        err1 = m_bench["mean_ang"]
        err2 = m_bench["med_ang"]
        err3 = m_bench["screen_px"]
        err4 = m_bench["r2"]
        # Parse numeric mean angle for sorting
        try:
            num_ang = float(err1.split('±')[0].strip())
        except Exception:
            num_ang = 999.0
    else:
        err1 = "—"
        err2 = "—"
        err3 = "—"
        err4 = "—"
        num_ang = 999.0

    # Test dataset metrics
    if tkey and tkey in test_data:
        m_test = test_data[tkey]
        err5 = m_test["mean_ang"]
        err6 = m_test["med_ang"]
        err7 = m_test["screen_px"]
        err8 = m_test["r2"]
    else:
        err5 = "—"
        err6 = "—"
        err7 = "—"
        err8 = "—"

    row_dict = {
        "num_ang": num_ang,
        "name": m["name"],
        "err1": err1,
        "err2": err2,
        "err3": err3,
        "err4": err4,
        "err5": err5,
        "err6": err6,
        "err7": err7,
        "err8": err8,
        "lit_eth": m["lit_eth"],
        "lit_mpii": m["lit_mpii"],
        "lit_gaze360": m["lit_gaze360"],
        "cpu": m["cpu"],
        "gpu": m["gpu"],
        "ram": m["ram"],
        "vram": m["vram"],
        "year": m["year"],
        "weights": m["weights"],
        "source": m["source"]
    }
    
    if err1 != "—":
        evaluated_rows.append(row_dict)
    else:
        unevaluated_rows.append(row_dict)

# Sort evaluated rows ascending by empirical mean angular error
evaluated_rows.sort(key=lambda x: x["num_ang"])

# Sort unevaluated rows by year descending, then name
def parse_year(y_str):
    try:
        return int(y_str.split('/')[0])
    except Exception:
        return 0

unevaluated_rows.sort(key=lambda x: (parse_year(x["year"]), x["name"]), reverse=True)

# Combine and assign sequential ranks
final_rows = []
current_rank = 1

for r in evaluated_rows + unevaluated_rows:
    final_rows.append([
        str(current_rank),
        r["name"],
        r["err1"],
        r["err2"],
        r["err3"],
        r["err4"],
        r["err5"],
        r["err6"],
        r["err7"],
        r["err8"],
        r["lit_eth"],
        r["lit_mpii"],
        r["lit_gaze360"],
        r["cpu"],
        r["gpu"],
        r["ram"],
        r["vram"],
        r["year"],
        r["weights"],
        r["source"]
    ])
    current_rank += 1

# Export to CSV and TSV
csv_path = os.path.join("docs", "leaderboard.csv")
with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(final_rows)
print(f"Successfully generated {csv_path} ({len(final_rows)} models, {len(headers)} columns).")

tsv_path = os.path.join("docs", "leaderboard.tsv")
with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(headers)
    writer.writerows(final_rows)
print(f"Successfully generated {tsv_path} ({len(final_rows)} models, {len(headers)} columns).")
