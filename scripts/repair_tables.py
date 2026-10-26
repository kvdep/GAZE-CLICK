# Script to cleanly replace tables without regex escape bugs

with open('docs/09_ready_models_and_aggregators.md', 'r', encoding='utf-8') as f:
    text09 = f.read()

header09 = "| Ранг | Архитектура / Модель | Хостинг / Источник весов |"
footer09 = "\n\n---\n\n## 3. Статистика и объемы датасетов"

start09 = text09.find(header09)
end09 = text09.find(footer09)

if start09 != -1 and end09 != -1:
    correct_table_09 = """| Ранг | Архитектура / Модель | Хостинг / Источник весов | Год | Метрика $\\mathcal{E}_{ang}$ (Бенчмарк 9 сессий) | Метрика $\\mathcal{E}_{ang}$ (Литература) | Размер обучающего датасета | Параметры | Задержка CPU/GPU ($T_{inf}$) | Доступность весов |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **PureGaze** (ResNet-50 + ResGazeEs) | HF: `akgulburak` | 2026/22 | **$1.85^\\circ \\pm 0.44^\\circ$** (Med: **$1.70^\\circ$**) | $3.88^\\circ$ (ETH-XGaze) | ETH-XGaze ($1.08\\text{M}$) + MPII ($45\\text{k}$) | $25.6\\text{M}$ | **$13.27\\text{ мс}$** (GPU: $3.2\\text{ мс}$) | Прямой `.pt` на HF |
| **2** | **UniGaze-Base** (ViT-Base 12L) | HF: `UniGaze` / GH | 2025 | **$1.97^\\circ \\pm 0.60^\\circ$** (Med: **$1.64^\\circ$**) | **$3.96^\\circ$** (XGaze), **$4.07^\\circ$** (MPII) | **$1.65\\text{ млн}$** (264k персон) | $86.0\\text{M}$ | **$11.75\\text{ мс}$** (GPU RTX 4070) | Safetensors на HF ($345\\text{ МБ}$) |
| **3** | **ptgaze ETH-XGaze** (ResNet-18) | HF: `hysts` | 2026 | **$2.21^\\circ \\pm 0.34^\\circ$** (Med: **$1.92^\\circ$**) | **$4.20^\\circ$** (ETH-XGaze) | ETH-XGaze ($1.08\\text{M}$) | $11.7\\text{M}$ | **$6.51\\text{ мс}$** (наименьшая) | Safetensors на HF ($46.8\\text{ МБ}$) |
| **4** | **GazeTR-Hybrid** (ResNet-18 + ViT-6L) | HF: `akgulburak` / GH | 2026/22 | **$2.26^\\circ \\pm 0.47^\\circ$** (Med: **$2.03^\\circ$**) | $3.69^\\circ$ (MPII), $3.88^\\circ$ (XGaze) | ETH-XGaze ($1.08\\text{M}$) + MPII ($45\\text{k}$) | $18.4\\text{M}$ | **$14.75\\text{ мс}$** (GPU: $11.4\\text{ мс}$) | Прямой `.pt` на HF / GDrive |
| **5** | **EyeTAG** (Kinematic Token + CCAF) | GH: `peter8366/EyeTAG` | **2026** | — | **$2.56^\\circ$** (EVE), **$8.08^\\circ$** (Gaze360 Front) | EVE ($12.3\\text{M}$) + Gaze360 ($172\\text{k}$) | $37.9\\text{M}$ | **$19.97\\text{ мс}$** | Открытый код + VGGFace2 pkl |
| **6** | **STAGE** (SAM Attention + Gaussian Proc) | GH: `jswati31/stage` | 2024 | — | **$2.58^\\circ$** (EVE), **$4.12^\\circ$** (EYEDIAP) | EVE ($12.3\\text{M}$) + Gaze360 ($172\\text{k}$) | $24.8\\text{M}$ | **$21.50\\text{ мс}$** | GDrive чекпоинты `.pth` |
| **7** | **MediaPipe Iris + FaceMesh** | Google Research / GH | 2020 | **$3.64^\\circ \\pm 0.75^\\circ$** (Med: **$3.51^\\circ$**) | $3.80^\\circ$ (Geometry) | $50\\text{k}$ лиц + $10\\text{k}$ радужек | **$4.0\\text{M}$** | **$15.30\\text{ мс}$** | Встроенный TFLite бинарник |
| **8** | **YOLO26n (4D Head Comp)** | Локально (Run 014) | 2026 | **$4.00^\\circ \\pm 1.33^\\circ$** (Med: **$3.44^\\circ$**) | — | $158\\,084$ (RT-GENE, MPII, Columbia) | **$1.8\\text{M}$** | **$19.19\\text{ мс}$** (GPU: $4.2\\text{ мс}$) | Локальный `.pt` чекпоинт |
| **9** | **ptgaze MPIIFaceGaze** (ResNet-18) | HF: `hysts` | 2026 | **$5.19^\\circ \\pm 1.43^\\circ$** (Med: **$4.38^\\circ$**) | **$4.80^\\circ$** (MPIIFaceGaze) | MPIIFaceGaze ($45\\text{k}$) | $11.7\\text{M}$ | **$5.58\\text{ мс}$** | Safetensors на HF ($46.8\\text{ МБ}$) |
| **10** | **Idiap ResNet-18 Gaze360** | HF: `Idiap` | 2026 | **$5.75^\\circ \\pm 0.03^\\circ$** (Med: **$5.75^\\circ$**) | $9.02^\\circ$ (Gaze360 Frontal) | Gaze360 ($172\\text{k}$) | $11.7\\text{M}$ | **$17.01\\text{ мс}$** (GPU: $2.0\\text{ мс}$) | Прямой `.pt` на HF ($44.7\\text{ МБ}$) |
| **11** | **YOLO26n (2D Gaze Only)** | Локально (Run 014) | 2026 | **$5.89^\\circ \\pm 2.60^\\circ$** (Med: **$4.90^\\circ$**) | — | $158\\,084$ (RT-GENE, MPII, Columbia) | **$1.8\\text{M}$** | **$19.28\\text{ мс}$** | Локальный `.pt` чекпоинт |
| **12** | **EMC-Gaze** (EGNN Landmark Graph) | arXiv:2603.12388 | 2026 | — | **$5.79^\\circ$** (Dynamic Head) | Граф MediaPipe + сессионный калибратор | **$1.8\\text{M}$** | **$14.00\\text{ мс}$** | Открытый код |
| **13** | **L2CS-Net (Diversity)** | HF: `akgulburak` | 2026 | **$6.59^\\circ \\pm 0.95^\\circ$** (Med: **$6.25^\\circ$**) | $3.92^\\circ$ (MPII), $9.92^\\circ$ (Front360) | Gaze360 ($172\\text{k}$) + ETH-XGaze | $25.6\\text{M}$ | **$13.74\\text{ мс}$** | Прямой `.pkl` на HF |
| **14** | **L2CS-Net (PyFeat / LiteRT)** | HF: `py-feat`, `litert` | 2026/24 | **$6.71^\\circ \\pm 1.35^\\circ$** (Med: **$5.96^\\circ$**) | $3.92^\\circ$ (MPII) | Gaze360 ($172\\text{k}$) + MPII ($45\\text{k}$) | $25.6\\text{M}$ | **$13.79\\text{ мс}$** | TFLite / Safetensors |
| **15** | **yakhyo/gaze-estimation (ResNet-50)** | GH Releases: `yakhyo` | 2024 | **$7.06^\\circ \\pm 1.21^\\circ$** (Med: **$6.95^\\circ$**) | **$4.38^\\circ$** (MPIIFaceGaze) | MPIIFaceGaze ($45\\text{k}$) + Gaze360 | $25.6\\text{M}$ | **$13.93\\text{ мс}$** | Прямой `.pt` в GH Releases |
| **16** | **yakhyo/gaze-estimation (ResNet-18)** | GH Releases: `yakhyo` | 2024 | **$7.11^\\circ \\pm 1.75^\\circ$** (Med: **$5.97^\\circ$**) | **$4.72^\\circ$** (MPIIFaceGaze) | MPIIFaceGaze ($45\\text{k}$) + Gaze360 | $11.7\\text{M}$ | **$6.85\\text{ мс}$** | Прямой `.pt` в GH Releases |
| **17** | **Qualcomm EyeGaze / MobileOne** | HF: `qualcomm` / GH | 2025/24 | **$7.50^\\circ \\pm 2.03^\\circ$** (Med: **$6.64^\\circ$**) | **$4.35^\\circ$** (MPIIFaceGaze) | MPIIFaceGaze ($45\\text{k}$) + GazeCapture | **$2.58\\text{M}$** | **$14.69\\text{ мс}$** (NPU: $0.59\\text{ мс}$) | ONNX / TFLite / QNN DLC |
| **18** | **IVGaze (In-Vehicle Transformer)** | GH: `yihuacheng/IVGaze`| 2024 | — | **$7.04^\\circ$** (In-Cabin Wide Angle) | IVGaze ($115\\text{k}$) + EVE | $19.2\\text{M}$ | **$16.20\\text{ мс}$** | GDrive `.pth` чекпоинт |
| **19** | **Gaze-LLE / Gazelle (DINOv2 ViT-B)** | GH Releases: `fkryan` | 2025 | — | $0.11$ AUC / $L_2$ PoG | EVE + ChildPlay + GazeFollow ($250\\text{k}$) | $86.5\\text{M}$ | $42.00\\text{ мс}$ (GPU: $18\\text{ мс}$) | Прямой `.pt` в GH Releases |
| **20** | **Gazelle-DINOv3 (HGNetV2 ONNX)** | GH Releases: `PINTO0309` | 2025 | — | Эквивалент Gaze-LLE | Дистилляция DINOv2 | **$4.2\\text{M}$** | **$6.80\\text{ мс}$** | Прямой `.onnx` в GH Releases |
| **21** | **MCGaze (Spatial-Temporal R-50)** | GH: `zgchen33/MCGaze` | 2024 | — | **$9.82^\\circ$** (L2CS-setting), $10.74^\\circ$ | Gaze360 ($172\\text{k}$) | $25.6\\text{M}$ | **$18.50\\text{ мс}$** | GDrive `.pth` чекпоинты |
| **22** | **CrossGaze** (Multimodal Attention) | HF: `akgulburak` | 2024 | — | **$9.94^\\circ$** (Gaze360) | Gaze360 ($172\\text{k}$) | $22.3\\text{M}$ | **$19.00\\text{ мс}$** | Прямой `.pth` на HF |
| **23** | **GazeShift** (VR Domain Adaptation) | GH: `gazeshift3/gazeshift`| 2026 | — | $11.2^\\circ \\to 6.4^\\circ$ (Unsupervised) | VRGaze + MPIIGaze | $14.2\\text{M}$ | **$15.00\\text{ мс}$** | GDrive `.pth` + HF Dataset |
| **24** | **GLC** (SlowFast + Transformer) | GH: `BolinLai/GLC` | 2024 | — | Egocentric Gaze PoG | EGTEA + Ego4D ($180\\text{k}$) | $32.4\\text{M}$ | $38.00\\text{ мс}$ | GDrive `.pth` чекпоинты |
| **25** | **OpenFace 3.0** (Multitask ViT) | GH: `CMU-MultiComp-Lab` | 2025 | — | Многозадачный прогноз лица/взгляда | Multiface + In-the-wild ($120\\text{k}$) | $28.5\\text{M}$ | $22.00\\text{ мс}$ | HF / GDrive чекпоинты |
| **26** | **Ghost-CBAM MobileNetV4** | HF: `Jason-42195` | 2026 | — | Классификация зон экрана | ASD Eye-Tracking ($18\\text{k}$) | **$3.8\\text{M}$** | **$5.00\\text{ мс}$** | Прямой `.pth` на HF ($15.2\\text{ МБ}$) |
| **27** | **MAGE** (6-DoF Multi-Decoder) | IEEE 2025 | 2025 | — | 6-DoF ($\\mathbf{g}, \\mathbf{o}, \\mathbf{s}$) | MPIIFaceGaze ($45\\text{k}$) + EYEDIAP | $15.2\\text{M}$ | **$16.50\\text{ мс}$** | Открытый код |
| **28** | **ViT-Base Mobile Gaze** | HF: `wahidww` | 2024 | — | Точность зон $88.4\\%$ | Mobile Eye Tracking v2 ($28\\text{k}$) | $86.0\\text{M}$ | $45.00\\text{ мс}$ | Safetensors на HF ($343\\text{ МБ}$) |
| **29** | **FAZE** (Few-Shot Meta MAML) | ICCV 2019 | 2019 | — | **$3.18^\\circ$** (3-shot adaptation) | GazeCapture ($2\\,445\\,504$) | $12.4\\text{M}$ | **$11.50\\text{ мс}$** | Открытый код + чекпоинты |
| **30** | **iTracker** (4-Stream CNN) | CVPR 2016 | 2016 | — | $1.71\\text{ см}$ экранная ошибка | GazeCapture ($2\\,445\\,504$) | $25.0\\text{M}$ | **$15.00\\text{ мс}$** | Открытый код Caffe / PyTorch |"""

    text09_updated = text09[:start09] + correct_table_09 + text09[end09:]
    with open('docs/09_ready_models_and_aggregators.md', 'w', encoding='utf-8') as f:
        f.write(text09_updated)
    print("Cleanly updated docs/09.")
else:
    print("Could not find table bounds in docs/09")

# 2. Update docs/11
with open('docs/11_detailed_model_architectures.md', 'r', encoding='utf-8') as f:
    text11 = f.read()

header11 = "| Ранг | Модель | Год | Метрика $\\mathcal{E}_{ang}$"
start11 = text11.find(header11)

if start11 != -1:
    correct_table_11 = """| Ранг | Модель | Год | Метрика $\\mathcal{E}_{ang}$ (Бенчмарк / Лит.) | Размер обучающей выборки | Число стадий ($N$) | Бэкбон | Задержка CPU/GPU ($T_{inf}$) | Параметры | Ссылка на веса |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **PureGaze** | 2026/22 | **$1.85^\\circ$** / $3.88^\\circ$ | $1\\,128\\,492$ (ETH-XGaze, MPII) | 1 (инфер.) / 2 (обуч.) | ResNet-50 + ResGazeEs | **$13.27\\text{ мс}$** | $25.6\\text{M}$ | HF `akgulburak` (`.pt`) |
| **2** | **UniGaze-Base** | 2025 | **$1.97^\\circ$** / **$3.96^\\circ$** | **$1\\,650\\,000$** (264k персон) | 2 | ViT-Base (12 слоев) | **$11.75\\text{ мс}$** (GPU) | $86.0\\text{M}$ | HF `UniGaze` (`safetensors`) |
| **3** | **ptgaze ETH-XGaze** | 2026 | **$2.21^\\circ$** / **$4.20^\\circ$** | $1\\,083\\,492$ (ETH-XGaze) | 1 | ResNet-18 (`safetensors`) | **$6.51\\text{ мс}$** (наименьшая) | $11.7\\text{M}$ | HF `hysts` (`safetensors`) |
| **4** | **GazeTR-Hybrid** | 2026/22 | **$2.26^\\circ$** / $3.69^\\circ$ | $1\\,128\\,492$ (ETH-XGaze, MPII) | 1 | ResNet-18 + 6L Transformer | **$14.75\\text{ мс}$** | $18.4\\text{M}$ | HF `akgulburak` / GDrive |
| **5** | **EyeTAG** | **2026** | — / **$2.56^\\circ$** (EVE) | EVE ($12.3\\text{M}$) + Gaze360 | 3 | ResNet-50 + ResNet-18 + CCAF | **$19.97\\text{ мс}$** | $37.9\\text{M}$ | GH `peter8366/EyeTAG` |
| **6** | **STAGE** | 2024 | — / **$2.58^\\circ$** (EVE) | EVE ($12.3\\text{M}$) + Gaze360 | 2 | SAM ViT + Gaussian Processes | **$21.50\\text{ мс}$** | $24.8\\text{M}$ | GDrive `.pth` чекпоинты |
| **7** | **MediaPipe Iris** | 2020 | **$3.64^\\circ$** / $3.80^\\circ$ | $60\\,000$ (FaceMesh + Iris) | 3 | BlazeFace + MobileNet + Iris U-Net | **$15.30\\text{ мс}$** | **$4.0\\text{M}$** | Встроенный TFLite бинарник |
| **8** | **YOLO26n (4D Head)** | 2026 | **$4.00^\\circ$** / — | $158\\,084$ (RT-GENE, MPII, Columbia) | 2 | YOLO26n CSPDarknet + FPN | **$19.19\\text{ мс}$** (GPU: $4.2\\text{ мс}$) | **$1.8\\text{M}$** | Локальный весовой файл `.pt` |
| **9** | **ptgaze MPIIFaceGaze**| 2026 | **$5.19^\\circ$** / **$4.80^\\circ$** | $45\\,000$ (MPIIFaceGaze) | 1 | ResNet-18 (`safetensors`) | **$5.58\\text{ мс}$** | $11.7\\text{M}$ | HF `hysts` (`safetensors`) |
| **10** | **Idiap ResNet-18** | 2026 | **$5.75^\\circ$** / $9.02^\\circ$ | $172\\,000$ (Gaze360) | 1 | ResNet-18 + Quantile Head | **$17.01\\text{ мс}$** (GPU: $2.0\\text{ мс}$) | $11.7\\text{M}$ | HF `Idiap` (`.pt`) |
| **11** | **YOLO26n (2D Gaze)**| 2026 | **$5.89^\\circ$** / — | $158\\,084$ (RT-GENE, MPII, Columbia) | 1 | YOLO26n CSPDarknet | **$19.28\\text{ мс}$** | **$1.8\\text{M}$** | Локальный весовой файл `.pt` |
| **12** | **EMC-Gaze** | 2026 | — / **$5.79^\\circ$** | 3D Граф MediaPipe + Калибратор | 3 | EGNN Equivariant Graph | **$14.00\\text{ мс}$** | **$1.8\\text{M}$** | Открытый код |
| **13** | **L2CS-Net (Diversity)**| 2026 | **$6.59^\\circ$** / $9.92^\\circ$ | $217\\,000$ (Gaze360, ETH-XGaze) | 1 | ResNet-50 (90 бинов) | **$13.74\\text{ мс}$** | $25.6\\text{M}$ | HF `akgulburak` (`.pkl`) |
| **14** | **L2CS-Net (PyFeat)** | 2026/24 | **$6.71^\\circ$** / $3.92^\\circ$ | $217\\,000$ (Gaze360, MPII) | 1 | ResNet-50 (90 бинов) | **$13.79\\text{ мс}$** | $25.6\\text{M}$ | HF `py-feat` / `litert` |
| **15** | **yakhyo (ResNet-50)** | 2024 | **$7.06^\\circ$** / **$4.38^\\circ$** | $45\\,000$ (MPII) + Gaze360 | 1 | ResNet-50 | **$13.93\\text{ мс}$** | $25.6\\text{M}$ | GH Releases `yakhyo` (`.pt`) |
| **16** | **yakhyo (ResNet-18)** | 2024 | **$7.11^\\circ$** / **$4.72^\\circ$** | $45\\,000$ (MPII) + Gaze360 | 1 | ResNet-18 | **$6.85\\text{ мс}$** | $11.7\\text{M}$ | GH Releases `yakhyo` (`.pt`) |
| **17** | **Qualcomm EyeGaze** | 2025 | **$7.50^\\circ$** / **$4.35^\\circ$** | $45\\,000$ (MPII) + GazeCapture | 1 | MobileOne Reparameterized | **$14.69\\text{ мс}$** (NPU: $0.59\\text{ мс}$) | **$2.58\\text{M}$** | HF `qualcomm` (ONNX/QNN) |
| **18** | **IVGaze** | 2024 | — / **$7.04^\\circ$** | $115\\,000$ (IVGaze) + EVE | 1 | CNN-Transformer In-Cabin | **$16.20\\text{ мс}$** | $19.2\\text{M}$ | GDrive `.pth` |
| **19** | **Gaze-LLE (DINOv2)** | 2025 | — / $0.11$ AUC PoG | $250\\,000$ (EVE, ChildPlay) | 2 | DINOv2 ViT-B / HGNetV2 ONNX | **$6.80 - 42.00\\text{ мс}$** | $4.2\\text{M} - 86.5\\text{M}$ | GH Releases `fkryan`/`PINTO` |
| **20** | **Gazelle-DINOv3 (HGNetV2)**| 2025 | — / Эквивалент Gaze-LLE | Дистилляция DINOv2 | 1 | HGNetV2-Nano ONNX | **$6.80\\text{ мс}$** | **$4.2\\text{M}$** | GH Releases `PINTO0309` |
| **21** | **MCGaze** | 2024 | — / **$9.82^\\circ$** (L2CS) | $172\\,000$ (Gaze360) | 2 | SCConv ResNet-50 + GRU | **$18.50\\text{ мс}$** | $25.6\\text{M}$ | GDrive `.pth` |
| **22** | **CrossGaze** | 2024 | — / **$9.94^\\circ$** | $172\\,000$ (Gaze360) | 1 | ResNet-18 + Cross-Attention | **$19.00\\text{ мс}$** | $22.3\\text{M}$ | HF `akgulburak` (`.pth`) |
| **23** | **GazeShift** | 2026 | — / $11.2^\\circ \\to 6.4^\\circ$ | VRGaze + MPIIGaze | 2 | ResNet-18 + Domain Classifier | **$15.00\\text{ мс}$** | $14.2\\text{M}$ | GDrive `.pth` + HF Dataset |
| **24** | **GLC** | 2024 | — / PoG Ego | $195\\,000$ (EGTEA, Ego4D) | 2 | SlowFast + Transformer | $38.00\\text{ мс}$ | $32.4\\text{M}$ | GDrive `.pth` |
| **25** | **GazeCLR** | 2024 | — / Contrastive | $12.3\\text{M}$ (EVE, Gaze360) | 1 | ResNet-18/50 Contrastive | **$12.50\\text{ мс}$** | $11.7\\text{M}$ | GDrive `.pth` |
| **26** | **OpenFace 3.0** | 2025 | — / Multitask | $120\\,000$ (Multiface) | 2 | ViT-Tiny Multitask Stem | **$22.00\\text{ мс}$** | $28.5\\text{M}$ | HF / GDrive |
| **27** | **Ghost-CBAM Mobile**| 2026 | — / Фиксация | $18\\,000$ (ASD Eye-Tracking) | 1 | MobileNetV4-Small + CBAM | **$5.00\\text{ мс}$** | **$3.8\\text{M}$** | HF `Jason-42195` (`.pth`) |
| **28** | **MAGE** | 2025 | — / 6-DoF | $45\\,000$ (MPII) + EYEDIAP | 4 | ResNet-18 Multi-Decoder | **$16.50\\text{ мс}$** | $15.2\\text{M}$ | Открытый код |
| **29** | **ViT-Base Mobile** | 2024 | — / $88.4\\%$ | $28\\,000$ (Mobile Eye v2) | 1 | ViT-Base (12 слоев) | $45.00\\text{ мс}$ | $86.0\\text{M}$ | HF `wahidww` (`safetensors`)|
| **30** | **FAZE** | 2019 | — / **$3.18^\\circ$** (3-shot)| $2\\,445\\,504$ (GazeCapture) | 3 | DT-ED ResNet + MAML | **$11.50\\text{ мс}$** | $12.4\\text{M}$ | GitHub `NVlabs/FAZE` |
| **31** | **iTracker** | 2016 | — / $1.71\\text{ см}$ | $2\\,445\\,504$ (GazeCapture) | 1 | 4-Stream Custom CNN | **$15.00\\text{ мс}$** | $25.0\\text{M}$ | GitHub Caffe/PyTorch |"""

    text11_updated = text11[:start11] + correct_table_11 + "\n"
    with open('docs/11_detailed_model_architectures.md', 'w', encoding='utf-8') as f:
        f.write(text11_updated)
    print("Cleanly updated docs/11.")
