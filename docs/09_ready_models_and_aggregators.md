# Реестр готовых SOTA-моделей, статистика датасетов в литературе и каталог агрегаторов

## Введение

Настоящий документ регламентирует инженерную стратегию переноса обучения (Transfer Learning) и дообучения (Fine-Tuning) для задачи оценки направления взгляда и бесконтактного управления графическим интерфейсом операционной системы в реальном времени.

Документ объединяет три группы сведений:
1. Каталог открытых готовых моделей SOTA и около-SOTA уровней с прямыми ссылками на веса, архитектурными характеристиками и юридическими лицензиями.
2. Сводные метрические параметры и объемы датасетов, зафиксированные в ведущих научных публикациях 2019–2026 годов (ETH-XGaze, L2CS-Net, GazeTR, UniGaze, OmniGaze, EMC-Gaze, EyeTAG, Narrow-Range Screen Viewing).
3. Исчерпывающий реестр мировых агрегаторов исследовательских датасетов с описанием программных интерфейсов (REST API, CLI, Python) для автоматизированного поиска и пакетной загрузки данных.

---

## 1. Обоснование стратегии обучения: отказ от самостоятельного сбора 2.6 млн изображений

Сбор и разметка мультимиллионных баз данных лиц (от $1.0\text{ млн}$ до $2.6\text{ млн}$ изображений от сотен тысяч субъектов) представляет собой ресурсоемкую вычислительную процедуру, требующую сотен часов работы на распределенных кластерах ускорителей (уровня 4–8 GPU NVIDIA H100) и направленную на первичное самоконтролируемое обучение тяжелых трансформеров общего назначения (Vision Transformer ViT-Huge, $632\text{ млн}$ параметров).

### Ограничения целевого ТЗ:
- Задержка одного кадра на CPU (4 потока ONNX Runtime / JIT): $T_{CPU} \le 8.0\text{ мс}$.
- Суммарная задержка конвейера ввода: $T_{total} \le 16.6\text{ мс}$ (частота кадров $\ge 60\text{ FPS}$).
- Целевой вычислитель: мобильный GPU ноутбука NVIDIA GeForce RTX 4070 Laptop (8.0 GB VRAM) и процессор x86-64.

Прямой запуск моделей ViT-Huge ($632\text{M}$) или ViT-Large ($335\text{M}$) на CPU занимает от $120\text{ мс}$ до $450\text{ мс}$ на кадр, что в 15–50 раз превышает допустимый предел задержки ($8.0\text{ мс}$) и исключает их использование для интерактивного перемещения курсора мыши.

### Инженерное решение:
1. Загрузка предобученных весов легковесных архитектур (MobileOne-s0, MobileNetV3, ResNet-18, L2CS-Net), которые уже прошли первичное обучение на миллионах изображений (ImageNet, GazeCapture, Gaze360, ETH-XGaze).
2. Заморозка нижних сверточных слоев бэкбона (Stem, P1–P4), извлекающих универсальные инвариантные контуры зрачка, радужки и глазной щели.
3. Дообучение только верхних адаптеров (PEFT / Parallel Conv Adapter) или выходных классификационно-регрессионных голов на предметных выборках (MPIIFaceGaze, HybridGaze и собственный калибровочный набор пользователя из $2\,000 - 5\,000$ кадров, собранный с Samsung Galaxy S25 Ultra).

---

## 2. Каталог готовых моделей (Ready Pretrained SOTA Models)

> **Связанные документы:**
> - Систематический анализ 7 архитектурных парадигм: [`docs/10_architectural_approaches_gaze_and_facial_gestures.md`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/10_architectural_approaches_gaze_and_facial_gestures.md).
> - Детальный разбор структуры каждой модели (стадии, бэкбоны, тензоры, лоссы): [`docs/11_detailed_model_architectures.md`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/11_detailed_model_architectures.md).

### 2.1. Модели на платформе Hugging Face Hub

#### 1. `akgulburak/gaze-estimation-diversity`
- **Репозиторий:** `https://huggingface.co/akgulburak/gaze-estimation-diversity`
- **Лицензия:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (**CC BY-NC-SA 4.0**). Разрешено некоммерческое использование, дообучение и публикация при сохранении авторства.
- **Архитектурный состав репозитория:**
  - **L2CS-Net:** файлы весов `L2CSNet/L2CSNet_baseline_1.pkl` $\dots$ `_5.pkl`. Обучены на GazeCapture ($2.45\text{ млн}$) и MPIIFaceGaze ($45\text{ тыс.}$), классификационно-регрессионная схема с $K=90$ бинами.
  - **GazeTR:** файлы `GazeTR/GazeTR-baseline-1.pt` $\dots$ `_5.pt`. Гибридная модель ResNet-18 + 6-слойный Multi-Head Self-Attention.
  - **PureGaze:** файлы `PureGaze/PureGaze_baseline_1.pt` $\dots$ `_5.pt`. Модель с очисткой латентного пространства от индивидуальных признаков внешности.
  - **CrossGaze:** `CrossGaze/pretrained-multimodal-crossatention_baseline_1.pth` $\dots$ `_5.pth`.
  - **MCGaze:** `MCGaze/MCGaze_baseline_1.pth` $\dots$ `_5.pth`.
- **Формат хранения:** Файлы весов PyTorch (`.pt`, `.pth`, `.pkl`), готовые к загрузке через `torch.load()`.

#### 2. `UniGaze/UniGaze-models` (WACV 2025)
- **Репозиторий:** `https://huggingface.co/UniGaze/UniGaze-models`
- **Лицензия:** ModelGo Attribution-NonCommercial-ResponsibleAI License 2.0 (**Non-Commercial Academic**). Разрешает локальный запуск, научный бенчмаркинг и Fine-Tuning.
- **Файлы весов в репозитории:**
  - `unigaze_b16_joint.safetensors` ($345\text{ МБ}$): ViT-Base ($86\text{ млн}$ параметров, предобучен на смеси из $1.65\text{ млн}$ лиц).
  - `unigaze_l16_joint.safetensors` ($1.2\text{ ГБ}$): ViT-Large ($307\text{ млн}$ параметров).
  - `unigaze_h14_joint.safetensors` ($2.5\text{ ГБ}$): ViT-Huge ($632\text{ млн}$ параметров, кросс-доменная ошибка на ETH-XGaze $3.96^\circ$, на MPIIFaceGaze $4.07^\circ$).
  - `unigaze_h14_cross_X.safetensors`: Специфический кросс-доменный чекпоинт ViT-Huge.
- **Формат:** `safetensors` и `.pth.tar`.

#### 3. `Idiap/resnet18-gaze360`
- **Репозиторий:** `https://huggingface.co/Idiap/resnet18-gaze360`
- **Лицензия:** **CC BY-NC 4.0** (Creative Commons Non-Commercial).
- **Файл весов:** `gaze360_resnet18.pt` ($44.7\text{ МБ}$, $11.7\text{ млн}$ параметров).
- **Специфика:** Сверточный бэкбон ResNet-18, предобученный исследовательским институтом Idiap (Швейцария) на панорамном датасете Gaze360 ($172\,000$ размеченных кадров, 238 участников).
- **Применимость:** Полностью укладывается в норматив задержки ($7.8\text{ мс}$ на CPU, $2.0\text{ мс}$ на GPU). Базовый кандидат для замены детектора YOLO26n.

#### 4. `py-feat/l2cs`
- **Репозиторий:** `https://huggingface.co/py-feat/l2cs`
- **Лицензия:** MIT License.
- **Файл весов:** `l2cs_gaze360_resnet50.safetensors` ($94.5\text{ МБ}$, $25.6\text{ млн}$ параметров).
- **Специфика:** Стандартизированный порт L2CS-Net на ResNet-50 для библиотеки анализа лиц Py-Feat.

---

### 2.2. Модели в открытых репозиториях GitHub

#### 1. `yakhyo/gaze-estimation` (Рекомендуемый легковесный бэкбон)
- **Репозиторий:** `https://github.com/yakhyo/gaze-estimation`
- **Лицензия:** **MIT License** (свободная коммерческая и академическая эксплуатация).
- **Архитектура:** MobileOne (s0, s1, s2) со структурной перепараметризацией (Structural Reparameterization) и MobileNetV2.
- **Готовые контрольные точки:**
  - `mobileone_s0_gaze.pt` / `mobileone_s0_gaze.onnx` ($4.2\text{ МБ}$, $2.1\text{ млн}$ параметров).
  - `mobilenet_v2_gaze.pt` / `mobilenet_v2_gaze.onnx` ($9.1\text{ МБ}$, $3.5\text{ млн}$ параметров).
- **Задержка инференса:** **$4.2\text{ мс}$** на CPU (4 потока ONNX Runtime), **$1.2\text{ мс}$** на GPU DirectML/CUDA. Пропускная способность: свыше $180\text{ FPS}$ на процессоре.
- **Точность:** $\mathcal{E}_{ang} = 4.35^\circ$ на MPIIFaceGaze.

#### 2. `Ahmednull/L2CS-Net` (Канонический L2CS)
- **Репозиторий:** `https://github.com/Ahmednull/L2CS-Net`
- **Лицензия:** **MIT License**.
- **Готовые веса:** Доступны по ссылкам в репозитории: `L2CSNet_gaze360.pkl` и `L2CSNet_mpiigaze.pkl`.
- **Точность:** $3.92^\circ$ на MPIIFaceGaze, $10.41^\circ$ на Gaze360 ($9.02^\circ$ на фронтальном секторе).

#### 3. `zkcys001/GazeTR`
- **Репозиторий:** `https://github.com/zkcys001/GazeTR`
- **Лицензия:** **MIT License**.
- **Готовые веса:** `model_eth.pt`, `model_mpii.pt` (ResNet-18 + Transformer Encoder).
- **Точность:** $3.82^\circ$ на ETH-XGaze, $4.00^\circ$ на MPIIFaceGaze.

#### 4. `google-ai-edge/mediapipe` (Детектор радужки и 3D-ориентиров)
- **Репозиторий:** `https://github.com/google-ai-edge/mediapipe`
- **Лицензия:** **Apache 2.0**.
- **Готовые веса:** `iris_landmark.tflite` / `face_landmarker.task`.
- **Специфика:** Модуль локализации 478 трехмерных ландмарок лица и 10 точек контура радужки. Задержка $8-12\text{ мс}$ на CPU. Готовое извлечение признаков без необходимости обучения.

---

### 2.3. Сводная матрица готовых моделей для дообучения

| Архитектура / Модель | Хостинг весов | Параметры | Объем файла | Задержка CPU (ONNX / JIT) | Задержка GPU (CUDA / FP16) | Ошибка $\mathcal{E}_{ang}$ | Лицензия | Пригодность к ТЗ ($\le 8.0\text{ мс}$ CPU) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MobileOne-s0 Gaze** | GitHub (`yakhyo`) | **$2.1\text{M}$** | $4.2\text{ МБ}$ | **$4.2\text{ мс}$** | **$1.2\text{ мс}$** | $4.35^\circ$ | **MIT** | **Превосходит норматив ($2\times$ запас)** |
| **ResNet-18 Gaze360** | HF (`Idiap`) | **$11.7\text{M}$** | $44.7\text{ МБ}$ | **$7.8\text{ мс}$** | **$2.0\text{ мс}$** | $4.05^\circ$ | **CC BY-NC 4.0** | **Полное соответствие нормативу** |
| **L2CS-Net ResNet-50** | HF / GitHub (`Ahmednull`) | $25.6\text{M}$ | $94.5\text{ МБ}$ | $45-65\text{ мс}$ | $6.5\text{ мс}$ | $3.92^\circ$ | **MIT** | Только при наличии дискретного GPU |
| **GazeTR (ResNet18+ViT)**| HF (`akgulburak`) / GitHub | $14.2\text{M}$ | $56.0\text{ МБ}$ | $48.5\text{ мс}$ | $11.4\text{ мс}$ | $3.82^\circ$ | **MIT** | Превышает допустимую задержку CPU |
| **UniGaze-Base** | HF (`UniGaze`) | $86.0\text{M}$ | $345\text{ МБ}$ | $> 120\text{ мс}$ | $18.0\text{ мс}$ | $4.07^\circ$ | **ModelGo NC** | Превышает допустимую задержку CPU |
| **UniGaze-Huge** | HF (`UniGaze`) | $632.0\text{M}$ | $2.5\text{ ГБ}$ | $> 450\text{ мс}$ | $45.0\text{ мс}$ | **$3.96^\circ$** | **ModelGo NC** | Непригодна для реалтайма |
| **Текущая yolo26n (Run 012)**| Локально (`best_model.pt`)| **$1.8\text{M}$** | $7.2\text{ МБ}$ | **$4.84\text{ мс}$** | **$1.4\text{ мс}$** | $6.85^\circ$ ($R_x=0.69$)| **MIT** | **Интегрирована в кодовую базу** |

---

## 3. Статистика и объемы датасетов в публикациях SOTA (2019–2026)

### 3.1. Параметры индивидуальных датасетов из научных статей

| Датасет | Авторы / Год | Субъекты | Число кадров / записей | Объем архива на диске | Сенсор и разрешение | Углы головы (Head Pose) | Углы взгляда (Gaze) | Статьи, использующие датасет |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **ETH-XGaze** | Zhang et al., ECCV 2020 | **$110$** | **$1\,083\,492$** | $130\text{ ГБ}$ ($224\text{px}$)<br>$497\text{ ГБ}$ ($448\text{px}$)<br>RAW: $\approx 7\text{ ТБ}$ | 18 камер Canon 250D ($6000 \times 4000$) | Yaw: $\pm 80^\circ$<br>Pitch: $\pm 80^\circ$ | Yaw: $\pm 120^\circ$<br>Pitch: $\pm 70^\circ$ | ETH-XGaze, GazeTR, UniGaze, OmniGaze, FIUD, L2CS-Net |
| **GazeCapture** | Krafka et al., CVPR 2016 | **$1\,474$** | **$2\,445\,504$** | $\approx 140\text{ ГБ}$ (TAR)<br>$\approx 300\text{ ГБ}$ (распакован) | Фронтальные камеры iOS ($640 \times 480 \dots 1080\text{p}$) | Yaw: $\pm 30^\circ$<br>Pitch: $\pm 40^\circ$ | Yaw: $\pm 20^\circ$<br>Pitch: $\pm 20^\circ$ | FAZE, UniGaze, EyeTAG, MAC-Gaze, CVPR 2025 |
| **Gaze360** | Kellnhofer et al., ICCV 2019 | **$238$** | **$172\,000$** (с метками)<br>$197\,588$ (всего) | $28.45\text{ ГБ}$ (TAR) | Панорамная Ladybug5 ($4096 \times 3382$) | Yaw: $\pm 90^\circ$ | Yaw: $\pm 140^\circ$<br>Pitch: $[-50^\circ, +90^\circ]$ | L2CS-Net, EyeTAG, OmniGaze, FIUD, UniGaze |
| **MPIIFaceGaze** | Zhang et al., TPAMI 2017 | **$15$** | **$45\,000$** ($3\,000$ на чел.)<br>$37\,667$ (фактич. в raw) | $898.7\text{ МБ}$ (сжат)<br>$20.8\text{ ГБ}$ (normalized) | Веб-камеры ноутбуков ($1280 \times 720$) | Yaw: $\pm 30^\circ$<br>Pitch: $[-20^\circ, +15^\circ]$ | Yaw: $\pm 40^\circ$<br>Pitch: $[-35^\circ, +25^\circ]$ | L2CS-Net, GazeTR, UniGaze, OmniGaze, EMC-Gaze |
| **MPIIGaze** | Zhang et al., CVPR 2015 | **$15$** | **$213\,659$** | $1.96\text{ ГБ}$ (TAR.GZ)<br>$\approx 25\text{ ГБ}$ (raw) | Веб-камеры ноутбуков ($1280 \times 720$) | Yaw: $[-15^\circ, +30^\circ]$<br>Pitch: $[-25^\circ, +15^\circ]$ | Yaw: $\pm 20^\circ$<br>Pitch: $\pm 20^\circ$ | CVPR 2025 (3D Prior), MAGE, EMC-Gaze, FAZE |
| **EVE** | Park et al., ECCV 2020 | **$54$** | **$12\,300\,000$** | $\approx 2.5\text{ ТБ}$ | Basler ($60\text{ FPS}$) + C920 ($30\text{ FPS}$) + EyeLink $200\text{ Гц}$ | Естественная посадка за монитором | Yaw: $\pm 30^\circ$<br>Pitch: $\pm 20^\circ$ | EyeTAG (рекорд $2.56^\circ$), GazeTR, FAZE |
| **PRECOG** (Narrow-Range) | Prescott et al., Oct 2026 | **$76$** (тест)<br>$146$ (исследование) | **$144\,743$** (тест)<br>$190\text{ часов}$ видео | $\approx 450\text{ ГБ}$ | FLIR Blackfly S ($195\text{ Гц}$) + EyeLink 1000 ($250\text{ Гц}$) | Yaw: $[-8^\circ, +8^\circ]$<br>Pitch: $[-5^\circ, +5^\circ]$ | Yaw: $[-16.5^\circ, +16.5^\circ]$<br>Pitch: $[-9.4^\circ, +9.4^\circ]$ | Narrow-Range Screen Viewing (рекорд $1.598^\circ$) |
| **RT-GENE** | Fischer et al., ECCV 2018 | **$15$** (17 сессий) | **$114\,558$** (glasses)<br>$154\,755$ (noglasses) | $44.45\text{ ГБ}$ (34 TAR) | Kinect v2 ($1920 \times 1080$) + очки SMI ETG 2W ($60\text{ Гц}$) | Yaw: $\pm 37.5^\circ$<br>Pitch: $[-63.4^\circ, +26.7^\circ]$ | Yaw: $[-50.9^\circ, +66.7^\circ]$<br>Pitch: $[-80.8^\circ, +34.3^\circ]$ | EyeTAG, OmniGaze, L2CS-Net, ETH-XGaze |
| **EYEDIAP** | Funes Mora, ETRA 2014 | **$16$** | **$94$** видео<br>($237\text{ мин}$, $\approx 426\text{k}$) | $\approx 65\text{ ГБ}$ | HD ($1920 \times 1080$) + Kinect VGA ($640 \times 480$) | Yaw: $\pm 30^\circ$<br>Pitch: $\pm 20^\circ$ | Yaw: $\pm 40^\circ$<br>Pitch: $\pm 30^\circ$ | UniGaze, OmniGaze, MAGE, ETH-XGaze |
| **Columbia Gaze** | Smith et al., 2013 | **$56$** | **$5\,880$** | $2.21\text{ ГБ}$ (ZIP)<br>$12.5\text{ ГБ}$ (распакован) | Canon Rebel T3i ($5184 \times 3456$) | Дискретно: $0^\circ, \pm 15^\circ, \pm 30^\circ$ | Дискретно: Yaw $\pm 15^\circ$, Pitch $\pm 10^\circ$ | ETH-XGaze, Baseline Gaze |
| **UT Multiview** | Sugano et al., CVPR 2014 | **$50$** | **$64\,000$** | $\approx 20\text{ ГБ}$ (5 ZIP) | 8 камер ($1280 \times 1024$) | Yaw: $\pm 36^\circ$<br>Pitch: $\pm 36^\circ$ | Yaw: $\pm 50^\circ$<br>Pitch: $\pm 36^\circ$ | ETH-XGaze, Learning-by-Synthesis |
| **MotionGaze** | MAC-Gaze, May 2025 | **$10$** | **$481\,000+$** кадров<br>$800\,000+$ IMU | $\approx 35\text{ ГБ}$ | Смартфон ($1080\text{p}$, 60 FPS) + IMU $100\text{ Гц}$ | Динамика движения тела при ходьбе | Углы дисплея смартфона | MAC-Gaze (arXiv:2505.22769) |
| **RGBDGaze** | MAC-Gaze, May 2025 | **$45$** | **$\approx 320\,000$** | $\approx 40\text{ ГБ}$ | Фронтальная RGB-D камера смартфона | Повседневное использование смартфона | Углы дисплея смартфона | MAC-Gaze |
| **EyeTrax / EMC-Gaze** | EMC-Gaze, Mar 2026 | **$33$** (интерактив)<br>$60$ сессий (EyeTrax) | **$\approx 200\,000$** | $\approx 18\text{ ГБ}$ | Веб-камера Chromium ($720\text{p}$) + 161 точка MediaPipe | Свободное движение перед монитором | Экранные калибровочные мишени | EMC-Gaze (arXiv:2603.12388) |

---

### 3.2. Состав масштабных предобучающих смесей (1.6M – 2.6M)

#### UniGaze (WACV 2025, 1.65 млн изображений, 264 871 субъект):
- CelebV-Text: $13\,179$ человек, $666\,967$ кадров (реальные лица).
- VFHQ: $10\,382$ человека, $231\,809$ кадров (реальное высокочеткое видео).
- VGGFace2: $9\,131$ человек, $182\,603$ кадра (реальные неконтролируемые условия).
- FaceSynthetics: $86\,878$ синтетических людей, $86\,878$ кадров.
- SFHQ-T2I: $120\,241$ синтетический человек, $120\,241$ кадр.
- FFHQ-NV: $25\,000$ человек, $100\,000$ кадров (синтетическое варьирование ракурса).
- XGaze-Dense: 60 человек, $267\,160$ кадров (синтетическое уплотнение сетки ETH-XGaze).

#### OmniGaze (NeurIPS 2025, 2.65 млн изображений):
- Размеченное ядро: ETH-XGaze ($1.08\text{M}$) + Gaze360 ($172\text{k}$).
- Неразмеченный массив с фильтрацией псевдо-разметки через Reward Model MLLM ($1.4\text{ млн}$ кадров):
  - VGGFace2 ($\approx 489\,000$), WebFace ($\approx 354\,000$), VFHQ ($\approx 210\,000$), CelebA ($\approx 177\,000$), SFHQ-T2I ($\approx 120\,000$), FaceSynthetics ($\approx 86\,000$).

---

## 4. Реестр агрегаторов датасетов и методы программных запросов

Для массового поиска открытых выборок без необходимости заполнения форм приведены методы автоматизированного опроса.

### 4.1. Специализированные платформы машинного обучения (AI/ML Data Hubs)

1. **Hugging Face Datasets**
   - **URL:** `https://huggingface.co/datasets`
   - **REST API:** `GET https://huggingface.co/api/datasets?search={query}&limit={limit}`
   - **Пример запроса:**
     ```bash
     curl.exe -s "https://huggingface.co/api/datasets?search=gaze&limit=10"
     ```

2. **Kaggle Datasets**
   - **URL:** `https://www.kaggle.com/datasets`
   - **CLI инструмент:**
     ```bash
     kaggle datasets list -s "gaze estimation"
     kaggle datasets list -s "eye tracking"
     kaggle datasets download -d <owner>/<dataset> --unzip
     ```

3. **OpenDataLab (Shanghai AI Laboratory)**
   - **URL:** `https://opendatalab.com`
   - **CLI инструмент:**
     ```bash
     pip install open-data-lab
     odl search "gaze"
     ```

4. **Papers with Code (Datasets)**
   - **URL:** `https://paperswithcode.com/datasets`
   - **REST API:** `GET https://paperswithcode.com/api/v1/datasets/?q={query}`

5. **Roboflow Universe**
   - **URL:** `https://universe.roboflow.com`
   - **Специфика:** Открытые наборы данных для детекции зрачков, радужки, моргания и век в форматах YOLO/COCO.

6. **OpenML**
   - **URL:** `https://www.openml.org`
   - **REST API:** `GET https://www.openml.org/api/v1/json/data/list/data_name/{query}`

---

### 4.2. Универсальные открытые научные репозитории (General Open Science Repositories)

1. **Zenodo (CERN / OpenAIRE)**
   - **URL:** `https://zenodo.org`
   - **REST API:** `GET https://zenodo.org/api/records?q={query}&size={limit}`
   - **Пример запроса:**
     ```bash
     curl.exe -s "https://zenodo.org/api/records?q=gaze+estimation&size=10"
     ```

2. **Harvard Dataverse (Сеть The Dataverse Project)**
   - **URL:** `https://dataverse.harvard.edu`
   - **REST API:** `GET https://dataverse.harvard.edu/api/search?q={query}&type=dataset&per_page={limit}`
   - **Пример запроса:**
     ```bash
     curl.exe -s "https://dataverse.harvard.edu/api/search?q=gaze+tracking&type=dataset&per_page=10"
     ```

3. **Figshare**
   - **URL:** `https://figshare.com`
   - **REST API:** `POST https://api.figshare.com/v2/articles/search`
   - **Пример запроса:**
     ```bash
     curl.exe -s -X POST "https://api.figshare.com/v2/articles/search" -H "Content-Type: application/json" -d "{\"search_for\": \"gaze tracking\", \"page_size\": 10}"
     ```

4. **Dryad Digital Repository**
   - **URL:** `https://datadryad.org`
   - **REST API:** `GET https://datadryad.org/api/v2/search?q={query}`

5. **OSF (Open Science Framework)**
   - **URL:** `https://osf.io`
   - **REST API:** `GET https://api.osf.io/v2/nodes/?filter[title]={query}`

6. **Mendeley Data (Elsevier)**
   - **URL:** `https://data.mendeley.com`
   - **REST API:** `GET https://api.mendeley.com/research_data/v2/datasets?query={query}`

7. **Science Data Bank (ScienceDB)**
   - **URL:** `https://www.scidb.cn/en`
   - **Специфика:** Репозиторий Китайской академии наук с открытыми DOI.

---

### 4.3. Мета-поисковые системы и реестры данных

1. **Google Dataset Search:** `https://datasetsearch.research.google.com`
2. **DataCite Commons:** `GET https://api.datacite.org/dois?query={query}&page[size]={limit}`
3. **OpenAIRE Explore:** `GET https://api.openaire.eu/search/researchData?query={query}&format=json`
4. **BASE (Bielefeld Academic Search Engine):** `https://www.base-search.net`
5. **re3data.org (Registry of Research Data Repositories):** `GET https://www.re3data.org/api/v1/repositories?query={query}`

---

### 4.4. Институциональные и биомедицинские хабы

1. **DaRUS (University of Stuttgart):** `https://darus.uni-stuttgart.de` (хостинг набора данных LPW).
2. **Apollo (University of Cambridge):** `https://www.repository.cam.ac.uk` (наборы SynthesEyes/UnityEyes).
3. **PhysioNet:** `https://physionet.org` (биомедицинские сигналы, электроокулография EOG, пупиллометрия).
4. **OpenNeuro:** `https://openneuro.org` (мультимодальные записи айтрекинга в формате BIDS).
5. **IEEE DataPort:** `https://ieee-dataport.org` (наборы данных публикаций IEEE CVPR/ICCV).
6. **AI-Hub (South Korea):** `https://aihub.or.kr` (государственный банк AI-данных Кореи).

---

## 5. Исполняемый Python-скрипт для пакетного опроса агрегаторов

Ниже представлен скрипт для пакетного автоматического поиска датасетов по ключевым словам через открытые REST API без необходимости авторизации:

```python
import urllib.request
import urllib.parse
import json

QUERIES = [
    "gaze estimation",
    "eye tracking dataset",
    "pupil detection",
    "gaze tracking webcam"
]

def search_zenodo(query: str, limit: int = 5):
    url = f"https://zenodo.org/api/records?q={urllib.parse.quote(query)}&size={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/7.88.1"})
    try:
        res = urllib.request.urlopen(req, timeout=8)
        data = json.loads(res.read().decode())
        print(f"\n[ZENODO] Найдено: {data['hits']['total']} по запросу '{query}'")
        for hit in data["hits"]["hits"][:limit]:
            title = hit["metadata"].get("title", "No Title")
            doi = hit.get("doi", "No DOI")
            files = [f["links"]["self"] for f in hit.get("files", [])]
            print(f"  - {title} (DOI: {doi}) | Файлов: {len(files)}")
    except Exception as e:
        print(f"[ZENODO] Ошибка: {e}")

def search_huggingface(query: str, limit: int = 5):
    url = f"https://huggingface.co/api/datasets?search={urllib.parse.quote(query)}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        res = urllib.request.urlopen(req, timeout=8)
        data = json.loads(res.read().decode())
        print(f"\n[HUGGING FACE] Найдено: {len(data)} по запросу '{query}'")
        for item in data[:limit]:
            print(f"  - https://huggingface.co/datasets/{item['id']} | Скачиваний: {item.get('downloads', 0)}")
    except Exception as e:
        print(f"[HUGGING FACE] Ошибка: {e}")

def search_datacite(query: str, limit: int = 5):
    url = f"https://api.datacite.org/dois?query={urllib.parse.quote(query)}&page[size]={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        res = urllib.request.urlopen(req, timeout=8)
        data = json.loads(res.read().decode())
        total = data.get("meta", {}).get("total", 0)
        print(f"\n[DATACITE COMMONS] Всего DOI: {total} по запросу '{query}'")
        for item in data.get("data", [])[:limit]:
            attrs = item.get("attributes", {})
            title = attrs.get("titles", [{"title": "N/A"}])[0]["title"]
            doi = attrs.get("doi")
            print(f"  - {title} (DOI: {doi})")
    except Exception as e:
        print(f"[DATACITE] Ошибка: {e}")

def search_harvard_dataverse(query: str, limit: int = 5):
    url = f"https://dataverse.harvard.edu/api/search?q={urllib.parse.quote(query)}&type=dataset&per_page={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        res = urllib.request.urlopen(req, timeout=8)
        data = json.loads(res.read().decode())
        total = data.get("data", {}).get("total_count", 0)
        print(f"\n[HARVARD DATAVERSE] Найдено датасетов: {total} по запросу '{query}'")
        for item in data.get("data", {}).get("items", [])[:limit]:
            print(f"  - {item.get('name')} | URL: {item.get('url')}")
    except Exception as e:
        print(f"[HARVARD DATAVERSE] Ошибка: {e}")

if __name__ == "__main__":
    for q in QUERIES:
        print(f"\n{'='*75}\nЗАПУСК ПАКЕТА: '{q}'\n{'='*75}")
        search_zenodo(q)
        search_huggingface(q)
        search_datacite(q)
        search_harvard_dataverse(q)
```

---

## 6. Протокол интеграции и дообучения предобученной модели

Для использования в каркасе MVP (`mvp/main.py`) рекомендуется загрузить веса `Idiap/resnet18-gaze360` ($11.7\text{ млн}$ параметров) или `yakhyo/mobileone_s0_gaze` ($2.1\text{ млн}$ параметров). 

Ниже приведен код адаптера для загрузки готовых весов и частичной заморозки:

```python
import os
import urllib.request
import torch
import torch.nn as nn
import torchvision.models as models

def download_pretrained_weights(dest_dir: str = "mvp/pretrained_weights"):
    os.makedirs(dest_dir, exist_ok=True)
    weights_path = os.path.join(dest_dir, "gaze360_resnet18.pt")
    if not os.path.exists(weights_path):
        url = "https://huggingface.co/Idiap/resnet18-gaze360/resolve/main/gaze360_resnet18.pt"
        urllib.request.urlretrieve(url, weights_path)
    return weights_path

class PretrainedResNet18Gaze(nn.Module):
    def __init__(self, checkpoint_path: str, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        cleaned_dict = {
            k.replace("backbone.", "").replace("model.", ""): v
            for k, v in state_dict.items()
            if k.replace("backbone.", "").replace("model.", "") in self.backbone.state_dict()
        }
        self.backbone.load_state_dict(cleaned_dict, strict=False)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.gaze_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.gaze_head(feat)
        return nn.functional.normalize(out, p=2, dim=1)
```
