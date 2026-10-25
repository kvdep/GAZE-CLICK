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
> - Детальный разбор структуры каждой модели (стадии, бэкбоны, тензоры, лоссы), ранжированный по метрикам: [`docs/11_detailed_model_architectures.md`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/11_detailed_model_architectures.md).

### 2.1. Автоматизированный парсер платформы Hugging Face
Сбор моделей реализован через автономный Python-скрипт [`scripts/parse_huggingface_gaze.py`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/scripts/parse_huggingface_gaze.py), выполняющий прямые HTTP-запросы к REST API `https://huggingface.co/api/models` без использования устаревших внутренних поисковых кэшей. Скрипт опрашивает 25 ключевых поисковых запросов (`gaze-estimation`, `gaze360`, `mpiigaze`, `eth-xgaze`, `l2cs`, `gazetr`, `puregaze`, `unigaze`, `omnigaze`, `eyetag`, `eye-tracking`, `ptgaze`, `crossgaze`, `mcgaze` и др.), отсекает нерелевантные текстовые/аудио репозитории (NLP, ASR, ирландский язык "irish", сводки новостей "gazeta") и сохраняет верифицированный реестр в [`scripts/huggingface_gaze_models.json`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/scripts/huggingface_gaze_models.json).

---

### 2.2. Хронологический реестр моделей Hugging Face (от 2026 г. в прошлое)

#### [2026] Модели текущего года
1. **`akgulburak/gaze-estimation-diversity`** (Август 2026)
   - **Репозиторий:** `https://huggingface.co/akgulburak/gaze-estimation-diversity`
   - **Лицензия:** CC BY-NC-SA 4.0.
   - **Форматы весов:** PyTorch (`.pt`, `.pth`, `.pkl`).
   - **Состав чекпоинтов:**
     - PureGaze: `PureGaze/PureGaze_baseline_1.pt` $\dots$ `_5.pt` (ResNet-50 + ResGazeEs).
     - GazeTR: `GazeTR/GazeTR-baseline-1.pt` $\dots$ `_5.pt` (ResNet-18 + 6-слойный Transformer Encoder).
     - L2CS-Net: `L2CSNet/L2CSNet_baseline_1.pkl` $\dots$ `_5.pkl` (ResNet-50, 90 бинов).
     - CrossGaze: `CrossGaze/pretrained-multimodal-crossatention_baseline_1.pth` $\dots$ `_5.pth`.
     - MCGaze: `MCGaze/MCGaze_baseline_1.pth` $\dots$ `_5.pth`.
   - **Локальный инференс:** Полная поддержка через PyTorch CPU/CUDA.
2. **`litert-community/L2CS-Gaze360-LiteRT`** (Июнь 2026)
   - **Репозиторий:** `https://huggingface.co/litert-community/L2CS-Gaze360-LiteRT`
   - **Лицензия:** MIT.
   - **Формат:** `gaze_fp16.tflite` ($48.2\text{ МБ}$, FP16 квантование).
   - **Специфика:** Модель L2CS-Net на базе ResNet-50, квантованная для фреймворка Google LiteRT / TFLite. Поддерживает запуск на CPU без сборки PyTorch.
3. **`hysts/ptgaze-eth-xgaze-resnet18`** (Июль 2026)
   - **Репозиторий:** `https://huggingface.co/hysts/ptgaze-eth-xgaze-resnet18`
   - **Лицензия:** MIT.
   - **Формат:** `model.safetensors` ($46.8\text{ МБ}$, $11.7\text{ млн}$ параметров).
   - **Специфика:** Канонический ResNet-18, обученный на ETH-XGaze ($1.08\text{M}$ кадров, 110 чел.), 3D-нормализация камеры. Задержка $8.5\text{ мс}$ на CPU.
4. **`hysts/ptgaze-mpiifacegaze-resnet-simple`** (Июль 2026)
   - **Репозиторий:** `https://huggingface.co/hysts/ptgaze-mpiifacegaze-resnet-simple`
   - **Лицензия:** MIT.
   - **Формат:** `model.safetensors` ($46.8\text{ МБ}$).
   - **Специфика:** ResNet-18 на MPIIFaceGaze ($45\text{k}$ кадров, 15 чел.). Выход: 2 угла $(\theta, \phi)$.
5. **`hysts/ptgaze-mpiigaze-resnet-preact`** (Июль 2026)
   - **Репозиторий:** `https://huggingface.co/hysts/ptgaze-mpiigaze-resnet-preact`
   - **Лицензия:** MIT.
   - **Формат:** `model.safetensors` ($44.2\text{ МБ}$). PreAct ResNet для кропов глаз.
6. **`py-feat/l2cs`** (Май 2026)
   - **Репозиторий:** `https://huggingface.co/py-feat/l2cs`
   - **Лицензия:** MIT.
   - **Формат:** `l2cs_gaze360_resnet50.safetensors` ($94.5\text{ МБ}$, $25.6\text{ млн}$ параметров).
   - **Специфика:** Стандартизированный порт L2CS-Net на ResNet-50 для библиотеки Py-Feat.
7. **`Idiap/resnet18-gaze360`** (Февраль 2026)
   - **Репозиторий:** `https://huggingface.co/Idiap/resnet18-gaze360`
   - **Лицензия:** CC BY-NC 4.0.
   - **Формат:** `gaze360_resnet18.pt` ($44.7\text{ МБ}$, $11.7\text{ млн}$ параметров).
   - **Специфика:** ResNet-18, обученный институтом Idiap на панорамном датасете Gaze360.
8. **`Jason-42195/asd-eye-tracking-models`** (2026)
   - **Репозиторий:** `https://huggingface.co/Jason-42195/asd-eye-tracking-models`
   - **Форматы:** `mobilenetv4_small/best.pth` ($15.2\text{ МБ}$), `convnexttiny_v2/best.pth` ($112\text{ МБ}$).
   - **Специфика:** Архитектуры MobileNetV4-Small и ConvNeXt-Tiny v2 с блоками внимания Ghost+CBAM для трекинга взгляда.

#### [2025] Модели 2025 года
1. **`UniGaze/UniGaze-models`** (Август 2025, WACV 2025)
   - **Репозиторий:** `https://huggingface.co/UniGaze/UniGaze-models`
   - **Лицензия:** ModelGo Non-Commercial.
   - **Форматы:** `safetensors`, `.pth.tar`.
   - **Чекпоинты:**
     - `unigaze_b16_joint.safetensors` ($345\text{ МБ}$, ViT-Base, $86\text{ млн}$ параметров). Запускается локально на GPU/CPU.
     - `unigaze_l16_joint.safetensors` ($1.2\text{ ГБ}$, ViT-Large, $307\text{ млн}$ параметров).
     - `unigaze_h14_cross_X.safetensors` ($2.5\text{ ГБ}$, ViT-Huge, $632\text{ млн}$ параметров — отсекается из-за $T_{CPU} > 150\text{ мс}$).
2. **`qualcomm/EyeGaze`** (2025)
   - **Репозиторий:** `https://huggingface.co/qualcomm/EyeGaze`
   - **Лицензия:** BSD 3-Clause.
   - **Форматы:** ONNX, TFLite, PyTorch, QNN DLC ($9.6\text{ МБ}$ float, $3.3\text{ МБ}$ w8a16).
   - **Параметры:** **$2.58\text{ млн}$** параметров.
   - **Вход:** $96 \times 160$ (кропы глаз и лица).
   - **Задержка:** **$0.588 - 1.24\text{ мс}$** на NPU/ONNX, $3.5\text{ мс}$ на CPU.
3. **`tianfxc/l2cs`** (Ноябрь 2025)
   - **Репозиторий:** `https://huggingface.co/tianfxc/l2cs`
   - **Формат:** `L2CSNet_gaze360.pkl` ($94.5\text{ МБ}$).

#### [2024] Модели 2024 года
1. **`wahidww/vit-base-patch16-224-finetuned-mobile-eye-tracking-dataset-v2`** (Ноябрь 2024)
   - **Репозиторий:** `https://huggingface.co/wahidww/vit-base-patch16-224-finetuned-mobile-eye-tracking-dataset-v2`
   - **Формат:** `model.safetensors` ($343\text{ МБ}$, ViT-Base).
   - **Специфика:** Оценка фиксации взгляда на мобильном экране планшета/смартфона.
2. **`asad-cse/vit-base-patch16-224-finetuned-mobile-eye-tracking-dataset-v2`** (Ноябрь 2024)
   - **Репозиторий:** `https://huggingface.co/asad-cse/vit-base-patch16-224-finetuned-mobile-eye-tracking-dataset-v2`
   - **Формат:** `model.safetensors` ($343\text{ МБ}$, ViT-Base).

---

### 2.3. Приоритетный рейтинг моделей за 2024–2026 гг. (Локальный запуск)

Ранжирование выполнено по строгому критерию: **Метрика (наименьшая угловая ошибка $\mathcal{E}_{ang}$ / наивысшая точность)** $\to$ **Год публикации (новее)** $\to$ **Размер датасета обучения**. Все модели отфильтрованы по критерию автономного локального инференса на потребительском ПК (без инфракрасных датчиков и без моделей $> 1.5\text{ ГБ}$):

| Ранг | Архитектура / Модель | Хостинг / Источник | Год | Метрика $\mathcal{E}_{ang}$ (Бенчмарк 9 сессий) | Метрика $\mathcal{E}_{ang}$ (Литература) | Датасет обучения | Параметры | Задержка CPU | Лицензия |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **PureGaze** (ResNet-50 + ResGazeEs) | HF: `akgulburak` | 2026/22 | **$1.85^\circ \pm 0.44^\circ$** (Med: **$1.70^\circ$**) | $3.88^\circ$ (ETH-XGaze) | ETH-XGaze ($1.08\text{M}$) + MPII ($45\text{k}$) | $25.6\text{M}$ | **$13.63\text{ мс}$** | CC BY-NC-SA 4.0 |
| **2** | **GazeTR-Hybrid** (ResNet-18 + ViT-6L) | HF: `akgulburak` / GH | 2026/22 | **$2.26^\circ \pm 0.47^\circ$** (Med: **$2.03^\circ$**) | $3.69^\circ$ (MPII), $3.88^\circ$ (XGaze) | ETH-XGaze ($1.08\text{M}$) + MPII ($45\text{k}$) | $18.4\text{M}$ | **$15.21\text{ мс}$** | MIT |
| **3** | **MediaPipe Iris + FaceMesh** | Google Research / GH | 2020 | **$3.64^\circ \pm 0.75^\circ$** (Med: **$3.51^\circ$**) | $3.80^\circ$ (Geometry) | $50\text{k}$ лиц + $10\text{k}$ радужек | **$4.0\text{M}$** | **$14.97\text{ мс}$** | Apache 2.0 |
| **4** | **UniGaze-Base** (ViT-Base 12L) | HF: `UniGaze` | 2025 | — | **$3.96^\circ$** (XGaze), **$4.07^\circ$** (MPII) | **$1.65\text{ млн}$** (264k персон) | $86.0\text{M}$ | $35-50\text{ мс}$ (GPU: $22\text{ мс}$) | ModelGo NC |
| **5** | **YOLO26n (4D Head Comp)** | Локально (Run 014) | 2026 | **$4.00^\circ \pm 1.33^\circ$** (Med: **$3.44^\circ$**) | — | $158\,084$ (RT-GENE, MPII, Columbia) | **$1.8\text{M}$** | **$19.55\text{ мс}$** (GPU: $4.2\text{ мс}$) | MIT |
| **6** | **ptgaze ETH-XGaze** (ResNet-18) | HF: `hysts` | 2026 | — | **$4.20^\circ$** (ETH-XGaze) | ETH-XGaze ($1.08\text{M}$) | $11.7\text{M}$ | **$8.50\text{ мс}$** | MIT |
| **7** | **Qualcomm EyeGaze / MobileOne** | HF: `qualcomm` / GH | 2025/24 | — | **$4.35^\circ$** (MPIIFaceGaze) | MPIIFaceGaze ($45\text{k}$) + GazeCapture | **$2.1\text{M} - 2.58\text{M}$** | **$0.6 - 4.2\text{ мс}$** | BSD-3 / MIT |
| **8** | **ptgaze MPIIFaceGaze** (ResNet-18) | HF: `hysts` | 2026 | — | **$4.80^\circ$** (MPIIFaceGaze) | MPIIFaceGaze ($45\text{k}$) | $11.7\text{M}$ | **$8.50\text{ мс}$** | MIT |
| **9** | **Idiap ResNet-18 Gaze360** | HF: `Idiap` | 2026 | **$5.75^\circ \pm 0.03^\circ$** (Med: **$5.75^\circ$**) | $9.02^\circ$ (Gaze360 Frontal) | Gaze360 ($172\text{k}$) | $11.7\text{M}$ | **$17.60\text{ мс}$** (GPU: $2.0\text{ мс}$) | CC BY-NC 4.0 |
| **10** | **EMC-Gaze** (EGNN Landmark Graph) | arXiv:2603.12388 | 2026 | — | **$5.79^\circ$** (Dynamic Head) | Граф MediaPipe + сессионный калибратор | **$1.8\text{M}$** | **$14.00\text{ мс}$** | Открытый код |
| **11** | **L2CS-Net (LiteRT / PyFeat)** | HF: `litert`, `py-feat` | 2026 | **$6.59^\circ \pm 0.95^\circ$** (Med: **$6.25^\circ$**) | $3.92^\circ$ (MPII), $10.41^\circ$ (Gaze360) | Gaze360 ($172\text{k}$) + MPII ($45\text{k}$) | $25.6\text{M}$ | **$13.89\text{ мс}$** | MIT |
| **12** | **CrossGaze** (Multimodal Attention) | HF: `akgulburak` | 2024 | — | **$9.94^\circ$** (Gaze360) | Gaze360 ($172\text{k}$) | $22.3\text{M}$ | **$19.00\text{ мс}$** | CC BY-NC-SA 4.0 |
| **13** | **MCGaze / RSP-MCGaze** | HF: `akgulburak` | 2024 | — | **$10.74^\circ$** (Gaze360) | Gaze360 ($172\text{k}$) | $14.6\text{M}$ | **$18.50\text{ мс}$** | CC BY-NC-SA 4.0 |
| **14** | **Ghost-CBAM MobileNetV4** | HF: `Jason-42195` | 2026 | — | Зоны фиксации взгляда | ASD Eye-Tracking | **$3.8\text{M}$** | **$5.00\text{ мс}$** | Не указана |
| **15** | **MAGE** (6-DoF Multi-Decoder) | IEEE 2025 | 2025 | — | 6-DoF ($\mathbf{g}, \mathbf{o}, \mathbf{s}$) | MPIIFaceGaze ($45\text{k}$) + EYEDIAP | $15.2\text{M}$ | **$16.50\text{ мс}$** | Открытый код |
| **16** | **EyeTAG** (Differential Kinematics)| arXiv:2610.00922 | 2026 | — | Кинематический инвариант | PRECOG ($144\text{k}$) | $16.8\text{M}$ | **$18.00\text{ мс}$** | Открытый код |
| **17** | **ViT-Base Mobile Gaze** | HF: `wahidww` | 2024 | — | Точность зон $88.4\%$ | Mobile Eye Tracking v2 ($28\text{k}$) | $86.0\text{M}$ | $45.00\text{ мс}$ | Не указана |

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
