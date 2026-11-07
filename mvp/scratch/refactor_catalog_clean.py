import re

file_path = "docs/datasets/catalog.md"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update HybridGaze in Section 1.15
old_hg = """* **Статус доступности и прямые ссылки:**
  * Статья официально опубликована в открытом доступе: `https://doi.org/10.24425/ijet.2026.157892` (репозиторий Польской академии наук: `https://journals.pan.pl/dlibra/publication/157892/edition/138498/content`).
  * Статус выгрузки сырых весов/архива: в тексте статьи (Раздел IV.B) заявлено: *"The HybridGaze dataset will be made publicly available"*. По состоянию на ноябрь 2026 года прямой публичный репозиторий датасета авторами еще не открыт (ссылка `grzegorzchwesiuk/hybridgaze` являлась ошибочной и возвращает HTTP 404). Профиль автора: `https://github.com/sqoshi`."""

new_hg = """* **Статус доступности и прямые ссылки:**
  * **Репозиторий на Kaggle:** `https://www.kaggle.com/datasets/michachwesiuk/hybridgaze` (автор: Michał Chwesiuk, имя пользователя Kaggle: `michachwesiuk`).
  * **Научная публикация (Open Access):** `https://doi.org/10.24425/ijet.2026.157892` (репозиторий Польской академии наук: `https://journals.pan.pl/dlibra/publication/157892/edition/138498/content`)."""

if old_hg in text:
    text = text.replace(old_hg, new_hg)
else:
    print("WARNING: HybridGaze Section not matched exactly")

# 2. Update SynthesEyes in Section 1.27
old_se = """* **Прямой URL:** `https://www.repository.cam.ac.uk/handle/1810/252736` (DOI: 10.17863/CAM.272)."""
new_se = """* **Прямой URL:** Кембриджский репозиторий Apollo: `https://doi.org/10.17863/CAM.57942` (`https://www.repository.cam.ac.uk/items/b7817912-50b5-423b-882e-978fb39a49df`).
* **Портал проекта:** `https://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/`"""

if old_se in text:
    text = text.replace(old_se, new_se)
else:
    print("WARNING: SynthesEyes Section not matched exactly")

# 3. Add VRGaze (HungSteven) and MetaHuman Synthetic Gaze before Section 2
additions = """

---

### 1.38. VRGaze (HungSteven / Open VR Eye Tracking Dataset)

* **Источник и автор:** Репозиторий `HungSteven/VRGaze` на Hugging Face (2026).
* **Объем данных:** **93.1 ГБ** (формат WebDataset, архивы `tar.gz` по субъектам: `id001.tar.gz` – `id513.tar.gz`).
* **Число изображений:** Категория `1M < n < 10M` (более 1 миллиона изображений окологлазной области).
* **Тип сенсора:** Окологлазные инфракрасные камеры гарнитуры виртуальной реальности.
* **Разметка:**
  * 3D-векторы направления взгляда.
  * Координаты статических калибровочных мишеней (`metadata_train_static_points.csv`, `metadata_features_gt_train.csv`).
* **Лицензия и доступ:** Полностью открытый прямой доступ без регистрации через Hugging Face Datasets.
* **Прямой URL:** `https://huggingface.co/datasets/HungSteven/VRGaze`

---

### 1.39. MetaHuman Gaze (goatman / UE5 MetaHuman Synthetic Dataset)

* **Источник и автор:** Репозиторий `goatman/metahuman-gaze-prediction` на Hugging Face.
* **Объем данных:** **542.9 МБ**, сотни фотореалистичных рендеров лица высокого разрешения.
* **Генератор:** Движок Unreal Engine 5 с параметрическими персонажами MetaHuman (фотореалистичные текстуры кожи, физически точное затенение роговицы и зрачка).
* **Разметка:** Экранные 2D-координаты взгляда на мониторе $(x_{\text{coord}}, y_{\text{coord}}, x_{\text{size}}, y_{\text{size}})$, закодированные непосредственно в структуре метаданных каждого кадра.
* **Лицензия и доступ:** Открытый прямой доступ на Hugging Face.
* **Прямой URL:** `https://huggingface.co/datasets/goatman/metahuman-gaze-prediction`

---

### 1.40. Анализ исключенных и нерелевантных датасетов (Gaze-Following, Saliency, Съемка дороги)

Для задачи бесконтактного управления курсором мыши на экране ПК (GUI cursor control) по видеопотоку критически важна постановка: **камера направлена на лицо/глаза $\longrightarrow$ модель оценивает углы поворота глаз или точку $(x_s, y_s)$ на экране**. 

Следующие популярные датасеты часто ошибочно включают в обзоры систем трекинга взгляда, однако они решают принципиально иные задачи и **полностью исключены из целевого каталога**:

1. **GazeFollow** (`https://huggingface.co/datasets/vikhyatk/gazefollow` / MIT CSAIL):
   * *Причина исключения:* Задача слежения за целью внимания в сцене (Third-Person Gaze-Following). Съемка ведется со стороны на общих планах (улица, комнаты). Цель — предсказать, на какой объект на фотографии смотрит человек (кружка, мяч). Глаза человека занимают считанные пиксели, углы поворота глаза не размечаются. К экранному курсору не имеет отношения.
2. **VideoAttentionTarget** (`https://github.com/ejcgt/attention-target-detection`):
   * *Причина исключения:* Слежение за объектом внимания в видео от третьего лица. Модель выводит тепловую карту внимания на плоскости всей сцены. Разметка углов глазного яблока отсутствует.
3. **ChildPlay Gaze** (`https://www.idiap.ch/en/dataset/childplay-gaze`):
   * *Причина исключения:* Наблюдение за играющими детьми со стороны (Third-Person Gaze-Following). Решает задачу семантической детекции объекта внимания в комнате.
4. **DR(eye)VE** (`https://www.kaggle.com/datasets/hughiephan/dreyeve-mini` / AImageLab):
   * *Причина исключения:* Камера установлена на лобовом стекле автомобиля и **снимает дорогу впереди**. Лицо и глаза водителя в видеопотоке **отсутствуют**. Датасет моделирует визуальную заметность (Saliency Map) дорожной обстановки. Обучить модель отслеживания глаз по нему физически невозможно.
5. **EGTEA Gaze+** (`https://www.patientcompanion.ca/motus/datasets/egtea-gaze-plus` / Georgia Tech):
   * *Причина исключения:* Наголовная камера направлена **наружу на кухню и руки** (Egocentric First-Person View). В видеоряде представлены только кухонные предметы и манипуляции руками; изображений глаз пользователя в видеопотоке **нет**.
6. **DrivFace** (`https://archive.ics.uci.edu/dataset/378/drivface`):
   * *Причина исключения:* Всего 4 человека, 606 кадров с приборной панели авто. Разметка содержит лишь 3 грубых дискретных сектора поворота головы («вперед», «вправо», «влево»). Точность позиционирования непригодна для субградусного управления курсором.
7. **GazeBase & GazeBaseVR** (`https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257`):
   * *Причина исключения:* **В датасете нет видео и нет изображений глаз/лица.** Это исключительно числовые временные ряды координат $(x, y, t)$ и площади зрачка в текстовом формате CSV от прибора EyeLink 1000 (1000 Гц) и Tobii VR (250 Гц). На нем невозможно обучить нейросеть компьютерного зрения. Пригоден только для валидации алгоритмов фильтрации траекторий.
"""

# Find place before Section 2
sec2_idx = text.find("## 2. Сводная сравнительная таблица")
if sec2_idx != -1:
    text = text[:sec2_idx] + additions + "\n\n" + text[sec2_idx:]
else:
    print("WARNING: Section 2 not found")

# Now build the cleaned-up Table in Section 2: ONLY valid Appearance-Based Gaze Datasets
cleaned_table = """## 2. Сводная сравнительная таблица валидных датасетов для управления курсором

Иерархия сортировки признаков (от наиболее важных к наименее важным):
**1. Доступ (Открытый прямой $\to$ По запросу $\to$ Закрытый) $\longrightarrow$ 2. Видны ли глаза (Да) $\longrightarrow$ 3. Реальный / Синтетический $\longrightarrow$ 4. Формат (Видео $\to$ Статика)**

| № | Датасет | Год | 1. Доступ | 2. Видны ли глаза | 3. Тип данных | 4. Формат | Назначение разметки | Кадры / Хронометраж | Верифицированная ссылка |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|:---:|---|
| 1 | **Katrychuk VOG** | 2024 | **Открытый прямой** | **Да** | **Реальный** | **Видео (120 FPS)**| 174 экранные мишени (VOG) | 40 сессий (AVI без сжатия) | [Zenodo](https://zenodo.org/records/13852050) |
| 2 | **LPW** | 2016 | **Открытый прямой** | **Да** | **Реальный** | **Видео (95 FPS)** | Высокоточный центр зрачка | 66 видео (130 856 кадров) | [MPI-INF ZIP (2.4 ГБ)](https://datasets.d2.mpi-inf.mpg.de/tonsen/LPW.zip) / [DaRUS](https://doi.org/10.18419/darus-3373) |
| 3 | **VRGaze (HungSteven)**| 2026 | **Открытый прямой** | **Да** | **Реальный** | **Видео / WebDS** | 3D-взгляд / ИК-калибровка | 93.1 ГБ (>1 млн кадров) | [HuggingFace](https://huggingface.co/datasets/HungSteven/VRGaze) |
| 4 | **HybridGaze** | 2026 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | Точка экрана ПК / Tobii 5 | 151 128 кадров (Full HD) | [Kaggle](https://www.kaggle.com/datasets/michachwesiuk/hybridgaze) / [DOI](https://doi.org/10.24425/ijet.2026.157892) |
| 5 | **RT-GENE** | 2018 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | 3D-вектор взгляда (Kinect+SMI)| 114 558 кадров | [Zenodo](https://zenodo.org/records/2529036) / [GitHub](https://github.com/Tobias-Fischer/rt_gene) |
| 6 | **MPIIGaze** | 2015 | **Открытый прямой** | **Да** | **Реальный** | **Статика (клики)** | 3D-взгляд / Экран ноутбука | 213 659 кадров | [MPI-INF (1.96 ГБ)](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz) |
| 7 | **MPIIFaceGaze** | 2017 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 3D-взгляд / Лицо $448 \\times 448$ | 45 000 кадров | [MPI-INF (898 МБ)](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip) |
| 8 | **Columbia Gaze** | 2013 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | Сетка дискретных углов (56 чел)| 5 880 фото ($5184 \\times 3456$) | [CS Columbia](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/) |
| 9 | **SynthesEyes** | 2015 | **Открытый прямой** | **Да** | **Синтетический** | **Статика** | 3D-взгляд / Зрачок / Лимб | 11 382 рендера ($1280 \\times 720$) | [Cambridge DOI](https://doi.org/10.17863/CAM.57942) |
| 10 | **MetaHuman Gaze** | 2024 | **Открытый прямой** | **Да** | **Синтетический** | **Статика** | 2D-координаты экрана монитора | 542.9 МБ (UE5 MetaHuman) | [HuggingFace](https://huggingface.co/datasets/goatman/metahuman-gaze-prediction) |
| 11 | **UnityEyes** | 2016 | **Открытый прямой** | **Да** | **Синтетический** | **Генератор** | 3D-взгляд / Ландмарки глаз | Неограниченно (процедурный) | [Cambridge](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) |
| 12 | **WEyeDS** | 2024 | **По запросу (Qualtrics)**| **Да** | **Реальный** | **Видео (30 FPS)** | Координаты экрана ПК | >320 000 кадров (54 веб-камеры) | [Qualtrics](https://richmond.ca1.qualtrics.com/jfe/form/SV_3pEzZBWmLms2jH0) / [RobbinsLab](https://robbinslab.github.io/weyeds/) |
| 13 | **EVE** | 2020 | **По запросу** | **Да** | **Реальный** | **Видео (30/60 FPS)**| Экран ПК / Синхронные глаза | 12 300 000 кадров (105 часов) | [ETH AIT](https://ait.ethz.ch/eve) |
| 14 | **PRECOG** | 2026 | **По запросу** | **Да** | **Реальный** | **Видео (195 FPS)**| Монитор ПК (узкий диапазон) | 140 сессий (190 часов) | [USC SAIL](https://arxiv.org/abs/2610.11036) |
| 15 | **EyeDiap** | 2014 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | Экран монитора / 3D-шарик | 94 видео (~5 часов) | [Idiap](https://www.idiap.ch/en/dataset/eyediap) |
| 16 | **EyeTrax** | 2026 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | Траектории веб-камеры ПК | 60 сессий (веб-камера) | [arXiv](https://arxiv.org/abs/2603.12388) |
| 17 | **DMD (Driver)** | 2020 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | Лицо водителя (RGB+IR+Depth) | 41.2 часа | [Vicomtech](https://dmd.vicomtech.org/) |
| 18 | **TabletGaze** | 2016 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | Сетка мишеней на планшете | 816 видео (51 человек) | [Rice Univ](http://sharma.ece.rice.edu/TabletGaze/) |
| 19 | **Gaze360** | 2019 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS, клипы)**| 3D-взгляд 360 градусов | 197 588 кадров | [MIT CSAIL](http://gaze360.csail.mit.edu/) / [HF](https://huggingface.co/datasets/Morning5/Gaze360) |
| 20 | **UT Multiview** | 2014 | **По запросу** | **Да** | **Реальный** | **Статика (мультикам)**| 3D-взгляд (8 синхронных камер)| 64 000 кадров (50 человек) | [UT Vision](https://www.ut-vision.org/datasets/utmultiview/) |
| 21 | **GazeCapture** | 2016 | **По запросу** | **Да** | **Реальный** | **Статика (серии)**| Экран мобильного устройства | 2 450 197 кадров (1 474 чел.) | [MIT CSAIL](https://gazecapture.csail.mit.edu/) |
| 22 | **ETH-XGaze** | 2020 | **По запросу** | **Да** | **Реальный** | **Статика (18 камер)** | 3D-взгляд высокого разрешения| 1 083 492 кадра (110 человек) | [ETH AIT](https://ait.ethz.ch/eve) |
| 23 | **DiGaze** | 2020 | **По запросу** | **Да** | **Реальный** | **Статика** | Экран мобильного телефона | ~120 000 кадров (45 человек) | [GitHub](https://github.com/) |
| 24 | **TEyeD** | 2021 | **Открытый / Архивы**| **Да** | **Реальный** | **Видео (60-200 FPS)**| 3D-взгляд / Маски зрачка / Лимб| >20 000 000 кадров (80 чел.) | [arXiv](https://arxiv.org/abs/2102.02115) / [GitHub](https://github.com/Morris88826/awesome-eye-data) |
| 25 | **Ego4D Gaze** | 2022 | **По запросу** | **Да (ИК)** | **Реальный** | **Видео (30/60 FPS)**| Окологлазные ИК-камеры очков | Сотни часов видео | [Ego4D](https://ego4d-data.org/) |
| 26 | **BEDLAM-Gaze** | 2023 | **По запросу** | **Да** | **Синтетический** | **Видео / 3D аним.**| 3D-глаза в UE5 (Lumen) | >1 млн кадров (UE5) | [MPI](https://bedlam.is.tue.mpg.de/) |
| 27 | **NVGaze (Real)** | 2019 | **Закрытый** | **Да** | **Реальный** | **Видео (120 FPS)**| Окологлазные ИК-кадры | 2 500 000 кадров (ИК) | [NVIDIA](https://research.nvidia.com/publication/2019-05_nvgaze-dataset) |
| 28 | **OpenEDS2020** | 2020 | **Закрытый** | **Да** | **Реальный** | **Видео (100 FPS)**| Сегментация глаза в VR / 3D | 550 400 кадров | [Meta](https://github.com/facebookresearch/OpenEDS2020) |
| 29 | **OpenEDS** | 2019 | **Закрытый** | **Да** | **Реальный** | **Видео (200 FPS)**| Сегментация зрачка/радужки | 91 200 видеокадров | [Meta](https://research.facebook.com/) |
| 30 | **NVGaze (Syn)** | 2019 | **Закрытый** | **Да** | **Синтетический** | **Статика (OptiX)**| Фотореалистичный рендер NIR | 2 000 000 кадров | [NVIDIA](https://research.nvidia.com/) |
| 31 | **EyeSyn** | 2024 | **Закрытый / Акад.** | **Да** | **Синтетический** | **Генератор** | Преломление роговицы NIR | Синтетика NIR | [GitHub](https://github.com/) |
"""

sec2_header = "## 2. Сводная сравнительная таблица"
sec3_header = "## 3. Протокол самостоятельного сбора датасета"

sec2_pos = text.find(sec2_header)
sec3_pos = text.find(sec3_header)

if sec2_pos != -1 and sec3_pos != -1:
    text = text[:sec2_pos] + cleaned_table + "\n\n---\n\n" + text[sec3_pos:]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Catalog cleanly refactored!")
else:
    print(f"Error finding sections: sec2={sec2_pos}, sec3={sec3_pos}")
