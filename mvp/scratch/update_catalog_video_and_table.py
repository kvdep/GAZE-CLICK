import os

catalog_path = r'D:\ПРОФ\ВУЗ\ВКР\docs\datasets\catalog.md'

with open(catalog_path, 'r', encoding='utf-8') as f:
    text = f.read()

split_mark = '## 2. Сводная сравнительная таблица параметров и условий доступа к датасетам'
assert split_mark in text

part1, rest = text.split(split_mark, 1)
table_end_mark = '## 3. Протокол самостоятельного сбора датасета'
assert table_end_mark in rest

_, part3 = rest.split(table_end_mark, 1)

new_video_datasets = """### 1.31. VideoAttentionTarget

* **Авторы и год публикации:** Eunji Chong, Dawei Du, Kyoungmin Lee, James M. Rehg (Georgia Tech), CVPR 2019.
* **Объем видеопотока:** 1 331 непрерывная видеопоследовательность (более 300 000 кадров), частота 30 FPS.
* **Разметка:** Покадровая непрерывная траектория взгляда каждого человека в кадре (3D-векторы и 2D-координаты точки фиксации внимания).
* **Лицензия и доступ:** Открытый прямой доступ (архивы на Zenodo и AWS).
* **Репозиторий:** `https://github.com/ejcgt/attention-target-detection`

---

### 1.32. ChildPlay Gaze

* **Авторы и год публикации:** Fiona Ryan et al. (Georgia Tech), CVPR 2025.
* **Объем видеопотока:** Непрерывные эпизоды естественного диалогового взаимодействия (30 FPS), крупные планы лиц и глаз без окклюзий.
* **Разметка:** Покадровый 3D-луч взгляда и 2D-координаты точки фиксации.
* **Лицензия и доступ:** Открытый прямой доступ.
* **Репозиторий:** `https://github.com/fkryan/childplay`

---

### 1.33. DR(eye)VE

* **Авторы и год публикации:** Stefano Alletto et al. (Università di Modena e Reggio Emilia), IEEE TPAMI 2018.
* **Объем видеопотока:** 555 000 видеокадров (74 видеозаписи по 5 минут при 25 FPS).
* **Специфика:** Камера направлена вперед из лобового стекла на дорогу (глаза водителя в кадр не попадают, фиксируются наголовным SMI ETG 2W 60 Гц).
* **Лицензия и доступ:** Открытый прямой доступ.
* **Страница:** `https://imagelab.ing.unimore.it/dreve/`

---

### 1.34. EGTEA Gaze+

* **Авторы и год публикации:** Yin Li, Miao Liu, James M. Rehg (Georgia Tech), 2018 / 2021.
* **Объем видеопотока:** 28 часов непрерывного видео от первого лица (30 FPS).
* **Специфика:** Наружная камера снимает сцену действий руками; глазная щель снимается скрытыми ИК-датчиками очков SMI.
* **Лицензия и доступ:** Открытый прямой доступ.
* **Страница:** `https://cbs.ic.gatech.edu/fpv/`

---

### 1.35. PRECOG (Narrow-Range Screen Viewing)

* **Авторы и год публикации:** Jordan Prescott, Kleanthis Avramidis, Shrikanth Narayanan (USC SAIL), arXiv:2610.11036, Октябрь 2026.
* **Объем видеопотока:** 140 длительных непрерывных видеозаписей (190 часов суммарно) с камеры FLIR Blackfly S с частотой 195 FPS. Крупный план лица и глаз.
* **Разметка:** Бинокулярный видеоокулограф EyeLink 1000 Plus с частотой 250 Гц.
* **Лицензия и доступ:** По запросу в лабораторию SAIL USC.

---

### 1.36. EyeTrax

* **Авторы и год публикации:** Chenkai Zhang, arXiv:2603.12388, Март 2026.
* **Объем видеопотока:** 60 сессий непрерывного интерактивного видеопотока с веб-камер (720p, 30 FPS).
* **Разметка:** Калибровочные 9-точечные траектории и непрерывная фиксация взгляда на мониторе.
* **Лицензия и доступ:** По запросу автору публикации EMC-Gaze.

---

### 1.37. Ego4D (Eye Gaze Subset)

* **Авторы и год публикации:** Kristen Grauman et al., CVPR 2022.
* **Объем видеопотока:** Сотни часов непрерывного видео первого лица (30/60 FPS).
* **Лицензия и доступ:** По запросу (Ego4D Consortium License).
* **Страница:** `https://ego4d-data.org/`

---

"""

new_table = """## 2. Сводная сравнительная таблица параметров датасетов по иерархии признаков

Иерархия сортировки признаков (от наиболее важных к наименее важным):
**1. Доступ (Открытый $\\to$ Запрос $\\to$ Закрытый) $\\longrightarrow$ 2. Видны ли глаза (Да $\\to$ Нет) $\\longrightarrow$ 3. Реальный / Синтетический $\\longrightarrow$ 4. Формат (Видео $\\to$ Статика)**

| № | Датасет | Год | 1. Доступ | 2. Видны ли глаза | 3. Тип данных | 4. Формат | Кадры / Хронометраж | Ссылка |
|---|---|:---:|:---:|:---:|:---:|:---:|---|---|
| 1 | **VideoAttentionTarget** | 2019 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | 1 331 видео (>300k кадров) | [GitHub](https://github.com/ejcgt/attention-target-detection) |
| 2 | **ChildPlay Gaze** | 2025 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | Непрерывные сессии | [GitHub](https://github.com/fkryan/childplay) |
| 3 | **WEyeDS** | 2024 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | >320 000 кадров (54 веб-камеры) | [ETRA](https://dl.acm.org/) |
| 4 | **Katrychuk VOG** | 2024 | **Открытый прямой** | **Да** | **Реальный** | **Видео (120 FPS)**| 40 сессий (AVI без сжатия) | [Zenodo](https://zenodo.org/records/13931668) |
| 5 | **LPW** | 2016 | **Открытый прямой** | **Да** | **Реальный** | **Видео (95 FPS)** | 66 видео (130 856 кадров) | [DaRUS](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-1718) |
| 6 | **HybridGaze** | 2026 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | 151 123 кадра (Full HD) | [Kaggle](https://www.kaggle.com/datasets/grzegorzchwesiuk/hybridgaze) |
| 7 | **RT-GENE** | 2018 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | 114 558 кадров (Kinect v2) | [Zenodo](https://zenodo.org/records/2529036) |
| 8 | **VRGaze / GazeShift** | 2026 | **Открытый прямой** | **Да** | **Реальный** | **Видео (60 FPS)** | >450 000 кадров (VR ИК) | [HuggingFace](https://huggingface.co/datasets/gazeshift/VRGaze) |
| 9 | **MPIIGaze** | 2015 | **Открытый прямой** | **Да** | **Реальный** | **Статика (клики)** | 213 659 кадров | [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz) |
| 10 | **MPIIFaceGaze** | 2017 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 45 000 кадров | [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip) |
| 11 | **Columbia Gaze** | 2013 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 5 880 фото (5184x3456) | [CS Columbia](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/) |
| 12 | **GazeFollow** | 2015 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 125 782 изображения | [MIT CSAIL](https://gazefollow.csail.mit.edu/) |
| 13 | **DrivFace** | 2016 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 606 фото | [UCI Repo](https://archive.ics.uci.edu/dataset/358/drivface) |
| 14 | **DR(eye)VE** | 2018 | **Открытый прямой** | **Нет (камера дороги)**| **Реальный** | **Видео (25 FPS)** | 74 видео по 5 мин (555k кадров) | [UniMoRe](https://imagelab.ing.unimore.it/dreve/) |
| 15 | **EGTEA Gaze+** | 2018 | **Открытый прямой** | **Нет (1st person)**| **Реальный** | **Видео (30 FPS)** | 28 часов видео | [GeorgiaTech](https://cbs.ic.gatech.edu/fpv/) |
| 16 | **GazeBase** | 2021 | **Открытый прямой** | **Нет (сигнал)** | **Реальный** | **Видео (1000 Гц)**| 12 334 сессии EyeLink | [Figshare](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257) |
| 17 | **GazeBaseVR** | 2022 | **Открытый прямой** | **Нет (сигнал)** | **Реальный** | **Видео (250 Гц)** | >8 000 сессий VR Tobii | [Figshare](https://figshare.com/articles/dataset/GazeBaseVR/19643442) |
| 18 | **SynthesEyes** | 2015 | **Открытый прямой** | **Да** | **Синтетический** | **Статика** | 11 382 рендера ($1280 \\times 720$) | [Apollo](https://www.repository.cam.ac.uk/handle/1810/252736) |
| 19 | **UnityEyes** | 2016 | **Открытый прямой** | **Да** | **Синтетический** | **Генератор** | Неограниченно | [GitHub](https://github.com/errollw/UnityEyes) |
| 20 | **EVE** | 2020 | **По запросу** | **Да** | **Реальный** | **Видео (30/60 FPS)**| 12 300 000 кадров (105 часов) | [ETH AIT](https://ait.ethz.ch/eve) |
| 21 | **PRECOG** | 2026 | **По запросу** | **Да** | **Реальный** | **Видео (195 FPS)**| 140 сессий (190 часов) | [USC SAIL](https://arxiv.org/abs/2610.11036) |
| 22 | **EyeDiap** | 2014 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | 94 видео (~5 часов) | [Idiap](https://www.idiap.ch/en/dataset/eyediap) |
| 23 | **EyeTrax** | 2026 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | 60 сессий (веб-камера) | [arXiv](https://arxiv.org/abs/2603.12388) |
| 24 | **DMD (Driver)** | 2020 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | 41.2 часа (RGB+IR+Depth) | [Vicomtech](https://dmd.vicomtech.org/) |
| 25 | **TabletGaze** | 2016 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | 816 видео | [Rice Univ](http://sharma.ece.rice.edu/TabletGaze/) |
| 26 | **Gaze360** | 2019 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS, клипы)**| 197 588 кадров | [MIT CSAIL](http://gaze360.csail.mit.edu/) |
| 27 | **UT Multiview** | 2014 | **По запросу** | **Да** | **Реальный** | **Статика (мультикам)**| 64 000 кадров (8 камер) | [UT Vision](https://www.ut-vision.org/datasets/utmultiview/) |
| 28 | **GazeCapture** | 2016 | **По запросу** | **Да** | **Реальный** | **Статика (серии)**| 2 450 197 кадров | [MIT CSAIL](https://gazecapture.csail.mit.edu/) |
| 29 | **ETH-XGaze** | 2020 | **По запросу** | **Да** | **Реальный** | **Статика (18 камер)** | 1 083 492 кадра | [ETH AIT](https://ait.ethz.ch/eve) |
| 30 | **DiGaze** | 2020 | **По запросу** | **Да** | **Реальный** | **Статика** | ~120 000 кадров | [DiGaze](https://github.com/) |
| 31 | **Ego4D Gaze** | 2022 | **По запросу** | **Нет (мир) / Да (ИК)**| **Реальный** | **Видео (30/60 FPS)**| Сотни часов видео | [Ego4D](https://ego4d-data.org/) |
| 32 | **BEDLAM-Gaze** | 2023 | **По запросу** | **Да** | **Синтетический** | **Видео / 3D аним.**| >1 млн кадров (UE5) | [MPI](https://bedlam.is.tue.mpg.de/) |
| 33 | **NVGaze (Real)** | 2019 | **Закрытый** | **Да** | **Реальный** | **Видео (120 FPS)**| 2 500 000 кадров (ИК) | [NVIDIA](https://research.nvidia.com/publication/2019-05_nvgaze-dataset) |
| 34 | **OpenEDS2020** | 2020 | **Закрытый** | **Да** | **Реальный** | **Видео (100 FPS)**| 550 400 кадров | [Meta](https://github.com/facebookresearch/OpenEDS2020) |
| 35 | **OpenEDS** | 2019 | **Закрытый** | **Да** | **Реальный** | **Видео (200 FPS)**| 91 200 видеокадров | [Meta](https://research.facebook.com/) |
| 36 | **TEyeD** | 2021 | **Закрытый / Zenodo** | **Да** | **Реальный** | **Видео (60-200 FPS)**| >20 000 000 кадров | [GitHub](https://github.com/Fuhl/TEyeD) |
| 37 | **NVGaze (Syn)** | 2019 | **Закрытый** | **Да** | **Синтетический** | **Статика (OptiX)**| 2 000 000 кадров | [NVIDIA](https://research.nvidia.com/) |
| 38 | **EyeSyn** | 2024 | **Закрытый / Акад.** | **Да** | **Синтетический** | **Генератор** | Синтетика NIR | [GitHub](https://github.com/) |

---

"""

full_content = part1 + new_video_datasets + new_table + table_end_mark + part3

with open(catalog_path, 'w', encoding='utf-8') as f:
    f.write(full_content)

print("catalog.md updated with video datasets and 4-level hierarchy table successfully!")
