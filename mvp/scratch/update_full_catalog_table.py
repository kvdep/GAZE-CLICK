with open('docs/datasets/catalog.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Generate the full corrected table
new_table = """## 2. Сводная сравнительная таблица параметров датасетов по иерархии признаков

Иерархия сортировки признаков (от наиболее важных к наименее важным):
**1. Доступ (Открытый $\to$ Запрос $\to$ Закрытый) $\longrightarrow$ 2. Видны ли глаза (Да $\to$ Нет) $\longrightarrow$ 3. Реальный / Синтетический $\longrightarrow$ 4. Формат (Видео $\to$ Статика)**

> [!NOTE]
> Датасеты разделены по характеру разметки:
> * **Оценка взгляда (Gaze Estimation / Tracking):** на вход подается лицо/глаза, разметка содержит углы поворота глаза $(\theta, \phi)$ или координаты точки на экране $(x_s, y_s)$ — **применимы для курсора**.
> * **Слежение за целью в сцене (Gaze-Following / Target Detection):** на вход подается вся сцена от третьего лица, разметка содержит 2D-координаты объекта в комнате/кадре — **не применимы для курсора**.

| № | Датасет | Год | 1. Доступ | 2. Видны ли глаза | 3. Тип данных | 4. Формат | Назначение разметки | Кадры / Хронометраж | Ссылка |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|:---:|---|
| 1 | **Katrychuk VOG** | 2024 | **Открытый прямой** | **Да** | **Реальный** | **Видео (120 FPS)**| Оценка взгляда (VOG) | 40 сессий (AVI без сжатия) | [Zenodo](https://zenodo.org/records/13852050) |
| 2 | **LPW** | 2016 | **Открытый прямой** | **Да** | **Реальный** | **Видео (95 FPS)** | Центры зрачков | 66 видео (130 856 кадров) | [MPI-INF ZIP](https://datasets.d2.mpi-inf.mpg.de/tonsen/LPW.zip) / [DaRUS](https://doi.org/10.18419/darus-3373) |
| 3 | **RT-GENE** | 2018 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | 3D-вектор взгляда | 114 558 кадров (Kinect v2) | [Zenodo](https://zenodo.org/records/2529036) / [GitHub](https://github.com/Tobias-Fischer/rt_gene) |
| 4 | **VRGaze / GazeShift** | 2026 | **Открытый прямой** | **Да** | **Реальный** | **Видео (60 FPS)** | 3D-взгляд в VR (ИК) | >450 000 кадров (VR ИК) | [HuggingFace](https://huggingface.co/datasets/gazeshift/VRGaze) |
| 5 | **MPIIGaze** | 2015 | **Открытый прямой** | **Да** | **Реальный** | **Статика (клики)** | 3D-взгляд / Экран ПК | 213 659 кадров | [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz) |
| 6 | **MPIIFaceGaze** | 2017 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 3D-взгляд / Лицо | 45 000 кадров | [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip) |
| 7 | **Columbia Gaze** | 2013 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | Сетка дискретных углов | 5 880 фото (5184×3456) | [CS Columbia](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/) |
| 8 | **DrivFace** | 2016 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | Зоны взгляда водителя | 606 фото | [UCI Repo](https://archive.ics.uci.edu/dataset/358/drivface) |
| 9 | **VideoAttentionTarget** | 2020 | **Открытый прямой** | **Нет (сцена)** | **Реальный** | **Видео (30 FPS)** | Цель в сцене (Gaze-Following) | 1 331 видео (>300k кадров) | [GitHub](https://github.com/ejcgt/attention-target-detection) |
| 10 | **ChildPlay Gaze** | 2023 | **Открытый прямой** | **Нет (сцена)** | **Реальный** | **Видео (30 FPS)** | Цель детей в сцене | Непрерывные сессии | [Idiap](https://www.idiap.ch/en/dataset/childplay-gaze) / [Zenodo](https://zenodo.org/records/8254823) |
| 11 | **GazeFollow** | 2015 | **Открытый прямой** | **Нет (сцена)** | **Реальный** | **Статика** | Цель на фото сцены | 125 782 изображения | [GitHub](https://github.com/recasens/Gaze-Following) |
| 12 | **DR(eye)VE** | 2018 | **Открытый прямой** | **Нет (дорога)**| **Реальный** | **Видео (25 FPS)** | Внимание водителя на дороге | 74 видео по 5 мин (555k кадров) | [AImageLab](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8) / [GitHub](https://github.com/ndrplz/dreyeve) |
| 13 | **EGTEA Gaze+** | 2018 | **Открытый прямой** | **Нет (1st person)**| **Реальный** | **Видео (30 FPS)** | Точка взгляда на руках/сцене| 28 часов видео | [GitHub Downloader](https://github.com/amitsou/EGTEA_Gaze_Plus_Downloader) |
| 14 | **GazeBase** | 2021 | **Открытый прямой** | **Нет (сигнал)** | **Реальный** | **Видео (1000 Гц)**| Временные ряды EyeLink | 12 334 сессии EyeLink | [Figshare](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257) |
| 15 | **GazeBaseVR** | 2022 | **Открытый прямой** | **Нет (сигнал)** | **Реальный** | **Видео (250 Гц)** | Временные ряды VR Tobii | >8 000 сессий VR Tobii | [Figshare](https://figshare.com/articles/dataset/GazeBaseVR/19643442) |
| 16 | **SynthesEyes** | 2015 | **Открытый прямой** | **Да** | **Синтетический** | **Статика** | 3D-взгляд / Зрачок / Лимб | 11 382 рендера ($1280 \\times 720$) | [Apollo](https://www.repository.cam.ac.uk/handle/1810/252736) |
| 17 | **UnityEyes** | 2016 | **Открытый прямой** | **Да** | **Синтетический** | **Генератор** | 3D-взгляд / Ландмарки глаз | Неограниченно | [Cambridge](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) |
| 18 | **WEyeDS** | 2024 | **По запросу (Qualtrics)**| **Да** | **Реальный** | **Видео (30 FPS)** | Координаты экрана ПК | >320 000 кадров (54 веб-камеры) | [Qualtrics](https://richmond.ca1.qualtrics.com/jfe/form/SV_3pEzZBWmLms2jH0) / [RobbinsLab](https://robbinslab.github.io/weyeds/) |
| 19 | **EVE** | 2020 | **По запросу** | **Да** | **Реальный** | **Видео (30/60 FPS)**| Экран ПК / Синхронные глаза | 12 300 000 кадров (105 часов) | [ETH AIT](https://ait.ethz.ch/eve) |
| 20 | **PRECOG** | 2026 | **По запросу** | **Да** | **Реальный** | **Видео (195 FPS)**| Монитор ПК (узкий диапазон) | 140 сессий (190 часов) | [USC SAIL](https://arxiv.org/abs/2610.11036) |
| 21 | **EyeDiap** | 2014 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | Экран монитора / 3D-шарик | 94 видео (~5 часов) | [Idiap](https://www.idiap.ch/en/dataset/eyediap) |
| 22 | **EyeTrax** | 2026 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | Траектории веб-камеры ПК | 60 сессий (веб-камера) | [arXiv](https://arxiv.org/abs/2603.12388) |
| 23 | **DMD (Driver)** | 2020 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | Зоны взгляда водителя | 41.2 часа (RGB+IR+Depth) | [Vicomtech](https://dmd.vicomtech.org/) |
| 24 | **TabletGaze** | 2016 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS)** | Планшетный экран | 816 видео | [Rice Univ](http://sharma.ece.rice.edu/TabletGaze/) |
| 25 | **Gaze360** | 2019 | **По запросу** | **Да** | **Реальный** | **Видео (30 FPS, клипы)**| 3D-взгляд 360 градусов | 197 588 кадров | [MIT CSAIL](http://gaze360.csail.mit.edu/) |
| 26 | **UT Multiview** | 2014 | **По запросу** | **Да** | **Реальный** | **Статика (мультикам)**| 3D-взгляд (8 камер) | 64 000 кадров (8 камер) | [UT Vision](https://www.ut-vision.org/datasets/utmultiview/) |
| 27 | **GazeCapture** | 2016 | **По запросу** | **Да** | **Реальный** | **Статика (серии)**| Экран смартфона/планшета | 2 450 197 кадров | [MIT CSAIL](https://gazecapture.csail.mit.edu/) |
| 28 | **ETH-XGaze** | 2020 | **По запросу** | **Да** | **Реальный** | **Статика (18 камер)** | 3D-взгляд высокого разрешения| 1 083 492 кадра | [ETH AIT](https://ait.ethz.ch/eve) |
| 29 | **DiGaze** | 2020 | **По запросу** | **Да** | **Реальный** | **Статика** | Экран мобильного телефона | ~120 000 кадров | [DiGaze](https://github.com/) |
| 30 | **HybridGaze** | 2026 | **В релизе (статья)** | **Да** | **Реальный** | **Видео (30 FPS)** | Экран ПК / Tobii 5 | 151 128 кадров (Full HD) | [IJET DOI](https://doi.org/10.24425/ijet.2026.157892) / [PAN](https://journals.pan.pl/dlibra/publication/157892/edition/138498/content) |
| 31 | **Ego4D Gaze** | 2022 | **По запросу** | **Нет (мир) / Да (ИК)**| **Реальный** | **Видео (30/60 FPS)**| Взгляд от 1-го лица | Сотни часов видео | [Ego4D](https://ego4d-data.org/) |
| 32 | **BEDLAM-Gaze** | 2023 | **По запросу** | **Да** | **Синтетический** | **Видео / 3D аним.**| 3D-глаза в UE5 (Lumen) | >1 млн кадров (UE5) | [MPI](https://bedlam.is.tue.mpg.de/) |
| 33 | **NVGaze (Real)** | 2019 | **Закрытый** | **Да** | **Реальный** | **Видео (120 FPS)**| Окологлазные ИК-кадры | 2 500 000 кадров (ИК) | [NVIDIA](https://research.nvidia.com/publication/2019-05_nvgaze-dataset) |
| 34 | **OpenEDS2020** | 2020 | **Закрытый** | **Да** | **Реальный** | **Видео (100 FPS)**| Сегментация глаза в VR | 550 400 кадров | [Meta](https://github.com/facebookresearch/OpenEDS2020) |
| 35 | **OpenEDS** | 2019 | **Закрытый** | **Да** | **Реальный** | **Видео (200 FPS)**| Сегментация зрачка/радужки | 91 200 видеокадров | [Meta](https://research.facebook.com/) |
| 36 | **TEyeD** | 2021 | **Закрытый / Zenodo** | **Да** | **Реальный** | **Видео (60-200 FPS)**| 3D-взгляд / Маски зрачка | >20 000 000 кадров | [GitHub](https://github.com/Fuhl/TEyeD) |
| 37 | **NVGaze (Syn)** | 2019 | **Закрытый** | **Да** | **Синтетический** | **Статика (OptiX)**| Фотореалистичный рендер NIR | 2 000 000 кадров | [NVIDIA](https://research.nvidia.com/) |
| 38 | **EyeSyn** | 2024 | **Закрытый / Акад.** | **Да** | **Синтетический** | **Генератор** | Преломление роговицы NIR | Синтетика NIR | [GitHub](https://github.com/) |"""

# Replace from '## 2. Сводная сравнительная таблица' to '## 3. Протокол самостоятельного сбора датасета'
sec2_start = text.find('## 2. Сводная сравнительная таблица')
sec3_start = text.find('## 3. Протокол самостоятельного сбора датасета')

if sec2_start != -1 and sec3_start != -1:
    new_content = text[:sec2_start] + new_table + "\n\n---\n\n" + text[sec3_start:]
    with open('docs/datasets/catalog.md', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Catalog table cleanly updated!")
else:
    print(f"Indices: sec2={sec2_start}, sec3={sec3_start}")
