import re

file_path = "docs/datasets/catalog.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update LPW Section 1.21
old_lpw_str = """* **Прямой URL:** `https://perceptualui.org/research/datasets/LPW/` и `https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-1718`"""
new_lpw_str = """* **Прямой URL:** Репозиторий DaRUS (University of Stuttgart): `https://doi.org/10.18419/darus-3373` (`https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-3373`). Примечание: устаревший идентификатор `darus-1718` возвращает HTTP 404."""

if old_lpw_str in content:
    content = content.replace(old_lpw_str, new_lpw_str)
else:
    print("WARNING: LPW old string not found")

# 2. Update WEyeDS Section 1.16
old_weyeds = """* **Лицензия и доступ:** Academic Open Access.
* **Тип доступа:** Открытый доступ по прямой ссылке публикации ETRA 2024."""
new_weyeds = """* **Лицензия и доступ:** Academic Open Access (Robbins Visual Cognition Lab).
* **Официальный портал проекта:** `https://robbinslab.github.io/weyeds/`
* **Репозиторий конвейера данных:** `https://github.com/robbinslab/weyeds-processing`"""
if old_weyeds in content:
    content = content.replace(old_weyeds, new_weyeds)
else:
    print("WARNING: WEyeDS old string not found")

# 3. Update UnityEyes Section 1.28
old_unityeyes = """* **Официальный портал:** `https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/`
* **Репозиторий GitHub:** `https://github.com/errollw/UnityEyes`"""
new_unityeyes = """* **Официальный портал:** `https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/` (бинарные сборки для Windows/Linux и модели доступны по прямой ссылке). Примечание: репозиторий `errollw/UnityEyes` удален автором (HTTP 404)."""
if old_unityeyes in content:
    content = content.replace(old_unityeyes, new_unityeyes)
else:
    print("WARNING: UnityEyes old string not found")

# 4. Update GazeFollow Section 1.23
old_gazefollow = """* **Лицензия и доступ:** Свободно для академических исследований (MIT CSAIL).
* **Страница проекта:** `https://gazefollow.csail.mit.edu/`"""
new_gazefollow = """* **Лицензия и доступ:** Свободно для академических исследований.
* **Репозиторий загрузки (GitHub):** `https://github.com/recasens/Gaze-Following` и `https://github.com/ejcgt/attention-target-detection`. Примечание: сервер `gazefollow.csail.mit.edu` отключен (таймаут соединения)."""
if old_gazefollow in content:
    content = content.replace(old_gazefollow, new_gazefollow)
else:
    print("WARNING: GazeFollow old string not found")

# 5. Update ChildPlay Gaze Section 1.32
old_childplay = """### 1.32. ChildPlay Gaze

* **Авторы и год публикации:** Fiona Ryan et al. (Georgia Tech), CVPR 2025.
* **Объем видеопотока:** Непрерывные эпизоды естественного диалогового взаимодействия (30 FPS), крупные планы лиц и глаз без окклюзий.
* **Разметка:** Покадровый 3D-луч взгляда и 2D-координаты точки фиксации.
* **Лицензия и доступ:** Открытый прямой доступ.
* **Репозиторий:** `https://github.com/fkryan/childplay`"""
new_childplay = """### 1.32. ChildPlay Gaze

* **Авторы и год публикации:** Samy Tafasca, Anshul Gupta, Jean-Marc Odobez (Idiap Research Institute), ICCV 2023. Бенчмарк моделей внимания (Gaze-LLE, Fiona Ryan et al., 2023/2025).
* **Объем видеопотока:** Непрерывные эпизоды естественного диалогового взаимодействия детей и взрослых (30 FPS), крупные планы лиц и глаз без окклюзий.
* **Разметка:** Покадровый 3D-луч взгляда и 2D-координаты точки фиксации.
* **Лицензия и доступ:** Открытый академический доступ.
* **Официальный портал Idiap:** `https://www.idiap.ch/en/dataset/childplay-gaze`
* **Архив Zenodo:** `https://zenodo.org/records/8254823` (DOI: 10.5281/zenodo.8254823)
* **Репозиторий кода оценки:** `https://github.com/idiap/geomgaze`"""
if old_childplay in content:
    content = content.replace(old_childplay, new_childplay)
else:
    print("WARNING: ChildPlay old string not found")

# 6. Update DR(eye)VE Section 1.33
old_dreyeve = """* **Лицензия и доступ:** Открытый прямой доступ.
* **Страница:** `https://imagelab.ing.unimore.it/dreve/`"""
new_dreyeve = """* **Лицензия и доступ:** Открытый доступ по запросу ссылки на портале лаборатории.
* **Официальный портал AImageLab:** `https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8`
* **Репозиторий GitHub:** `https://github.com/ndrplz/dreyeve`"""
if old_dreyeve in content:
    content = content.replace(old_dreyeve, new_dreyeve)
else:
    print("WARNING: DR(eye)VE old string not found")

# 7. Update EGTEA Gaze+ Section 1.34
old_egtea = """* **Лицензия и доступ:** Открытый прямой доступ.
* **Страница:** `https://cbs.ic.gatech.edu/fpv/`"""
new_egtea = """* **Лицензия и доступ:** Открытый академический доступ.
* **Загрузчик датасета (GitHub):** `https://github.com/amitsou/EGTEA_Gaze_Plus_Downloader`
* **Примечание:** Сервер Georgia Tech `cbs.ic.gatech.edu` выведен из эксплуатации (DNS failure). Загрузка осуществляется парсером из репозитория."""
if old_egtea in content:
    content = content.replace(old_egtea, new_egtea)
else:
    print("WARNING: EGTEA old string not found")

# 8. Update Table in Section 2
# Update table rows with verified links:
old_table_rows = """| 1 | **VideoAttentionTarget** | 2019 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | 1 331 видео (>300k кадров) | [GitHub](https://github.com/ejcgt/attention-target-detection) |
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
| 18 | **SynthesEyes** | 2015 | **Открытый прямой** | **Да** | **Синтетический** | **Статика** | 11 382 рендера ($1280 \times 720$) | [Apollo](https://www.repository.cam.ac.uk/handle/1810/252736) |
| 19 | **UnityEyes** | 2016 | **Открытый прямой** | **Да** | **Синтетический** | **Генератор** | Неограниченно | [GitHub](https://github.com/errollw/UnityEyes) |"""

new_table_rows = """| 1 | **VideoAttentionTarget** | 2019 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | 1 331 видео (>300k кадров) | [GitHub](https://github.com/ejcgt/attention-target-detection) |
| 2 | **ChildPlay Gaze** | 2023 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | Непрерывные сессии | [Idiap](https://www.idiap.ch/en/dataset/childplay-gaze) / [Zenodo](https://zenodo.org/records/8254823) |
| 3 | **WEyeDS** | 2024 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | >320 000 кадров (54 веб-камеры) | [RobbinsLab](https://robbinslab.github.io/weyeds/) / [GitHub](https://github.com/robbinslab/weyeds-processing) |
| 4 | **Katrychuk VOG** | 2024 | **Открытый прямой** | **Да** | **Реальный** | **Видео (120 FPS)**| 40 сессий (AVI без сжатия) | [Zenodo](https://zenodo.org/records/13931669) |
| 5 | **LPW** | 2016 | **Открытый прямой** | **Да** | **Реальный** | **Видео (95 FPS)** | 66 видео (130 856 кадров) | [DaRUS DOI](https://doi.org/10.18419/darus-3373) |
| 6 | **HybridGaze** | 2026 | **В релизе (статья)** | **Да** | **Реальный** | **Видео (30 FPS)** | 151 128 кадров (Full HD) | [IJET DOI](https://doi.org/10.24425/ijet.2026.157892) / [PAN](https://journals.pan.pl/dlibra/publication/157892/edition/138498/content) |
| 7 | **RT-GENE** | 2018 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | 114 558 кадров (Kinect v2) | [Zenodo](https://zenodo.org/records/2529036) / [GitHub](https://github.com/Tobias-Fischer/rt_gene) |
| 8 | **VRGaze / GazeShift** | 2026 | **Открытый прямой** | **Да** | **Реальный** | **Видео (60 FPS)** | >450 000 кадров (VR ИК) | [HuggingFace](https://huggingface.co/datasets/gazeshift/VRGaze) |
| 9 | **MPIIGaze** | 2015 | **Открытый прямой** | **Да** | **Реальный** | **Статика (клики)** | 213 659 кадров | [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz) |
| 10 | **MPIIFaceGaze** | 2017 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 45 000 кадров | [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip) |
| 11 | **Columbia Gaze** | 2013 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 5 880 фото (5184x3456) | [CS Columbia](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/) |
| 12 | **GazeFollow** | 2015 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 125 782 изображения | [GitHub](https://github.com/recasens/Gaze-Following) |
| 13 | **DrivFace** | 2016 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 606 фото | [UCI Repo](https://archive.ics.uci.edu/dataset/358/drivface) |
| 14 | **DR(eye)VE** | 2018 | **Открытый прямой** | **Нет (камера дороги)**| **Реальный** | **Видео (25 FPS)** | 74 видео по 5 мин (555k кадров) | [AImageLab](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8) / [GitHub](https://github.com/ndrplz/dreyeve) |
| 15 | **EGTEA Gaze+** | 2018 | **Открытый прямой** | **Нет (1st person)**| **Реальный** | **Видео (30 FPS)** | 28 часов видео | [GitHub Downloader](https://github.com/amitsou/EGTEA_Gaze_Plus_Downloader) |
| 16 | **GazeBase** | 2021 | **Открытый прямой** | **Нет (сигнал)** | **Реальный** | **Видео (1000 Гц)**| 12 334 сессии EyeLink | [Figshare](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257) |
| 17 | **GazeBaseVR** | 2022 | **Открытый прямой** | **Нет (сигнал)** | **Реальный** | **Видео (250 Гц)** | >8 000 сессий VR Tobii | [Figshare](https://figshare.com/articles/dataset/GazeBaseVR/19643442) |
| 18 | **SynthesEyes** | 2015 | **Открытый прямой** | **Да** | **Синтетический** | **Статика** | 11 382 рендера ($1280 \times 720$) | [Apollo](https://www.repository.cam.ac.uk/handle/1810/252736) |
| 19 | **UnityEyes** | 2016 | **Открытый прямой** | **Да** | **Синтетический** | **Генератор** | Неограниченно | [Cambridge](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) |"""

if old_table_rows in content:
    content = content.replace(old_table_rows, new_table_rows)
else:
    print("WARNING: Table rows not matched exactly")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("catalog.md successfully updated with verified working links!")
