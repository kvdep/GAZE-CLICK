import re

file_path = "docs/datasets/catalog.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Section 1.14 (Katrychuk VOG)
old_katrychuk_sec = """* **Прямой URL репозитория:** `https://zenodo.org/records/13931668` (DOI: 10.5281/zenodo.13931668)."""
new_katrychuk_sec = """* **Прямой URL репозитория:** `https://zenodo.org/records/13852050` (DOI: 10.5281/zenodo.13852050).
* **Содержимое репозитория:** 40 архивов субъектов (`Subject01.zip` – `Subject40.zip`), исходные скрипты `src.zip`, коллаж превью `data_preview_collage.png`. Примечание: идентификатор 13931669 относился к нерелевантной публикации по лингвистике."""

if old_katrychuk_sec in content:
    content = content.replace(old_katrychuk_sec, new_katrychuk_sec)
else:
    print("WARNING: Katrychuk Section not matched")

# 2. Update Section 1.16 (WEyeDS)
old_weyeds_sec = """* **Лицензия и доступ:** Academic Open Access (Robbins Visual Cognition Lab).
* **Официальный портал проекта:** `https://robbinslab.github.io/weyeds/`
* **Репозиторий конвейера данных:** `https://github.com/robbinslab/weyeds-processing`"""

new_weyeds_sec = """* **Лицензия и доступ:** Academic Non-Commercial (Robbins Visual Cognition Lab).
* **Тип доступа:** **По запросу (Qualtrics)**. Прямая ссылка на открытое скачивание отсутствует; требуется заполнение институциональной формы: `https://richmond.ca1.qualtrics.com/jfe/form/SV_3pEzZBWmLms2jH0`. Контакт лаборатории: `robbinslab@richmond.edu`.
* **Официальный портал проекта:** `https://robbinslab.github.io/weyeds/`
* **Репозиторий конвейера данных:** `https://github.com/robbinslab/weyeds-processing`"""

if old_weyeds_sec in content:
    content = content.replace(old_weyeds_sec, new_weyeds_sec)
else:
    print("WARNING: WEyeDS Section not matched")

# 3. Update Section 1.21 (LPW)
old_lpw_sec = """* **Прямой URL:** Репозиторий DaRUS (University of Stuttgart): `https://doi.org/10.18419/darus-3373` (`https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-3373`). Примечание: устаревший идентификатор `darus-1718` возвращает HTTP 404."""

new_lpw_sec = """* **Прямой URL для скачивания:** Прямой архив института Макса Планка: `https://datasets.d2.mpi-inf.mpg.de/tonsen/LPW.zip` (объем: **2 576 433 029 байт**, ~2.40 ГБ, HTTP 200).
* **Постоянный репозиторий:** DaRUS (University of Stuttgart): `https://doi.org/10.18419/darus-3373` (`https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-3373`). Примечание: устаревший идентификатор `darus-1718` возвращает HTTP 404."""

if old_lpw_sec in content:
    content = content.replace(old_lpw_sec, new_lpw_sec)
else:
    print("WARNING: LPW Section not matched")

# 4. Update Section 1.31 (VideoAttentionTarget)
old_vat_sec = """### 1.31. VideoAttentionTarget

* **Авторы и год публикации:** Eunji Chong, Dawei Du, Kyoungmin Lee, James M. Rehg (Georgia Tech), CVPR 2019.
* **Объем видеопотока:** 1 331 непрерывная видеопоследовательность (более 300 000 кадров), частота 30 FPS.
* **Разметка:** Покадровая непрерывная траектория взгляда каждого человека в кадре (3D-векторы и 2D-координаты точки фиксации внимания).
* **Лицензия и доступ:** Открытый прямой доступ (архивы на Zenodo и AWS).
* **Репозиторий:** `https://github.com/ejcgt/attention-target-detection`"""

new_vat_sec = """### 1.31. VideoAttentionTarget (Gaze Following in Video)

* **Авторы и год публикации:** Eunji Chong, Yong Jae Lee, James M. Rehg et al. (Georgia Tech), CVPR 2020.
* **Постановка задачи:** Слежение за целью внимания в видео от третьего лица (*Gaze-Following / Attention Target Detection*).
* **Входные данные:** Полный кадр видео сцены $I_t \in \mathbb{R}^{H \times W \times 3}$ и ограничивающий прямоугольник головы наблюдаемого человека $B_t = (x_1, y_1, x_2, y_2)$.
* **Выходные данные:** Двумерная тепловая карта внимания (Attention Heatmap) $H_t \in [0, 1]^{H' \times W'}$ на плоскости кадра сцены и вероятность того, что цель находится внутри кадра ($p_{in} \in [0, 1]$).
* **Разметка (Ground Truth):** 2D-координаты объекта внимания в кадре сцены $(x_{target}, y_{target})$.
* **Специфика в контексте задачи ВКР:** **Не содержит углов поворота глазных яблок (Pitch, Yaw) и 3D-векторов взгляда.** Глаза людей в кадре занимают незначительное число пикселей. Датасет решает задачу внешнего поиска объекта внимания в сцене, а не геометрической оценки вектора взгляда по лицу, и **непригоден для позиционирования экранного курсора**.
* **Объем данных:** 1 331 видеопоследовательность (более 300 000 кадров), 30 FPS.
* **Лицензия и доступ:** Открытый академический доступ.
* **Репозиторий кода и скриптов загрузки:** `https://github.com/ejcgt/attention-target-detection`"""

if old_vat_sec in content:
    content = content.replace(old_vat_sec, new_vat_sec)
else:
    print("WARNING: VideoAttentionTarget Section not matched")

# 5. Update Section 1.32 (ChildPlay Gaze)
old_cp_sec = """### 1.32. ChildPlay Gaze

* **Авторы и год публикации:** Samy Tafasca, Anshul Gupta, Jean-Marc Odobez (Idiap Research Institute), ICCV 2023. Бенчмарк моделей внимания (Gaze-LLE, Fiona Ryan et al., 2023/2025).
* **Объем видеопотока:** Непрерывные эпизоды естественного диалогового взаимодействия детей и взрослых (30 FPS), крупные планы лиц и глаз без окклюзий.
* **Разметка:** Покадровый 3D-луч взгляда и 2D-координаты точки фиксации.
* **Лицензия и доступ:** Открытый академический доступ.
* **Официальный портал Idiap:** `https://www.idiap.ch/en/dataset/childplay-gaze`
* **Архив Zenodo:** `https://zenodo.org/records/8254823` (DOI: 10.5281/zenodo.8254823)
* **Репозиторий кода оценки:** `https://github.com/idiap/geomgaze`"""

new_cp_sec = """### 1.32. ChildPlay Gaze (Gaze Target Benchmark)

* **Авторы и год публикации:** Samy Tafasca, Anshul Gupta, Jean-Marc Odobez (Idiap Research Institute), ICCV 2023.
* **Постановка задачи:** Слежение за целью внимания детей от третьего лица во время игровых сессий (*Gaze Following in Video*).
* **Выходные данные:** Тепловая карта внимания на сцене и 2D-координаты объекта внимания в пространстве комнаты.
* **Специфика в контексте задачи ВКР:** Аналогично VideoAttentionTarget и GazeFollow, датасет решает задачу семантической локализации объекта внимания в сцене, а не картирования взгляда пользователя веб-камеры в экранные координаты курсора.
* **Лицензия и доступ:** Открытый академический доступ.
* **Официальный портал Idiap:** `https://www.idiap.ch/en/dataset/childplay-gaze`
* **Архив Zenodo:** `https://zenodo.org/records/8254823` (DOI: 10.5281/zenodo.8254823)
* **Репозиторий кода:** `https://github.com/idiap/geomgaze`"""

if old_cp_sec in content:
    content = content.replace(old_cp_sec, new_cp_sec)
else:
    print("WARNING: ChildPlay Section not matched")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("catalog.md content sections successfully updated!")
