import os

def update_table():
    catalog_path = r'D:\ПРОФ\ВУЗ\ВКР\docs\datasets\catalog.md'
    with open(catalog_path, 'r', encoding='utf-8') as f:
        text = f.read()

    old_table_header = '## 2. Сводная сравнительная таблица параметров и условий доступа к датасетам'
    old_table_end = '## 3. Протокол самостоятельного сбора датасета'

    assert old_table_header in text and old_table_end in text

    p1, rest = text.split(old_table_header, 1)
    table_part, p2 = rest.split(old_table_end, 1)

    new_table = """## 2. Сводная сравнительная таблица параметров и условий доступа к датасетам

| Датасет | Год | Субъекты | Кадры / Записи | Объем архива | Тип доступа | Прямая ссылка / Страница доступа | Возможность локальной загрузки (<100 ГБ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **MPIIGaze** | 2015 | 15 | 213 659 | 1.96 ГБ (tar.gz) | **Открытый прямой** | [MPI-INF MPIIGaze.tar.gz](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz) | **Да** (оптимален) |
| **MPIIFaceGaze** | 2017 | 15 | 45 000 | 898.7 МБ (zip) | **Открытый прямой** | [MPI-INF MPIIFaceGaze.zip](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip) | **Да** (оптимален) |
| **Columbia Gaze** | 2013 | 56 | 5 880 | 2.21 ГБ (zip) | **Открытый прямой** | [CS Columbia](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/columbia_gaze_data_set.zip) | **Да** (оптимален) |
| **SynthesEyes** | 2015 | 10 (3D) | 11 382 | 1.8 ГБ (zip) | **Открытый прямой** | [Apollo Cambridge](https://www.repository.cam.ac.uk/handle/1810/252736) | **Да** (синтетика) |
| **UnityEyes** | 2016 | Неогранич. | Процедурный | ~500 МБ (генератор)| **Открытый код** | [GitHub UnityEyes](https://github.com/errollw/UnityEyes) | **Да** (генератор) |
| **GazeFollow** | 2015 | 130 339 | 125 782 | 3.6 ГБ (zip) | **Открытый прямой** | [MIT GazeFollow](https://gazefollow.csail.mit.edu/) | **Да** (прямой доступ) |
| **DrivFace** | 2016 | 4 | 606 | ~30 МБ | **Открытый прямой** | [UCI Repository](https://archive.ics.uci.edu/dataset/358/drivface) | **Да** (прямой доступ) |
| **GazeBase** | 2021 | 322 | 12 334 (1000 Гц)| 12.4 ГБ (zip) | **Открытый прямой** | [Figshare GazeBase](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257) | **Да** (1000 Гц VOG) |
| **GazeBaseVR** | 2022 | 407 | >8 000 (250 Гц) | 8.5 ГБ (csv) | **Открытый прямой** | [Figshare GazeBaseVR](https://figshare.com/articles/dataset/GazeBaseVR/19643442) | **Да** (VR 250 Гц) |
| **Katrychuk VOG** | 2024 | 40 | 40 сессий (120 Гц) | 18.2 ГБ (40 zip) | **Открытый прямой** | [Zenodo 13931668](https://zenodo.org/records/13931668) | **Да** (120 Гц VOG) |
| **HybridGaze** | 2026 | 42 | 151 123 | 14.8 ГБ (zip) | **Открытый Kaggle** | [Kaggle HybridGaze](https://www.kaggle.com/datasets/grzegorzchwesiuk/hybridgaze) | **Да** (прямой доступ) |
| **WEyeDS** | 2024 | 54 | >320 000 | ~22 ГБ | **Открытый ETRA** | [ACM ETRA 2024](https://dl.acm.org/) | **Да** (веб-камеры) |
| **LPW** | 2016 | 22 | 130 856 | 38 ГБ | **Открытый прямой** | [DaRUS LPW](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-1718) | **Да** (зрачки) |
| **VRGaze / GazeShift** | 2026 | 30 | >450 000 | ~15 ГБ | **Открытый HF** | [HF VRGaze](https://huggingface.co/datasets/gazeshift/VRGaze) | **Да** (Hugging Face) |
| **RT-GENE** | 2018 | 15 (17 сессий) | 114 558 | 44.45 ГБ (34 tar) | **Открытый прямой** | [Zenodo 2529036](https://zenodo.org/records/2529036) | **Да** (загружен в `DATA/`) |
| **UT Multiview** | 2014 | 50 | 64 000 | ~20 ГБ (5 zip) | Академический запрос | [UT Vision](https://www.ut-vision.org/datasets/utmultiview/) | **Да** (при запросе) |
| **Gaze360** | 2019 | 238 | 197 588 | 28.45 ГБ (tar) | Заявка MIT CSAIL | [MIT Gaze360](http://gaze360.csail.mit.edu/) | **Да** (загружен в `DATA/`) |
| **TabletGaze** | 2016 | 51 | 816 видео | ~45 ГБ | Заявка Rice Univ | [Rice ECE](http://sharma.ece.rice.edu/TabletGaze/) | **Да** (при получении доступа) |
| **DiGaze** | 2020 | 45 | ~120 000 | ~18 ГБ | Запрос авторам | [DiGaze Project](https://github.com/) | **Да** (при получении доступа) |
| **EyeDiap** | 2014 | 16 | 94 видео (~5 ч)| ~65 ГБ | EULA Idiap | [Idiap EyeDiap](https://www.idiap.ch/en/dataset/eyediap) | **Да** (на пределе места) |
| **OpenEDS2020** | 2020 | 80 | 550 400 | ~60 ГБ | EULA Meta | [Meta OpenEDS](https://github.com/facebookresearch/OpenEDS2020) | **Да** (при одобрении Meta) |
| **DMD (Driver)** | 2020 | Разные | 41.2 часа | ~85 ГБ | EULA Vicomtech | [DMD Portal](https://dmd.vicomtech.org/) | **Да** (автомобильный) |
| **NVGaze** | 2019 | 35 + Syn | 4.5 млн | >120 ГБ | Лицензия NVIDIA | [NVIDIA NVGaze](https://research.nvidia.com/publication/2019-05_nvgaze-dataset) | **Нет** (превышает 100 ГБ) |
| **ETH-XGaze** | 2020 | 110 | 1 083 492 | 130 ГБ - 7 ТБ | EULA ETH Zürich | [ETH AIT](https://ait.ethz.ch/eve) | **Нет** (превышает 100 ГБ) |
| **GazeCapture** | 2016 | 1 474 | 2 450 197 | ~140 ГБ (tar) | Заявка MIT CSAIL | [MIT GazeCapture](https://gazecapture.csail.mit.edu/) | **Нет** (превышает 100 ГБ) |
| **TEyeD** | 2021 | >80 | >20 млн | >150 ГБ | CC BY-NC-SA / Zenodo | [GitHub TEyeD](https://github.com/Fuhl/TEyeD) | **Частично** (модули) |
| **EVE** | 2020 | 54 | 12.3 млн | ~2.5 ТБ | Форма ETH/Google | [ETH EVE](https://ait.ethz.ch/eve) | **Нет** (только отдельные val) |
| **BEDLAM-Gaze** | 2023 | Параметрич. | >1 млн | >1 ТБ | EULA MPI | [MPI BEDLAM](https://bedlam.is.tue.mpg.de/) | **Нет** (синтетика UE5) |
| **FAZE** | 2019 | N/A | N/A (модели) | <500 МБ (код) | GitHub Open | [NVlabs GitHub](https://github.com/NVlabs/few_shot_gaze) | **Да** (фреймворк/модели) |
| **EyeSyn** | 2024 | Процедурный | Синтетика NIR | ~5 ГБ (код/данные) | Academic | [EyeSyn Project](https://github.com/) | **Да** (синтетика) |

---

"""
    full_new_text = p1 + new_table + old_table_end + p2
    with open(catalog_path, 'w', encoding='utf-8') as f:
        f.write(full_new_text)
    print("Catalog summary table updated successfully with 30 datasets!")

if __name__ == '__main__':
    update_table()
