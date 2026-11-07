with open('docs/datasets/catalog.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_open_rows = [
    "| 1 | **VideoAttentionTarget** | 2019 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | 1 331 видео (>300k кадров) | [GitHub](https://github.com/ejcgt/attention-target-detection) |\n",
    "| 2 | **ChildPlay Gaze** | 2023 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | Непрерывные сессии | [Idiap](https://www.idiap.ch/en/dataset/childplay-gaze) / [Zenodo](https://zenodo.org/records/8254823) |\n",
    "| 3 | **WEyeDS** | 2024 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | >320 000 кадров (54 веб-камеры) | [RobbinsLab](https://robbinslab.github.io/weyeds/) / [GitHub](https://github.com/robbinslab/weyeds-processing) |\n",
    "| 4 | **Katrychuk VOG** | 2024 | **Открытый прямой** | **Да** | **Реальный** | **Видео (120 FPS)**| 40 сессий (AVI без сжатия) | [Zenodo](https://zenodo.org/records/13931669) |\n",
    "| 5 | **LPW** | 2016 | **Открытый прямой** | **Да** | **Реальный** | **Видео (95 FPS)** | 66 видео (130 856 кадров) | [DaRUS DOI](https://doi.org/10.18419/darus-3373) |\n",
    "| 6 | **HybridGaze** | 2026 | **В релизе (статья)** | **Да** | **Реальный** | **Видео (30 FPS)** | 151 128 кадров (Full HD) | [IJET DOI](https://doi.org/10.24425/ijet.2026.157892) / [PAN](https://journals.pan.pl/dlibra/publication/157892/edition/138498/content) |\n",
    "| 7 | **RT-GENE** | 2018 | **Открытый прямой** | **Да** | **Реальный** | **Видео (30 FPS)** | 114 558 кадров (Kinect v2) | [Zenodo](https://zenodo.org/records/2529036) / [GitHub](https://github.com/Tobias-Fischer/rt_gene) |\n",
    "| 8 | **VRGaze / GazeShift** | 2026 | **Открытый прямой** | **Да** | **Реальный** | **Видео (60 FPS)** | >450 000 кадров (VR ИК) | [HuggingFace](https://huggingface.co/datasets/gazeshift/VRGaze) |\n",
    "| 9 | **MPIIGaze** | 2015 | **Открытый прямой** | **Да** | **Реальный** | **Статика (клики)** | 213 659 кадров | [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz) |\n",
    "| 10 | **MPIIFaceGaze** | 2017 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 45 000 кадров | [MPI-INF](https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip) |\n",
    "| 11 | **Columbia Gaze** | 2013 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 5 880 фото (5184x3456) | [CS Columbia](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/) |\n",
    "| 12 | **GazeFollow** | 2015 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 125 782 изображения | [GitHub](https://github.com/recasens/Gaze-Following) |\n",
    "| 13 | **DrivFace** | 2016 | **Открытый прямой** | **Да** | **Реальный** | **Статика** | 606 фото | [UCI Repo](https://archive.ics.uci.edu/dataset/358/drivface) |\n",
    "| 14 | **DR(eye)VE** | 2018 | **Открытый прямой** | **Нет (камера дороги)**| **Реальный** | **Видео (25 FPS)** | 74 видео по 5 мин (555k кадров) | [AImageLab](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8) / [GitHub](https://github.com/ndrplz/dreyeve) |\n",
    "| 15 | **EGTEA Gaze+** | 2018 | **Открытый прямой** | **Нет (1st person)**| **Реальный** | **Видео (30 FPS)** | 28 часов видео | [GitHub Downloader](https://github.com/amitsou/EGTEA_Gaze_Plus_Downloader) |\n",
    "| 16 | **GazeBase** | 2021 | **Открытый прямой** | **Нет (сигнал)** | **Реальный** | **Видео (1000 Гц)**| 12 334 сессии EyeLink | [Figshare](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257) |\n",
    "| 17 | **GazeBaseVR** | 2022 | **Открытый прямой** | **Нет (сигнал)** | **Реальный** | **Видео (250 Гц)** | >8 000 сессий VR Tobii | [Figshare](https://figshare.com/articles/dataset/GazeBaseVR/19643442) |\n",
    "| 18 | **SynthesEyes** | 2015 | **Открытый прямой** | **Да** | **Синтетический** | **Статика** | 11 382 рендера ($1280 \times 720$) | [Apollo](https://www.repository.cam.ac.uk/handle/1810/252736) |\n",
    "| 19 | **UnityEyes** | 2016 | **Открытый прямой** | **Да** | **Синтетический** | **Генератор** | Неограниченно | [Cambridge](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) |\n"
]

# Find range from line containing '| 1 |' to '| 19 |'
start_idx = None
end_idx = None
for i, l in enumerate(lines):
    if '| 1 |' in l and '**VideoAttentionTarget**' in l:
        start_idx = i
    if '| 19 |' in l and '**UnityEyes**' in l:
        end_idx = i

print(f"Replacing rows {start_idx} to {end_idx}")
lines[start_idx:end_idx+1] = new_open_rows

with open('docs/datasets/catalog.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Table rows successfully replaced!")
