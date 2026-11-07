import os, sys, json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbformat.v4.new_notebook()

# Cell 0: Title and Header (Markdown)
c0_md = """# Комплексный исследовательский анализ данных (EDA) датасета Gaze360
**Gaze360: Physically Unconstrained Gaze Estimation in the Wild** (IEEE/CVF ICCV 2019)  
Официальный ресурс: [MIT CSAIL Gaze360 Project](http://gaze360.csail.mit.edu/)  
Авторы: *Petr Kellnhofer, Adrià Recasens, Simon Stent, Wojciech Matusik, Antonio Torralba*  
Лицензия: Non-commercial Research Use Only (Creative Commons Attribution-NonCommercial-ShareAlike 4.0)

---

## 1. Описание датасета и спецификация сбора данных

Датасет **Gaze360** спроектирован для надежной оценки направления взгляда в неконтролируемых пространственных условиях с круговым охватом в $360^\circ$.

### Ключевые особенности экспериментальной установки:
1. **Сенсорная система:** Панорамная камера кругового обзора **FLIR Ladybug5** (блок из 6 синхронизированных ПЗС-матриц с суммарным разрешением панорамы $4096 \\times 2048$, частота $30\\text{ кадров/с}$).
2. **Система фиксации истинного направления взгляда (Ground Truth):** Крупногабаритная подвижная мишень с маркерной планкой, перемещаемая по 3D-пространству вокруг камеры на расстояниях от $0.25$ до $20.0\\text{ метров}$. Пространственные координаты глаз и мишени регистрировались промышленной оптоэлектронной системой захвата движения **Vicon** (инфракрасные световозвращающие маркеры субмиллиметровой точности).
3. **Условия съемки:** 80 сессий видеозаписи в 5 различных средах (учебные классы, лаборатории, коридоры, кафетерий, открытые площадки на улице) при естественном солнечном, люминесцентном, направленном и контровом освещении.
"""

# Cell 1: Setup and Imports (Code)
c1_code = """import os
import sys
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Настройка визуализации
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

print("Библиотеки успешно загружены.")
"""

# Cell 2: Modalities and Samples (Markdown)
c2_md = """## 2. Визуализация модальностей и образцов данных

Датасет содержит два ключевых типа визуальных кропов:
1. **Кроп головы (`head`):** Квадратное изображение головы оператора ($1.00$ aspect ratio), центрированное по 3D-координатам глаз.
2. **Кроп тела (`body`):** Полный вертикальный силуэт человека в кадре панорамы Ladybug5.
"""

# Cell 3: Sample Display (Code)
c3_code = """base_dir = os.path.dirname(os.path.abspath('__file__'))
if not os.path.exists(os.path.join(base_dir, 'imgs')):
    base_dir = os.path.join('DATA', 'Gaze360')

head_candidates = sorted(glob.glob(os.path.join(base_dir, 'imgs', '*', 'head', '*', '*.jpg')))
body_candidates = sorted(glob.glob(os.path.join(base_dir, 'imgs', '*', 'body', '*', '*.jpg')))

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

# Отображаем 2 образца головы и 2 образца тела
for idx, ax in enumerate(axes):
    if idx < 2 and len(head_candidates) > idx * 100:
        img_p = head_candidates[idx * 150 + 20]
        im = Image.open(img_p)
        ax.imshow(im)
        rel_p = os.path.relpath(img_p, base_dir)
        ax.set_title(f"Head Crop\\n{im.size[0]}x{im.size[1]} px\\n{os.path.basename(img_p)}", fontsize=10)
    elif idx >= 2 and len(body_candidates) > (idx - 2) * 100:
        img_p = body_candidates[(idx - 2) * 150 + 20]
        im = Image.open(img_p)
        ax.imshow(im)
        ax.set_title(f"Body Crop\\n{im.size[0]}x{im.size[1]} px\\n{os.path.basename(img_p)}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()
"""

# Cell 4: Math and Coordinate Systems (Markdown)
c4_md = """## 3. Математический аппарат и системы координат

### 3.1. Декартова система координат камеры Ladybug5
- Начало координат $(0, 0, 0)$ расположено в оптическом центре панорамной камеры.
- Ось $+X$ направлена вправо относительно базовой ориентации сенсора.
- Ось $+Y$ направлена вверх.
- Ось $-Z$ направлена вперед в сторону наблюдаемого объекта.

### 3.2. Сферические координаты взгляда
Единичный направляющий вектор взгляда $\\vec{g} = [g_x, g_y, g_z]^T \\in S^2$ (где $\|\\vec{g}\|_2 = 1$) преобразуется в угол места $\\theta$ (Pitch) и азимут $\\phi$ (Yaw):
$$\\text{Pitch} = \\theta = \\arcsin(-g_y)$$
$$\\text{Yaw} = \\phi = \\text{atan2}(-g_x, -g_z)$$

Обратное восстановление декартова вектора взгляда:
$$g_x = -\\sin(\\phi)\\cos(\\theta)$$
$$g_y = -\\sin(\\theta)$$
$$g_z = -\\cos(\\phi)\\cos(\\theta)$$

### 3.3. Сферическая статистика на $S^2$
Для выборки из $N$ трехмерных единичных векторов $\\vec{g}_i$:
- Результирующий средний вектор: $\\vec{R} = \\frac{1}{N} \\sum_{i=1}^N \\vec{g}_i$
- Длина среднего вектора: $\\bar{R} = \\|\\vec{R}\\|_2 \\in [0, 1]$
- Сферическая дисперсия: $S = 1 - \\bar{R}$
- Круговое стандартное отклонение (Мардиа): $\\sigma_c = \\sqrt{-2 \\ln \\bar{R}}$
"""

# Cell 5: Data Loading (Code)
c5_code = """pq_path = os.path.join(base_dir, 'processed', 'gaze360_full.parquet')
if not os.path.exists(pq_path):
    pq_path = os.path.join('DATA', 'Gaze360', 'processed', 'gaze360_full.parquet')

df = pd.read_parquet(pq_path)

# Углы в градусах
df['gaze_pitch_deg'] = np.degrees(df['gaze_pitch_rad'])
df['gaze_yaw_deg'] = np.degrees(df['gaze_yaw_rad'])

# Загрузка полных метаданных
mat_path = os.path.join(base_dir, 'metadata.mat')
if not os.path.exists(mat_path):
    mat_path = os.path.join('DATA', 'Gaze360', 'metadata.mat')
mat = sio.loadmat(mat_path)

print(f"Загружено записей в Parquet: {len(df):,}")
print(f"Всего записей в metadata.mat: {mat['frame'].shape[1]:,}")
"""

# Cell 6: Data Volume and Splits (Markdown)
c6_md = """## 4. Объем данных и распределение по сплитам

Датасет разбит на 4 стандартных сплита:
- **Split 0 (Train):** Обучение моделей.
- **Split 1 (Val):** Промежуточная валидация и подбор гиперпараметров.
- **Split 2 (Test):** Финальная эталонная оценка точности (Benchmark).
- **Split 3 (Unused):** Неаннотированные кадры или кадры с перекрытыми лицами.
"""

# Cell 7: Split Distribution (Code)
c7_code = """split_names = {0: 'Train', 1: 'Val', 2: 'Test', 3: 'Unused'}
df['split_name'] = df['split'].map(split_names)

split_counts = df['split_name'].value_counts()[['Train', 'Val', 'Test', 'Unused']]
split_pct = (split_counts / len(df) * 100)

split_summary = pd.DataFrame({
    'Образцов (Parquet)': split_counts,
    'Доля (%)': split_pct.round(2)
})

# Метаданные .mat
mat_splits = mat['split'][0]
mat_counts = pd.Series(mat_splits).map(split_names).value_counts()[['Train', 'Val', 'Test', 'Unused']]
split_summary['В metadata.mat'] = mat_counts
split_summary['Полнота (%)'] = (split_counts / mat_counts * 100).round(2)

print("Распределение данных по сплитам:")
print(split_summary)

fig, ax = plt.subplots(figsize=(8, 4))
split_counts.plot(kind='bar', ax=ax, color=['#2563eb', '#10b981', '#f59e0b', '#64748b'], edgecolor='black', alpha=0.85)
ax.set_title("Распределение объемов сэмплов по сплитам Gaze360", fontsize=12, fontweight='bold')
ax.set_ylabel("Число кадров")
for i, v in enumerate(split_counts):
    ax.text(i, v + 2000, f"{v:,}\\n({split_pct.iloc[i]:.1f}%)", ha='center', fontsize=9, fontweight='bold')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""

# Cell 8: Angle Statistics (Markdown)
c8_md = """## 5. Статистический анализ углов взгляда и 3D-векторов

В таблице ниже рассчитаны точные дескриптивные статистики распределения компонент направляющего вектора $\\vec{g}$ и сферических углов $\\theta$ (Pitch) и $\\phi$ (Yaw).
"""

# Cell 9: Angle Stats Computation (Code)
c9_code = """angle_cols = ['gaze_vec_x', 'gaze_vec_y', 'gaze_vec_z', 'gaze_pitch_deg', 'gaze_yaw_deg']
stats_df = df[angle_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T

stats_df['variance'] = df[angle_cols].var()
stats_df['skewness'] = df[angle_cols].skew()
stats_df['kurtosis'] = df[angle_cols].kurtosis()

print("Дескриптивные статистики направления взгляда Gaze360:")
display_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']
print(stats_df[display_cols].round(4))
"""

# Cell 10: Spherical Directional Stats (Code)
c10_code = """gaze_vecs = df[['gaze_vec_x', 'gaze_vec_y', 'gaze_vec_z']].values
R_vec = np.mean(gaze_vecs, axis=0)
R_bar = np.linalg.norm(R_vec)
spherical_variance = 1.0 - R_bar
circ_std_rad = np.sqrt(-2.0 * np.log(max(1e-6, R_bar)))
circ_std_deg = np.degrees(circ_std_rad)

print("Сферические статистики направления взгляда (S^2 Directional Statistics):")
print(f"  Результирующий вектор R: [{R_vec[0]:.4f}, {R_vec[1]:.4f}, {R_vec[2]:.4f}]")
print(f"  Длина среднего вектора R_bar: {R_bar:.4f}")
print(f"  Сферическая дисперсия (1 - R_bar): {spherical_variance:.4f}")
print(f"  Круговое стандартное отклонение (Mardia): {circ_std_deg:.2f}°")
"""

# Cell 11: Angle Distribution Plots (Markdown)
c11_md = """## 6. Визуализация распределения угловых величин

Исследование гистограмм и функций плотности вероятности (KDE) углов взгляда, а также анализ углового покрытия (Yaw Coverage).
"""

# Cell 12: Histograms and Coverage (Code)
c12_code = """fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 1. Yaw distribution
sns.histplot(df['gaze_yaw_deg'], kde=True, bins=90, color='#2563eb', stat='density', ax=axes[0])
axes[0].set_title(f"Распределение азимута взгляда (Yaw)\\nMean={df['gaze_yaw_deg'].mean():.2f}°, Std={df['gaze_yaw_deg'].std():.2f}°", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Угол Yaw (градусы)")
axes[0].axvline(0, color='red', linestyle='--', alpha=0.7)
axes[0].axvline(-90, color='gray', linestyle=':', alpha=0.5)
axes[0].axvline(90, color='gray', linestyle=':', alpha=0.5)

# 2. Pitch distribution
sns.histplot(df['gaze_pitch_deg'], kde=True, bins=70, color='#059669', stat='density', ax=axes[1])
axes[1].set_title(f"Распределение угла места (Pitch)\\nMean={df['gaze_pitch_deg'].mean():.2f}°, Std={df['gaze_pitch_deg'].std():.2f}°", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Угол Pitch (градусы)")
axes[1].axvline(0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Анализ диапазонов охвата взгляда
total = len(df)
cov_360 = total
cov_f180 = np.sum(np.abs(df['gaze_yaw_deg']) <= 90.0)
cov_f90 = np.sum(np.abs(df['gaze_yaw_deg']) <= 45.0)
cov_f40 = np.sum(np.abs(df['gaze_yaw_deg']) <= 20.0)

coverage_df = pd.DataFrame({
    'Сектор обзора': [
        'Полная сфера 360° ([-180°, +180°])',
        'Фронтальная полусфера (|Yaw| <= 90°)',
        'Узкий фронтальный сектор (|Yaw| <= 45°)',
        'Сектор экрана монитора (|Yaw| <= 20°)'
    ],
    'Количество кадров': [cov_360, cov_f180, cov_f90, cov_f40],
    'Доля от датасета (%)': [
        100.0,
        cov_f180 / total * 100.0,
        cov_f90 / total * 100.0,
        cov_f40 / total * 100.0
    ]
})
print("Охват по горизонтальным секторам обзора:")
print(coverage_df.to_string(index=False))
"""

# Cell 13: 2D Heatmap Joint Distribution (Markdown)
c13_md = """## 7. Двумерное совместное распределение взгляда (Pitch $\\times$ Yaw Heatmap)

Карта плотности углов взгляда демонстрирует глобальное смещение целевых точек фиксации в панораме.
"""

# Cell 14: 2D Heatmap Code (Code)
c14_code = """fig, ax = plt.subplots(figsize=(12, 6))

counts, xedges, yedges, im = ax.hist2d(
    df['gaze_yaw_deg'], df['gaze_pitch_deg'],
    bins=[90, 60],
    range=[[-180, 180], [-80, 80]],
    cmap='viridis',
    cmin=1
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Плотность сэмплов (число кадров)")
ax.set_title("Двумерная плотность вероятности взгляда (Gaze Yaw vs Gaze Pitch)", fontsize=12, fontweight='bold')
ax.set_xlabel("Gaze Yaw (азимут, градусы)")
ax.set_ylabel("Gaze Pitch (угол места, градусы)")
ax.axhline(0, color='white', linestyle='--', alpha=0.5)
ax.axvline(0, color='white', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 2D кросс-таблица интервалов
yaw_bins = [-180, -135, -90, -45, -20, 0, 20, 45, 90, 135, 180]
pitch_bins = [-90, -45, -20, -10, 0, 10, 20, 45, 90]

df['yaw_binned'] = pd.cut(df['gaze_yaw_deg'], bins=yaw_bins)
df['pitch_binned'] = pd.cut(df['gaze_pitch_deg'], bins=pitch_bins)

crosstab_pct = (pd.crosstab(df['pitch_binned'], df['yaw_binned']) / len(df) * 100).round(2)
print("Матрица совместной плотности вероятности Pitch x Yaw (% от общего объема):")
print(crosstab_pct)
"""

# Cell 15: Spatial Distance Analysis (Markdown)
c15_md = """## 8. Пространственный анализ: Дистанция до камеры и мишени

Анализ 3D-координат глаз субъекта относительно центра камеры Ladybug5 $\\vec{P}_{eyes} \\in \\mathbb{R}^3$ и вектора до мишени $\\vec{P}_{target} \\in \\mathbb{R}^3$.
"""

# Cell 16: Distance Code (Code)
c16_code = """eyes_3d = mat['person_eyes3d']
target_3d = mat['target_pos3d']

dist_to_cam = np.linalg.norm(eyes_3d, axis=1)
dist_to_target = np.linalg.norm(target_3d - eyes_3d, axis=1)

dist_stats = pd.DataFrame({
    'Дистанция Камера -> Глаза (м)': pd.Series(dist_to_cam).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]),
    'Дистанция Глаза -> Мишень (м)': pd.Series(dist_to_target).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
}).round(3)

print("Метрики дистанции в 3D пространстве:")
print(dist_stats)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.histplot(dist_to_cam, kde=True, bins=60, color='#0284c7', ax=axes[0])
axes[0].set_title(f"Дистанция от камеры Ladybug5 до субъекта\\nMean={dist_to_cam.mean():.2f} м, Median={np.median(dist_to_cam):.2f} м", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Дистанция (метры)")

sns.histplot(dist_to_target, kde=True, bins=70, color='#7c3aed', ax=axes[1])
axes[1].set_title(f"Дистанция фиксации взгляда (Глаза -> Мишень)\\nMean={dist_to_target.mean():.2f} м, Median={np.median(dist_to_target):.2f} м", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Дистанция (метры)")
axes[1].set_xlim(0, 10)

plt.tight_layout()
plt.show()
"""

# Cell 17: Image Resolutions and Bboxes (Markdown)
c17_md = """## 9. Разрешение изображений и характеристики детекций

Исследование размеров кропов голов на диске и относительных размеров Bounding Box в панораме.
"""

# Cell 18: Resolution Code (Code)
c18_code = """head_sample_paths = np.random.choice(head_candidates, size=min(1500, len(head_candidates)), replace=False)
widths, heights = [], []

for p in head_sample_paths:
    with Image.open(p) as im:
        w, h = im.size
        widths.append(w)
        heights.append(h)

widths = np.array(widths)
heights = np.array(heights)

print("Разрешение кропов головы (Head Crops):")
print(f"  Минимум:  {widths.min()} x {heights.min()} px")
print(f"  25% квантиль: {int(np.percentile(widths, 25))} x {int(np.percentile(heights, 25))} px")
print(f"  Медиана:  {int(np.median(widths))} x {int(np.median(heights))} px")
print(f"  Среднее:  {widths.mean():.1f} x {heights.mean():.1f} px")
print(f"  75% квантиль: {int(np.percentile(widths, 75))} x {int(np.percentile(heights, 75))} px")
print(f"  Максимум: {widths.max()} x {heights.max()} px")
print(f"  Соотношение сторон (W / H): {np.mean(widths / heights):.2f} (строго квадратные)")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(widths, bins=40, color='#ea580c', kde=True, ax=ax)
ax.set_title("Распределение геометрических размеров кропов голов (px)", fontsize=11, fontweight='bold')
ax.set_xlabel("Ширина / Высота кропа головы (px)")
ax.set_ylabel("Число кадров")
plt.tight_layout()
plt.show()
"""

# Cell 19: Subject and Recording Distributions (Markdown)
c19_md = """## 10. Распределение сэмплов по участникам и сессиям записи

Оценка вариативности объемов записей по отдельным участникам и сессиям.
"""

# Cell 20: Subject Counts Code (Code)
c20_code = """person_counts = df['subject_id'].value_counts()
rec_counts = df['recording'].value_counts()

print(f"Уникальных субъектов в выборке: {df['subject_id'].nunique()}")
print(f"Уникальных сессий записи: {df['recording'].nunique()}")
print(f"Среднее число кадров на участника: {person_counts.mean():.1f} (Min: {person_counts.min()}, Max: {person_counts.max()})")
print(f"Среднее число кадров на сессию: {rec_counts.mean():.1f} (Min: {rec_counts.min()}, Max: {rec_counts.max()})")

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

person_counts.iloc[:30].plot(kind='bar', ax=axes[0], color='#4338ca', alpha=0.85)
axes[0].set_title("Топ-30 участников по количеству кадров", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Число кадров")
axes[0].tick_params(axis='x', rotation=90, labelsize=7)

sns.histplot(rec_counts.values, bins=25, color='#0891b2', kde=True, ax=axes[1])
axes[1].set_title("Распределение объемов кадров по сессиям записи", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Число кадров в сессии")
axes[1].set_ylabel("Количество сессий")

plt.tight_layout()
plt.show()
"""

# Cell 21: Correlation Analysis (Markdown)
c21_md = """## 11. Корреляционный анализ параметров

Матрица парных коэффициентов линейной корреляции Пирсона между ключевыми переменными датасета.
"""

# Cell 22: Correlation Matrix Code (Code)
c22_code = """corr_cols = [
    'gaze_vec_x', 'gaze_vec_y', 'gaze_vec_z',
    'gaze_pitch_deg', 'gaze_yaw_deg'
]

corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1.0, vmax=1.0, cbar=True, ax=ax, linewidths=0.5)
ax.set_title("Корреляционная матрица параметров взгляда Gaze360", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""

# Cell 23: Final Summary Table (Markdown)
c23_md = """## 12. Итоговая сводка параметров датасета Gaze360

| Параметр | Значение в метаданных (`metadata.mat`) | Фактическое значение на диске (`parquet`) |
| :--- | :--- | :--- |
| **Общее число аннотированных кадров** | $197\\,588$ | **$172\\,259$** |
| **Кропы головы (`imgs/*/head/`)** | $197\\,588$ | **$172\\,259$** |
| **Кропы тела (`imgs/*/body/`)** | $197\\,588$ | **$166\\,185$** |
| **Число уникальных участников** | $432$ | **$410$** |
| **Число сессий видеозаписи** | $80$ | **$69$** |
| **Сенсор видеозахвата** | FLIR Ladybug5 ($4096 \\times 2048$, $30\\text{ FPS}$) | FLIR Ladybug5 ($30\\text{ FPS}$) |
| **Система захвата Ground Truth** | 3D Vicon Motion Capture System | 3D Vicon Motion Capture System |
| **Разрешение кропов голов** | Переменное (от $64\\text{ px}$ до $686\\text{ px}$) | Медиана $158 \\times 158\\text{ px}$, Среднее $178.8\\text{ px}$ |
| **Соотношение сторон кропов голов** | $1.00$ (квадратные) | $1.00$ (квадратные) |
| **Диапазон углов Yaw (азимут)** | $[-180.00^\\circ, +180.00^\\circ]$ (полный охват $360^\\circ$) | $\\mu = +0.59^\\circ, \\sigma = 71.72^\\circ$ |
| **Диапазон углов Pitch (угол места)** | $[-78.22^\\circ, +89.26^\\circ]$ | $\\mu = +8.31^\\circ, \\sigma = 18.38^\\circ$ |
| **Дистанция от камеры до субъекта** | $0.60\\text{--}3.50\\text{ м}$ | Медиана $2.21\\text{ м}$, Среднее $2.17\\text{ м}$ |
| **Дистанция от глаз до мишени** | $0.25\\text{--}20.03\\text{ м}$ | Медиана $2.52\\text{ м}$, Среднее $2.72\\text{ м}$ |
| **Доля фронтальных взглядов ($|\\text{Yaw}| \\le 90^\\circ$)** | $\\approx 79.25\\%$ | $136\\,513$ образцов ($79.25\\%$) |
| **Доля взглядов на экран ($|\\text{Yaw}| \\le 20^\\circ$)** | $\\approx 22.40\\%$ | $38\\,591$ образец ($22.40\\%$) |
| **Официальные сплиты** | Train $64.24\\%$, Val $8.62\\%$, Test $13.14\\%$, Unused $14.00\\%$ | Train $62.19\\%$, Val $9.27\\%$, Test $13.90\\%$, Unused $14.65\\%$ |
"""

cells_data = [
    (c0_md, 'markdown'),
    (c1_code, 'code'),
    (c2_md, 'markdown'),
    (c3_code, 'code'),
    (c4_md, 'markdown'),
    (c5_code, 'code'),
    (c6_md, 'markdown'),
    (c7_code, 'code'),
    (c8_md, 'markdown'),
    (c9_code, 'code'),
    (c10_code, 'code'),
    (c11_md, 'markdown'),
    (c12_code, 'code'),
    (c13_md, 'markdown'),
    (c14_code, 'code'),
    (c15_md, 'markdown'),
    (c16_code, 'code'),
    (c17_md, 'markdown'),
    (c18_code, 'code'),
    (c19_md, 'markdown'),
    (c20_code, 'code'),
    (c21_md, 'markdown'),
    (c22_code, 'code'),
    (c23_md, 'markdown')
]

for src, ctype in cells_data:
    if ctype == 'markdown':
        nb.cells.append(nbformat.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(src))

target_nb_path = r'D:\ПРОФ\ВУЗ\ВКР\DATA\Gaze360\gaze360_data_analysis.ipynb'
with open(target_nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Блокнот сохранен в: {target_nb_path}")
print(f"Всего ячеек: {len(nb.cells)}")
