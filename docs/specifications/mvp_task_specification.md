# Техническое задание и регламент исследования: Разработка легковесной системы оценки направления взгляда для управления GUI в реальном времени

> **Статус документа:** Обязательный нормативный регламент архитектуры, данных, метрик и функций потерь.  
> **Целевая аудитория:** Агент-разработчик (Executor Agent), исследователь, научный руководитель ВКР.  
> **Дата фиксации:** Октябрь 2026 г.

---

## 1. Строгая математическая и инженерная постановка задачи

### 1.1. Контекст и целевой продукт
Разрабатывается легковесная кросс-платформенная система компьютерного зрения для управления курсором мыши операционной системы в режиме реального времени ($\ge 60\text{ FPS}$) по видеопотоку монокулярной камеры.

### 1.2. Входные и выходные пространства
* **Вход системы в момент времени $t$:**
  $$I_t \in \mathbb{R}^{H \times W \times 3}, \quad I_t \in [0, 255]^{H \times W \times 3}$$
  Кадр видеопотока RGB (разрешение $640 \times 480$ или $1280 \times 720$).

* **Выходные переменные нейросетевой модели:**
  1. Непрерывные сферические углы направления взгляда в нормализованной системе координат:
     $$\mathbf{y}_{gaze} = \begin{bmatrix} \theta \\ \phi \end{bmatrix} \in [-\theta_{max}, \theta_{max}] \times [-\phi_{max}, \phi_{max}] \subset \mathbb{R}^2$$
     где $\theta$ — угол тангажа (pitch, вертикаль), $\phi$ — угол рыскания (yaw, горизонталь). Диапазон: $\theta_{max} = 45^\circ$ ($0.7854\text{ рад}$), $\phi_{max} = 45^\circ$ ($0.7854\text{ рад}$).
  2. Нормализованный единичный 3D-вектор направления взгляда:
     $$\mathbf{g} = \begin{bmatrix} g_x \\ g_y \\ g_z \end{bmatrix} = \begin{bmatrix} -\cos\theta \sin\phi \\ -\sin\theta \\ -\cos\theta \cos\phi \end{bmatrix} \in \mathbb{R}^3, \quad \|\mathbf{g}\|_2 = 1$$
  3. Прямые или спроецированные экранные координаты курсора:
     $$\mathbf{s}_t = \begin{bmatrix} u_s \\ v_s \end{bmatrix} \in [0, W_{screen}] \times [0, H_{screen}] \subset \mathbb{R}^2$$

### 1.3. Сквозной конвейер (End-to-End Pipeline)

```
                       Входной кадр веб-камеры (RGB)
                                     │
                                     ▼
        [1. Face & Landmark Detection (MediaPipe / BlazeFace)]
         ├── Координаты Bounding Box лица
         ├── 468 3D-ключевых точек лица
         └── 6 опорных точек для Perspective-n-Point (EPnP)
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
    [2. Оценка позы головы]              [3. Пространственная нормализация]
     R_head ∈ SO(3), t_head ∈ R^3        Матрица Warp: M = S · R_norm
                                         Кроп лица: I_norm (224 x 224 x 3)
                   │                                   │
                   │                                   ▼
                   │                  [4. Легковесный Backbone (Forward)]
                   │                   MobileOne-s0 / MobileNetV3-Small
                   │                                   │
                   │                                   ▼
                   │                  [5. Dual-Head L2CS (Classification + Exp)]
                   │                   Распределение Softmax по K=90 бинам
                   │                   Вычисление углов: θ_pred, φ_pred
                   │                   Единичный вектор: g_pred ∈ R^3
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     ▼
            [6. Геометрическое проецирование (Ray-Plane Intersection)]
             Луч: P(t) = o_eye + t · (R_head · g_pred)
             Плоскость монитора: n^T (P - P_0) = 0
             Пересечение: P_target = (x_mm, y_mm)
                                     │
                                     ▼
            [7. Калибровочный слой персонализации (Few-Shot)]
             Коррекция угла каппа κ: s_calib = W_calib · φ(P_target)
                                     │
                                     ▼
            [8. Адаптивная временная фильтрация (1-Euro Filter)]
             Динамическая частота среза: f_c = f_c_min + β · |dx/dt|
             Подавление физиологического тремора (микросаккад)
                                     │
                                     ▼
            [9. Инъекция в ОС через Win32 API]
             ctypes.windll.user32.SetCursorPos(u_s, v_s)
```

---

## 2. Данные: единая мульти-датасетная база обучения

Обучение каждой модели проводится **одновременно на объединенной выборке из всех трех доступных датасетов**. Никаких экспериментов на изолированных одиночных датасетах не допускается.

### 2.1. Каталог и пути к данным

| Датасет | Абсолютный / Относительный путь | Формат аннотаций | Число записей | Число субъектов | Характер условий | Наличие экранных координат |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **Columbia Gaze** | `DATA/columbia_gaze/processed/columbia_gaze_full.parquet` | Parquet (таблица) | $5\,880$ | 56 | Лабораторные, сетка 5 поз головы $\times$ 21 угол взгляда | Нет (`has_screen = False`) |
| **MPIIFaceGaze** | `DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet` | Parquet + `.mat` калибровки | $37\,667$ | 15 (`p00`–`p14`) | In-the-wild, ноутбуки, реальные клики мыши | **Есть** (`screen_x_px`, `screen_y_px`) |
| **RT-GENE** | `DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet` | Parquet (таблица) | $114\,558$ | 15 | In-the-wild, Vicon MoCap, дистанции 0.5–2.2 м | Нет (`has_screen = False`) |
| **ИТОГО** | — | — | **$158\,105$** | **86** | Мультидоменная выборка | — |

### 2.2. Стратегия разбиения (Stratified Subject-Disjoint Split)
Строго запрещено случайное перемешивание кадров (`random train_test_split`), так как последовательные кадры одного субъекта приводят к утечке данных (data leakage).

* **Разбиение строго по субъектам (Subject-Disjoint):**
  * **Train Set (75% субъектов):**
    * Columbia: 42 субъекта ($4\,410$ сэмплов).
    * MPIIFaceGaze: 11 субъектов `p00`–`p10` ($27\,812$ сэмплов).
    * RT-GENE: 11 субъектов ($84\,120$ сэмплов).
    * *Всего Train:* $116\,342$ сэмпла.
  * **Val Set (12.5% субъектов):**
    * Columbia: 7 субъектов ($735$ сэмплов).
    * MPIIFaceGaze: 2 субъекта `p11`, `p12` ($4\,890$ сэмплов).
    * RT-GENE: 2 субъекта ($15\,218$ сэмплов).
    * *Всего Val:* $20\,843$ сэмпла.
  * **Test Set (12.5% субъектов, отложенный чистый тест):**
    * Columbia: 7 субъектов ($735$ сэмплов).
    * MPIIFaceGaze: 2 субъекта `p13`, `p14` ($4\,965$ сэмплов).
    * RT-GENE: 2 субъекта ($15\,220$ сэмплов).
    * *Всего Test:* $20\,920$ сэмплов.

### 2.3. Механизм балансировки батча (Balanced Batch Sampler)
*Проблема:* RT-GENE составляет $72.5\%$ объединенного датасета, а Columbia — всего $3.7\%$. При равномерном сэмплировании градиенты будут полностью подавлены спецификой RT-GENE.  
*Решение:* `WeightedRandomSampler` формирует каждый батч с равными долями источников:
$$P(\text{sample}_i \in D_k) = \frac{1}{3 \cdot N_{D_k}}$$
В каждом батче размера $B=64$ приблизительно по $21$ сэмплу из каждого датасета.

---

## 3. Целевые метрики и их иерархия приоритетов

### 3.1. Иерархия метрик

```
[Приоритет 1] Угловая ошибка направления взгляда (E_ang, градусы)
      │       -> Рассчитывается на ВСЕХ 3 датасетах (158 105 сэмплов)
      ▼
[Приоритет 2] Задержка инференса и FPS (Latency, мс / FPS)
      │       -> Замер на 4 потоках CPU (ONNX AVX2) и GPU
      ▼
[Приоритет 3] Экранная ошибка позиционирования (E_screen, мм и пиксели)
      │       -> Рассчитывается на MPIIFaceGaze (где есть физический экран)
      ▼
[Приоритет 4] Покомпонентная ошибка углов (MAE Pitch, MAE Yaw, градусы)
              -> Диагностика асимметрии вертикального и горизонтального каналов
```

### 3.2. Математические определения метрик

#### 1. Угловая ошибка взгляда $\mathcal{E}_{ang}$ (Приоритет 1)
Вычисляется через скалярное произведение истинного $\mathbf{g}_{gt}$ и предсказанного $\hat{\mathbf{g}}$ единичных 3D-векторов взгляда:

$$\mathcal{E}_{ang} = \arccos\left(\text{clip}\left(\frac{\mathbf{g}_{gt} \cdot \hat{\mathbf{g}}}{\|\mathbf{g}_{gt}\|_2 \|\hat{\mathbf{g}}\|_2}, -1.0, 1.0\right)\right) \cdot \frac{180^\circ}{\pi}$$

Эквивалентная формулировка через сферические углы $(\theta, \phi)$:

$$\mathcal{E}_{ang} = \arccos\left(\text{clip}\left(\cos\theta_{gt}\cos\hat{\theta}\cos(\phi_{gt} - \hat{\phi}) + \sin\theta_{gt}\sin\hat{\theta}, -1.0, 1.0\right)\right) \cdot \frac{180^\circ}{\pi}$$

#### 2. Задержка и пропускная способность (Приоритет 2)
* $T_{model}$ — чистое время выполнения прямого прохода нейросети (Forward pass) на одном кадре:
  $$T_{model} \le 8.0\text{ мс на CPU (4 потока)}, \quad T_{model} \le 2.0\text{ мс на GPU}$$
* $\text{FPS}_{pipeline}$ — частота сквозного конвейера (захват $\to$ детекция лица $\to$ нейросеть $\to$ проекция $\to$ фильтр):
  $$\text{FPS}_{pipeline} = \frac{1}{T_{total}} \ge 60\text{ кадров/сек} \iff T_{total} \le 16.67\text{ мс}$$

#### 3. Экранная ошибка позиционирования $E_{screen}$ (Приоритет 3)
* В пикселях (Евклидово расстояние):
  $$E_{screen}^{px} = \sqrt{(u_{gt} - \hat{u})^2 + (v_{gt} - \hat{v})^2}$$
* В миллиметрах:
  $$E_{screen}^{mm} = \sqrt{\left((u_{gt} - \hat{u}) \cdot \Delta_{px, x}\right)^2 + \left((v_{gt} - \hat{v}) \cdot \Delta_{px, y}\right)^2}$$
  где $\Delta_{px, x}, \Delta_{px, y}$ — шаг пикселя монитора из `screenSize.mat`.

#### 4. Покомпонентная ошибка углов (Приоритет 4)
$$\text{MAE}_{\theta} = \frac{1}{N} \sum_{i=1}^N |\theta_{gt}^{(i)} - \hat{\theta}^{(i)}| \cdot \frac{180^\circ}{\pi}, \quad \text{MAE}_{\phi} = \frac{1}{N} \sum_{i=1}^N |\phi_{gt}^{(i)} - \hat{\phi}^{(i)}| \cdot \frac{180^\circ}{\pi}$$

### 3.3. Контрольные численные ориентиры (Бейзлайны из статей и целевые пороги)

| Уровень требований | Угловая ошибка $\mathcal{E}_{ang}$ | Экранная ошибка $E_{screen}^{px}$ (FHD 24") | Экранная ошибка $E_{screen}^{mm}$ ($d=600\text{ мм}$) | Задержка CPU (4 ядра) | FPS Pipeline | Источник / Обоснование |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Худший бейзлайн (RT-GENE baseline)** | **$8.20^\circ$** | $> 300\text{ px}$ | $> 85\text{ мм}$ | $> 350\text{ мс}$ | $< 3$ FPS | Fischer et al. (4x VGG-16) |
| **Открытый SOTA CPU (yakhyo/gaze)** | **$4.35^\circ$** | $\approx 160\text{ px}$ | $\approx 45.6\text{ мм}$ | $4.2\text{ мс}$ | $> 100$ FPS | MobileOne-s0 ONNX |
| **Открытый SOTA GPU (L2CS-Net)** | **$3.92^\circ$** | $\approx 148\text{ px}$ | $\approx 41.1\text{ мм}$ | $95\text{ мс}$ | $10$ FPS (CPU) | Ahmed et al. (ResNet-50) |
| **Открытый SOTA Transformer (GazeTR)**| **$3.82^\circ$** | $\approx 144\text{ px}$ | $\approx 40.0\text{ мм}$ | $180\text{ мс}$ | $5$ FPS (CPU) | Cheng & Lu (ResNet-18 + ViT) |
| **MVP Порог приемки (Gate Requirement)**| **$\le 5.20^\circ$** | **$\le 180\text{ px}$** | **$\le 50.0\text{ мм}$** | **$\le 8.0\text{ мс}$** | **$\ge 60$ FPS** | **Минимальный допуск для перехода к оптимизациям** |
| **Целевой научный результат (Target)** | **$\le 3.60^\circ$** | **$\le 130\text{ px}$** | **$\le 36.0\text{ мм}$** | **$\le 5.0\text{ мс}$** | **$\ge 90$ FPS** | **Превосходство над L2CS-Net при скорости MobileOne** |
| **С 9-точечной калибровкой (Few-Shot)** | **$\le 1.80^\circ$** | **$\le 65\text{ px}$** | **$\le 18.0\text{ мм}$** | **$\le 5.5\text{ мс}$** | **$\ge 85$ FPS** | **Готовность к практическому управлению GUI** |

---

## 4. Математический аппарат функций потерь (Loss Functions)

Поскольку датасет является гетерогенным (часть данных не имеет экранных координат), применяется **составная маскированная многозадачная функция потерь**.

$$\mathcal{L}_{total} = \mathcal{L}_{L2CS} + \lambda_{cos} \mathcal{L}_{cosine} + \lambda_{scr} \mathcal{L}_{screen}$$

### 4.1. L2CS Loss (Classification + Softmax Expectation Regression)
Предотвращает стягивание предсказаний к математическому ожиданию выборки и обеспечивает уверенное предсказание на периферии.

1. Диапазон углов $[-\theta_{max}, \theta_{max}]$ разбивается на $K = 90$ бинов шириной $w = 1.0^\circ$ ($0.01745\text{ рад}$).
2. Центр $i$-го бина:
   $$c_i = -\theta_{max} + \left(i - \frac{1}{2}\right) w, \quad i \in \{1, \dots, K\}$$
3. Истинный угол переводится в индекс бина:
   $$y_{bin} = \text{clip}\left(\left\lfloor \frac{\theta_{gt} + \theta_{max}}{w} \right\rfloor, 0, K-1\right)$$
4. Сеть предсказывает логиты $z \in \mathbb{R}^K$. Распределение вероятностей через Softmax:
   $$p_i = \frac{e^{z_i / \tau}}{\sum_{j=1}^K e^{z_j / \tau}}, \quad \tau = 1.0 \text{ (температура)}$$
5. Непрерывный угол вычисляется как дифференцируемое математическое ожидание:
   $$\hat{\theta} = \sum_{i=1}^K p_i \cdot c_i$$
6. Функция потерь объединяет Cross-Entropy по распределению бинов и MSE по ожиданию:
   $$\mathcal{L}_{L2CS} = \mathcal{L}_{CE}(p_\theta, y_{bin}^\theta) + \mathcal{L}_{CE}(p_\phi, y_{bin}^\phi) + \alpha_{reg} \left(\|\theta_{gt} - \hat{\theta}\|_2^2 + \|\phi_{gt} - \hat{\phi}\|_2^2\right)$$
   где $\alpha_{reg} = 1.0$.

### 4.2. Векторный косинусный лосс (Cosine Proximity Loss)
Напрямую минимизирует целевую метрику $\mathcal{E}_{ang}$ в пространстве 3D-векторов:

$$\mathcal{L}_{cosine} = 1 - \frac{\mathbf{g}_{gt} \cdot \hat{\mathbf{g}}}{\|\mathbf{g}_{gt}\|_2 \|\hat{\mathbf{g}}\|_2} = 1 - \left(g_x \hat{g}_x + g_y \hat{g}_y + g_z \hat{g}_z\right)$$

Коэффициент в общей функции потерь: $\lambda_{cos} = 2.0$.

### 4.3. Маскированный экранный лосс (Masked Screen Loss)
Активируется **только для сэмплов MPIIFaceGaze**, где флаг `has_screen_coords = True`. Для Columbia и RT-GENE маска зануляет градиенты:

$$\mathcal{L}_{screen} = \frac{1}{\sum_{i=1}^B m_i} \sum_{i=1}^B m_i \cdot \text{Smooth}_{L1}\left(\frac{\mathbf{s}_{gt}^{(i)} - \hat{\mathbf{s}}^{(i)}}{\sigma_{norm}}\right)$$

где:
* $m_i \in \{0, 1\}$ — булев индикатор наличия экранных меток.
* $\sigma_{norm} = \begin{bmatrix} W_{screen} \\ H_{screen} \end{bmatrix}$ — нормализующий вектор разрешения для приведения ошибки к масштабу $[0, 1]$.
* $\text{Smooth}_{L1}(x) = \begin{cases} 0.5 x^2, & \text{если } |x| < 1 \\ |x| - 0.5, & \text{иначе} \end{cases}$
* Коэффициент: $\lambda_{scr} = 1.0$.

---

## 5. Пул исследуемых легковесных архитектур

В соответствии с требованием работы $\ge 60\text{ FPS}$ на CPU, исследуются только компактные бэкбоны с числом параметров $\le 12\text{M}$ и сложностью $\le 1.0\text{ GFLOPs}$.

| ID | Название архитектуры | Число параметров | FLOPs (вход 224x224) | Latency CPU (4 ядра) | Назначение в исследовании |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A1** | **`mobilenet_v3_small`** | $2.5\text{M}$ | $0.06\text{ GFLOPs}$ | $\approx 5.5\text{ мс}$ | **Базовый легковесный референс (MVP Baseline)** |
| **A2** | **`mobileone_s0`** | $2.1\text{M}$ | $0.27\text{ GFLOPs}$ | $\approx 4.2\text{ мс}$ | **Фаворит по скорости инференса (структурная перепараметризация)** |
| **A3** | **`repvgg_a0`** | $8.3\text{M}$ | $0.80\text{ GFLOPs}$ | $\approx 4.0\text{ мс}$ | **Максимальная плотность вычислений AVX2/GPU (чистый стек $3\times3$)** |
| **A4** | **`fastvit_t8`** | $4.0\text{M}$ | $0.70\text{ GFLOPs}$ | $\approx 6.8\text{ мс}$ | **Гибридный сверточно-трансформерный бэкбон (Apple Research)** |
| **A5** | **`resnet18`** | $11.7\text{M}$ | $1.80\text{ GFLOPs}$ | $\approx 18.0\text{ мс}$ | **Верхняя граница точности (Benchmark Ceiling, без лимита на CPU FPS)** |

---

## 6. Протокол обучения и инфраструктурные контракты

### 6.1. Гиперпараметры оптимизации
* **Оптимизатор:** `torch.optim.AdamW`.
  * Начальный Learning Rate: $\eta_0 = 5 \times 10^{-4}$.
  * Weight Decay: $10^{-4}$.
  * Betas: $(0.9, 0.999)$, $\epsilon = 10^{-8}$.
* **Шедулер:** `torch.optim.lr_scheduler.CosineAnnealingLR`.
  * $T_{max} = 150$ эпох.
  * $\eta_{min} = 10^{-6}$.
* **Размер батча:** $B = 64$ (динамически снижается до 32 при нехватке VRAM).
* **Early Stopping:** Пациенс $P = 15$ эпох по валидационной метрике $\mathcal{E}_{ang}^{val}$.
* **Минимальное число эпох до остановки:** 30 эпох.

### 6.2. Аугментации данных
Применяются на лету только к обучающей выборке (Train):
1. **Цветовая фотометрия:** `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)`.
2. **Гауссов шум:** $\mathcal{N}(0, \sigma^2)$ с $\sigma \in [0, 0.02]$.
3. **Размытие:** `GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))` с вероятностью $p=0.2$.
4. **Строгий запрет:** Горизонтальное отражение (`RandomHorizontalFlip`) **категорически запрещено** без одновременной инверсии знака угла рыскания ($\phi \to -\phi$) и $x$-компоненты вектора ($g_x \to -g_x$).

### 6.3. Контракт сохранения контрольных точек (Checkpointing)
Файл контрольной точки `.pt` обязан содержать словарь метаданных для автономного восстановления без внешнего кода:
```python
checkpoint_payload = {
    "epoch": current_epoch,
    "architecture_name": model.architecture_name,
    "backbone_name": model.backbone_name,
    "hyperparameters": {
        "num_bins": 90,
        "max_angle_deg": 45.0,
        "lr": current_lr,
        "batch_size": batch_size
    },
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_angular_error": best_val_metric,
    "metrics_history": metrics_tracker.to_dict()
}
```

---

## 7. Чек-лист готовности для Агента-Исполнителя

Перед запуском любого тренировочного скрипта агент обязан верифицировать:

- [ ] **Датасеты:** Проверено наличие файлов `columbia_gaze_full.parquet`, `mpiifacegaze_full.parquet`, `rtgene_full_labels.parquet`.
- [ ] **Балансировка:** Инициализирован `WeightedRandomSampler` для соотношения сэмплов $1:1:1$ по трем источникам.
- [ ] **Сплиты:** Подтверждено строгое отсутствие пересечений субъектов (`subject_id`) между `train`, `val` и `test`.
- [ ] **Loss:** Функция потерь использует составной лосс (L2CS + Cosine + Masked Screen).
- [ ] **Метрики:** Логгер рассчитывает истинную угловую ошибку через $\arccos$ векторного произведения (не MSE градусов).
- [ ] **Память:** Перед тренировочным циклом вызван `MemoryManager.aggressive_cleanup()`.
- [ ] **Критерий успеха MVP:** Достигнута угловая ошибка на валидации $\mathcal{E}_{ang}^{val} \le 5.20^\circ$ на бэкбоне `mobilenet_v3_small` или `mobileone_s0`.
