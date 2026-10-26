# Системное и техническое руководство по разработке и исследованию моделей машинного обучения для управления графическим интерфейсом на основе оценки направления взгляда

Документ устанавливает инженерные регламенты, архитектурные контракты взаимодействия агентов, правила управления памятью, требования к коду и механизмы отказоустойчивости для выпускной квалификационной работы (ВКР).

Связанные файлы:
- Техническая постановка темы: [Тема.txt](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/Тема.txt)
- Настоящее руководство: [00_system_and_agent_guidelines.md](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md)

---

## 1. Архитектурные правила взаимодействия агентов

### 1.1. Ролевая модель и конвейер исполнения

Взаимодействие агентов организовано в виде направленного ациклического графа передачи артефактов со строгой изоляцией ответственности:

```
[Постановка задачи] 
         │
         ▼
[Кодер (Coder)] ──(Код модели и пайплайна)──► [Верификатор (Code Verifier)]
                                                         │
                                        ┌────────────────┴────────────────┐
                                   FAIL │                                 │ PASS
                                        ▼                                 ▼
                                  [Возврат Кодеру]               [Тренер на GPU (GPU Trainer)]
                                                                          │
                                                                   (Чекпоинт + Логи)
                                                                          ▼
                                                                [Аналитик метрик]
                                                                          │
                                                                          ▼
                                                               [Реестр гипотез]
```

1. <a id="coder"></a>**Кодер ([`Coder`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#coder)):**
   - Генерирует код моделей, функций потерь, слоев и пайплайнов обработки данных.
   - Ограничения: функционал в приоритете, краткость кода, использование `;` для объединения связанных инструкций, полный запрет комментариев, запрет изменения названий и сигнатур существующих функций при рефакторинге, обязательная явная типизация входных и выходных параметров через модуль `typing`.

2. <a id="codeverifier"></a>**Верификатор кода ([`CodeVerifier`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#codeverifier)):**
   - Выполняет статический анализ типов через `mypy` и синтаксический контроль.
   - Проверяет инварианты размерностей тензоров на каждом слое сети.
   - Анализирует код на наличие утечек памяти: поиск незакрытых графов вычислений (отсутствие `.detach()` при сохранении промежуточных значений лосса), отсутствие очистки промежуточных ссылок, корректность освобождения файловых дескрипторов.
   - Результат: бинарный вердикт `PASS` или `FAIL` с указанием конкретной строки и нарушенного инварианта. Переход к следующему этапу возможен только при статусе `PASS`.

3. <a id="gputrainer"></a>**Тренер на GPU ([`GPUTrainer`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#gputrainer)):**
   - Запускает процесс обучения исключительно в изолированном дочернем процессе с системным тайм-аутом (`watchdog`).
   - Контролирует динамику потребления оперативной (RAM) и видеопамяти (VRAM).
   - Исполняет протокол обучения: минимум 150 эпох, механизм ранней остановки ([`EarlyStopping`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#earlystopping)), поэпошное логирование метрик, сохранение чекпоинтов с полной конфигурацией гиперпараметров.

4. <a id="metricsanalyst"></a>**Аналитик метрик ([`MetricsAnalyst`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#metricsanalyst)):**
   - Вычисляет финальные физические и геометрические метрики на тестовой выборке: средняя ошибка в пикселях $\mathcal{E}_{\text{pix}}$, угловая ошибка $\mathcal{E}_{\text{deg}}$, время задержки вывода $T_{\text{inf}}$ (мс), 95-й перцентиль задержки $P_{95}$, пиковое потребление VRAM (МБ).
   - Формирует сводную сравнительную таблицу `pandas.DataFrame` со стилизацией через `background_gradient`.
   - Заносит структурированную запись в файл реестра гипотез.
   - **Регламент лидерборда решений:** При фиксации любой новой конфигурации, дообученной модели или внешнего адаптера агент обязан обновить официальный онлайн-лидерборд решений в Google Sheets: [Google Sheets Gaze Tracking Leaderboard](https://docs.google.com/spreadsheets/d/1J4LHhZXiZYnBFjymL6ayOt2jAafnt4sISfhsUsNiKLo/edit?usp=sharing), а также локальные реестры [`docs/leaderboard.csv`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/leaderboard.csv) и [`docs/leaderboard.tsv`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/leaderboard.tsv) через [`scripts/export_leaderboard.py`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/scripts/export_leaderboard.py) и [`scripts/sync_google_sheet_leaderboard.py`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/scripts/sync_google_sheet_leaderboard.py).
   - **Строгий порядок сортировки лидерборда:** Таблица должна оставаться отсортированной по невозрастанию качества:
     $$\text{Метрика (наименьшая угловая ошибка } \mathcal{E}_{ang}\text{)} \longrightarrow \text{Год создания (2026 } \to 2024\text{)} \longrightarrow \text{Размер обучающего датасета}.$$
   - **Схема колонок лидерборда:**
     `Ранг | Название | Ошибка 1 | Ошибка 2 | ... | Ошибка N | Задержка CPU | Задержка GPU | Потребление RAM | Потребление VRAM | Год или время создания | Место, где лежат веса | Источник если есть`

---

### 1.2. Мета-луп: жизненный цикл гипотез

Мета-луп определяет последовательность действий до написания рабочего кода обучения:

```
[Постановка темы] ──► [Декомпозиция аспектов] ──► [Метрики и Losses] ──► [Замкнутая проверка] ──► [Список гипотез] ──► [Эксперимент]
```

1. **Постановка темы:** 
   Оценка направления взгляда (Gaze Tracking) для прямого управления координатами курсора графического интерфейса $(x, y) \in [0, W] \times [0, H]$ по видеопотоку стандартной веб-камеры с целевой задержкой инференса $T_{\text{inf}} \le 16.6 \text{ мс}$ (частота кадров $\ge 60 \text{ FPS}$).

2. **Декомпозиция аспектов:**
   - Локализация лица и областей глаз (Face and Eye ROI extraction).
   - Выделение признаков положения головы (Head Pose: pitch, yaw, roll).
   - Оценка вектора взгляда в трехмерном пространстве камеры $\mathbf{g} = (g_x, g_y, g_z)^T \in \mathbb{R}^3$, где $\|\mathbf{g}\|_2 = 1$.
   - Отображение вектора $\mathbf{g}$ в экранные координаты $(x_{\text{screen}}, y_{\text{screen}})$ через матрицу проекции и геометрическую калибровку.
   - Фильтрация дрожания координат (Anti-jitter filter: One-Euro Filter или фильтр Калмана).

3. **Метрики и функции потерь (Losses):**
   - **Угловая ошибка (Angular Error):**
     $$\mathcal{E}_{\text{deg}} = \frac{180^\circ}{\pi} \arccos\left(\frac{\mathbf{g} \cdot \hat{\mathbf{g}}}{\|\mathbf{g}\|_2 \|\hat{\mathbf{g}}\|_2}\right)$$
   - **Евклидова ошибка на экране (Pixel MAE):**
     $$\mathcal{E}_{\text{pix}} = \sqrt{(x - \hat{x})^2 + (y - \hat{y})^2}$$
   - **Нормализованная ошибка экрана:**
     $$\mathcal{E}_{\text{norm}} = \frac{\mathcal{E}_{\text{pix}}}{\sqrt{W^2 + H^2}}$$
   - **Функции потерь обучения:**
     - Угловой лосс: $\mathcal{L}_{\text{ang}} = 1 - \frac{\mathbf{g} \cdot \hat{\mathbf{g}}}{\|\mathbf{g}\|_2 \|\hat{\mathbf{g}}\|_2}$
     - Smooth L1 Loss для экранных координат:
       $$\mathcal{L}_{\text{smooth}}(\mathbf{p}, \hat{\mathbf{p}}) = \frac{1}{2} \sum_{i \in \{x,y\}} \begin{cases} 0.5 (p_i - \hat{p}_i)^2, & \text{если } |p_i - \hat{p}_i| < 1 \\ |p_i - \hat{p}_i| - 0.5, & \text{иначе} \end{cases}$$

4. **Замкнутая проверка (Sanity Check):**
   - Перед запуском полного цикла обучения модель обязана переобучиться на подвыборке из ровно 4 сэмплов до значения функции потерь $\mathcal{L} < 10^{-4}$ за число итераций $N \le 50$.
   - Проверка градиентов: $\forall \theta \in \Theta: \nabla_\theta \mathcal{L} \ne 0$ и $\text{isnan}(\nabla_\theta \mathcal{L}) = \text{False}$, $\text{isinf}(\nabla_\theta \mathcal{L}) = \text{False}$.

5. **Формирование списка гипотез:**
   - Фиксация гипотезы с присвоением уникального идентификатора (`H-001`, `H-002`, ...).

---

### 1.3. Основной луп эксперимента

Основной луп выполняется строго по следующим шагам:
1. Запрос системных ресурсов: измерение свободной оперативной памяти и видеопамяти.
2. Инициализация генератора псевдослучайных чисел фиксированным сидом (`seed=42`):
   ```python
   import os, random, numpy as np, torch
   def set_deterministic(seed: int = 42) -> None:
       random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
   ```
3. Проверка готовности VRAM: объем свободной видеопамяти $\ge 14.5 \text{ ГБ}$ (для Colab T4).
4. Загрузка весов базовой модели в RAM (`cpu`).
5. Перенос модели в VRAM (`cuda`) строго перед входом в цикл эпох.
6. Выполнение эпохи обучения с расчетом $\mathcal{L}_{\text{train}}$.
7. Выполнение эпохи валидации с расчетом $\mathcal{L}_{\text{val}}$.
8. Вызов [`EarlyStopping`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#earlystopping) и проверка критериев остановки.
9. При достижении критерия остановки: выгрузка модели из VRAM в RAM, выполнение очистки кэша CUDA, расчет метрик на тестовом наборе $\mathcal{L}_{\text{test}}$, фиксация результатов.

---

### 1.4. Условия остановки (Stopping Criteria)

Процесс обучения немедленно прерывается при наступлении любого из следующих событий:
1. **Сходимость EarlyStopping:** значение $\mathcal{L}_{\text{val}}$ не улучшилось более чем на $\delta = 10^{-4}$ в течение $\tau = 15$ последовательных эпох.
2. **Превышение лимита времени (Watchdog):** общее время исполнения задачи достигло $T_{\text{max}} = 1140 \text{ секунд}$ (19 минут).
3. **Критическое исчерпание RAM:** объем свободной оперативной памяти снизился до $RAM_{\text{free}} \le 1.0 \text{ ГБ}$ (в Colab предел OOM cgroups — 12.0 ГБ).
4. **Критическое исчерпание VRAM:** объем свободной видеопамяти снизился до $VRAM_{\text{free}} \le 0.5 \text{ ГБ}$.
5. **Дивергенция градиентов:** значение лосса $\mathcal{L}$ принимает значения `NaN` или `Inf` на протяжении 3 батчей подряд.

---

### 1.5. Реестр гипотез: схема и правила ведения

Все результаты экспериментов подлежат обязательной записи в единый структурированный реестр `c:\Users\jack_\MAIN\ВУЗ\ВКР\docs\experiments_registry.json`.

Схема записи реестра:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": [
    "hypothesis_id",
    "timestamp",
    "architecture",
    "backbone",
    "input_resolution",
    "batch_size",
    "epochs_trained",
    "loss_train",
    "loss_val",
    "loss_test",
    "pixel_mae",
    "angular_error_deg",
    "latency_ms",
    "vram_peak_gb",
    "weights_path",
    "status"
  ],
  "properties": {
    "hypothesis_id": {"type": "string", "pattern": "^H-[0-9]{3}$"},
    "timestamp": {"type": "string", "format": "date-time"},
    "architecture": {"type": "string"},
    "backbone": {"type": "string"},
    "input_resolution": {"type": "string", "pattern": "^[0-9]+x[0-9]+$"},
    "batch_size": {"type": "integer", "minimum": 1},
    "epochs_trained": {"type": "integer", "minimum": 1},
    "loss_train": {"type": "number"},
    "loss_val": {"type": "number"},
    "loss_test": {"type": "number"},
    "pixel_mae": {"type": "number"},
    "angular_error_deg": {"type": "number"},
    "latency_ms": {"type": "number"},
    "vram_peak_gb": {"type": "number"},
    "weights_path": {"type": "string"},
    "status": {"type": "string", "enum": ["ACCEPTED", "REJECTED", "CRASHED"]}
  }
}
```

Правило принятия гипотезы (`ACCEPTED`):
Гипотеза принимается тогда и только тогда, когда:
$$\mathcal{E}_{\text{deg}} \le \mathcal{E}_{\text{deg}}^{\text{baseline}} \times 0.95 \quad \text{и} \quad T_{\text{inf}} \le 16.6 \text{ мс}$$
Если улучшение метрики сопровождается превышением задержки $T_{\text{inf}} > 16.6 \text{ мс}$, гипотеза получает статус `REJECTED`.

---

## 2. Правила среды исполнения (Google Colab T4 и локальная среда)

### 2.1. Аппаратные ограничения и бюджет ресурсов

| Параметр | Google Colab (Tesla T4) | Локальная среда (Baseline) | Критический лимит |
|---|---|---|---|
| Общая системная RAM | 12.7 ГБ (13620844 КБ) | Согласно хосту | OOM Kill при > 12.0 ГБ в Colab |
| Зарезервировано системой/Colab runtime | ~2.0 ГБ (ядро, python3, daemons) | ~3.0 - 4.0 ГБ ОС | Недоступно для аллокаций |
| Доступная RAM для обучения | **10.7 ГБ** | $RAM_{\text{total}} - RAM_{\text{OS}}$ | Аварийный сброс при $RAM_{\text{free}} < 1.0 \text{ ГБ}$ |
| Видеопамять (VRAM) | 15.0 ГБ (15360 МиБ GDDR6) | Согласно GPU | CUDA OOM при превышении |
| Доступная VRAM в чистом состоянии | **$\ge 14.5$ ГБ** | Согласно GPU | Старт запрещен при $VRAM_{\text{free}} < 14.5 \text{ ГБ}$ |
| Лимит времени одной сессии Colab | 12 часов максимум | Не ограничен | Прерывание задачи при $T \ge 20 \text{ мин}$ |

---

### 2.2. Запрет хардкода и динамическое измерение свободной памяти

**Запрещено:** фиксировать `batch_size` константой без предварительного вычисления допустимого объема аллокации.

Формула оценки объема тензора активаций для сверточного или полносвязного слоя:
$$M_{\text{batch}} = B \times \sum_{l=1}^L (C_l \times H_l \times W_l \times S_{\text{dtype}})$$
где $B$ — размер батча, $C_l, H_l, W_l$ — размерности выхода слоя $l$, $S_{\text{dtype}} = 4 \text{ байта}$ (для `torch.float32`) или $2 \text{ байта}$ (для `torch.float16`).

<a id="memoryguard"></a>
Класс динамического контроля памяти [`MemoryGuard`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#memoryguard):

```python
import os, gc, psutil, torch
from typing import Tuple

class MemoryGuard:
    @staticmethod
    def get_ram_free_gb() -> float:
        return psutil.virtual_memory().available / (1024.0 ** 3)

    @staticmethod
    def get_vram_free_gb(device_idx: int = 0) -> float:
        if not torch.cuda.is_available(): return 0.0
        free_bytes, _ = torch.cuda.mem_get_info(device_idx)
        return free_bytes / (1024.0 ** 3)

    @staticmethod
    def check_pretraining_invariants(min_vram_gb: float = 14.5, min_ram_gb: float = 4.0) -> None:
        MemoryGuard.aggressive_cleanup()
        vram_free = MemoryGuard.get_vram_free_gb()
        ram_free = MemoryGuard.get_ram_free_gb()
        if vram_free < min_vram_gb:
            raise RuntimeError(f"VRAM invariant violation: Free={vram_free:.2f}GB < Required={min_vram_gb:.2f}GB. Kill background GPU processes.")
        if ram_free < min_ram_gb:
            raise RuntimeError(f"RAM invariant violation: Free={ram_free:.2f}GB < Required={min_ram_gb:.2f}GB.")

    @staticmethod
    def aggressive_cleanup() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
```

---

### 2.3. Алгоритм и протокол агрессивной очистки VRAM

При возникновении исключения `torch.cuda.OutOfMemoryError` или при плановой смене модели выполняется жесткий протокол деаллокации:

```python
import gc, torch

def release_gpu_resources(*objects_to_delete) -> None:
    for obj in objects_to_delete:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
```

Если контекст CUDA поврежден или не освобождает память: выполнение задач в дочерних процессах (`multiprocessing.Process` со способом старта `spawn`). При завершении процесса ядро операционной системы принудительно освобождает все файловые дескрипторы устройства `/dev/nvidia*`, полностью сбрасывая контекст драйвера.

---

### 2.4. Правила размещения моделей в памяти

1. Модель инициализируется строго в системной памяти: `model = GazeModel().to('cpu')`.
2. Перенос в VRAM разрешен исключительно перед вызовом функции обучения или батчевого инференса:
   ```python
   model.to('cuda:0')
   ```
3. Контекстный менеджер изоляции исполнения на GPU:

```python
from contextlib import contextmanager
from typing import Generator
import torch, gc

@contextmanager
def scoped_gpu_context(model: torch.nn.Module) -> Generator[torch.nn.Module, None, None]:
    try:
        model.to('cuda:0')
        yield model
    finally:
        model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

---

## 3. Требования к коду и логированию

### 3.1. Стандартизированный формат логирования

Каждый вывод в консоль обязан соответствовать регулярному выражению:
`^\[\d{2}:\d{2}:\d{2}\] \[\d{2}:\d{2}:\d{2}\] \[(?:CPU|GPU)\] \[RAM: \d+\.\dGB\] \[VRAM: \d+\.\dGB\] .+$`

- `Время 1`: Текущее астрономическое время по Московскому часовому поясу (UTC+3, MSK) в формате `HH:MM:SS`.
- `Время 2`: Прошедшее время ($\Delta t$) с момента предыдущего вывода лога в формате `HH:MM:SS`.
- `[CPU/GPU]`: Текущий контекст вычислений.
- `[RAM: X.XGB]`: Объем свободной оперативной памяти с одним десятичным знаком.
- `[VRAM: Y.YGB]`: Объем свободной видеопамяти с одним десятичным знаком.
- `Сообщение`: Текст без переводов строк внутри сообщения.

<a id="metriclogger"></a>
Реализация класса [`MetricLogger`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#metriclogger):

```python
import sys, time, psutil, torch
from datetime import datetime, timezone, timedelta
from typing import Optional

class MetricLogger:
    def __init__(self, target_tz_offset_hours: int = 3) -> None:
        self.tz = timezone(timedelta(hours=target_tz_offset_hours))
        self.last_timestamp: float = time.monotonic()
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    def log(self, message: str, device: str = "GPU") -> None:
        now_mono = time.monotonic()
        delta_sec = int(now_mono - self.last_timestamp)
        self.last_timestamp = now_mono
        
        elapsed_str = f"{delta_sec // 3600:02d}:{(delta_sec % 3600) // 60:02d}:{delta_sec % 60:02d}"
        now_dt = datetime.now(self.tz).strftime("%H:%M:%S")
        
        ram_gb = psutil.virtual_memory().available / (1024.0 ** 3)
        vram_gb = 0.0
        if torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info(0)
            vram_gb = free_b / (1024.0 ** 3)
            
        formatted_msg = f"[{now_dt}] [{elapsed_str}] [{device}] [RAM: {ram_gb:.1f}GB] [VRAM: {vram_gb:.1f}GB] {message}"
        print(formatted_msg, flush=True)

LOGGER = MetricLogger()
```

---

### 3.2. Требования к протоколу обучения

1. Минимальное количество эпох: **150**.
2. Обязательный механизм прерывания: [`EarlyStopping`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#earlystopping).
3. Формат вывода на каждой эпохе:
   `Epoch {epoch}/{total} | L Train: {loss_train:.4f} | L Val: {loss_val:.4f}`
4. Формат вывода при тестировании:
   `L Test: {loss_test:.4f}`

<a id="earlystopping"></a>
Реализация модуля [`EarlyStopping`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#earlystopping):

```python
import copy, torch
from typing import Optional, Dict, Any

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min", restore_best_weights: bool = True) -> None:
        self.patience = patience; self.min_delta = min_delta; self.mode = mode; self.restore_best_weights = restore_best_weights
        self.counter: int = 0; self.best_score: Optional[float] = None; self.early_stop: bool = False
        self.best_weights: Optional[Dict[str, Any]] = None; self.best_loss: float = float('inf') if mode == "min" else float('-inf')

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        score = -val_loss if self.mode == "min" else val_loss
        if self.best_score is None:
            self.best_score = score; self.best_loss = val_loss
            if self.restore_best_weights: self.best_weights = copy.deepcopy(model.state_dict())
            return False
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score; self.best_loss = val_loss
            if self.restore_best_weights: self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        return self.early_stop

    def restore(self, model: torch.nn.Module) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
```

---

### 3.3. Сериализация и версионирование контрольных точек (Checkpoints)

Файл весов (.pt / .pth / .safetensors) обязан включать словарь с архитектурными гиперпараметрами, состоянием модели, состоянием оптимизатора и планировщика шага. Запрещено сохранять чистый `state_dict` без словаря конфигурации.

<a id="modelcheckpointmanager"></a>
Реализация менеджера контрольных точек [`ModelCheckpointManager`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#modelcheckpointmanager):

```python
import os, torch
from typing import Dict, Any

class ModelCheckpointManager:
    @staticmethod
    def save_checkpoint(
        filepath: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'hyperparameters': hyperparameters,
            'metrics': metrics
        }
        torch.save(checkpoint, filepath)

    @staticmethod
    def load_checkpoint(
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found: {filepath}")
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded model weights from file: {os.path.abspath(filepath)}", flush=True)
        return checkpoint
```

---

### 3.4. Сводное сравнительное табличное представление метрик (DataFrame Styler)

Для визуализации и сравнения результатов используется `pandas.DataFrame` со стилизацией градиентом от зеленого (лучший результат) к красному (худший результат).

<a id="render_model_comparison_table"></a>
Функция [`render_model_comparison_table`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#render_model_comparison_table):

```python
import pandas as pd
from typing import List, Dict, Any

def render_model_comparison_table(records: List[Dict[str, Any]]) -> pd.io.formats.style.Styler:
    df = pd.DataFrame(records)
    cols_order = [
        "hypothesis_id", "architecture", "backbone", 
        "val_loss", "test_loss", "pixel_mae", "deg_error", 
        "latency_ms", "vram_peak_mb"
    ]
    df = df[[c for c in cols_order if c in df.columns]]
    
    styler = df.style.format({
        "val_loss": "{:.4f}",
        "test_loss": "{:.4f}",
        "pixel_mae": "{:.1f} px",
        "deg_error": "{:.2f}°",
        "latency_ms": "{:.2f} ms",
        "vram_peak_mb": "{:.0f} MB"
    })
    
    minimize_cols = ["val_loss", "test_loss", "pixel_mae", "deg_error", "latency_ms", "vram_peak_mb"]
    for col in minimize_cols:
        if col in df.columns:
            styler = styler.background_gradient(subset=[col], cmap="RdYlGn_r")
            
    return styler
```

---

### 3.5. Спецификация стиля кодовой базы

Кодер обязан следовать правилам:
1. **Компактность синтаксиса:** объединение тривиальных присваиваний через `;` (например: `self.w = w; self.h = h`).
2. **Запрет комментариев:** удалять все строчные и блочные комментарии `# ...`. Код обязан описываться собственными типами данных и сигнатурами.
3. **Сохранение сигнатур:** запрещено переименовывать существующие классы, функции и аргументы при рефакторинге.
4. **Покрытие краевых случаев:**
   - Деление на норму: $\mathbf{g}_{\text{norm}} = \frac{\mathbf{g}}{\max(\|\mathbf{g}\|_2, \epsilon)}$, где $\epsilon = 10^{-8}$.
   - Отсутствие лица/глаз в кадре: возврат нулевого флага валидности `valid=False` и последнего валидного предсказания без возбуждения исключений в потоке.
   - Ограничение экранных координат: строгий clamp в диапазон экрана:
     $$x_{\text{clamped}} = \min(\max(x, 0), W - 1); \quad y_{\text{clamped}} = \min(\max(y, 0), H - 1)$$

---

## 4. Предотвращение зависаний, сбоев и утечек ресурсов

### 4.1. Временные ограничения и сторожевой таймер (Watchdog)

Лимит времени выполнения любого отдельного скрипта или задачи: строго **$< 20$ минут (1200 секунд)**. Превышение лимита означает зависание процесса (deadlock или бесконечный цикл) и влечет немедленное уничтожение процесса через `SIGKILL`.

<a id="executionwatchdog"></a>
Исполняемый сторожевой запуск скриптов [`ExecutionWatchdog`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/00_system_and_agent_guidelines.md#executionwatchdog):

```python
import subprocess, sys

class ExecutionWatchdog:
    @staticmethod
    def run_guarded(command_args: list[str], max_timeout_seconds: int = 1140) -> subprocess.CompletedProcess:
        full_cmd = [sys.executable, "-u"] + command_args
        try:
            result = subprocess.run(
                full_cmd,
                timeout=max_timeout_seconds,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return result
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"Task exceeded execution limit of {max_timeout_seconds}s. Terminated by Watchdog.") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Task crashed with return code {e.returncode}. Stderr: {e.stderr}") from e
```

---

### 4.2. Буферизация стандартных потоков вывода

Зависание вывода в Colab и Jupyter Notebook происходит из-за буферизации `stdout`.

Обязательные требования:
1. Запуск любых скриптов строго с флагом `-u`: `python -u train.py`.
2. В начале входного файла проекта выполняется принудительная переконфигурация потоков:
   ```python
   import sys
   sys.stdout.reconfigure(line_buffering=True)
   sys.stderr.reconfigure(line_buffering=True)
   ```

---

### 4.3. Оптимизация и безопасность DataLoader в PyTorch

Неправильная настройка `torch.utils.data.DataLoader` в среде Linux (Ubuntu / Colab) приводит к зависанию дочерних воркеров (deadlock в `shm_open` / IPC POSIX).

| Параметр | Значение Colab T4 | Локальная среда Windows | Причина фиксации |
|---|---|---|---|
| `num_workers` | **2** | **0** или **2** | 2 ядра vCPU в Colab. Значения $> 2$ приводят к троттлингу контекстных переключений. На Windows `num_workers > 0` требует изоляции `if __name__ == '__main__':` |
| `pin_memory` | **True** | **True** | Ускоряет асинхронную пересылку Host $\to$ Device через DMA страницы |
| `persistent_workers` | **True** | **False** (на Windows) | Исключает пересоздание пула процессов на каждой эпохе (экономия 2-3 с на эпоху) |
| `prefetch_factor` | **2** | **2** | Кэширование ровно 2 батчей на воркера, предотвращает переполнение RAM |

Конфигурация DataLoader:

```python
import torch
from torch.utils.data import DataLoader, Dataset

def build_secure_dataloader(dataset: Dataset, batch_size: int, is_train: bool) -> DataLoader:
    is_linux = sys.platform.startswith('linux')
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=2 if is_linux else 0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(is_linux and 2 > 0),
        prefetch_factor=2 if (is_linux and 2 > 0) else None,
        drop_last=is_train
    )
```

---

### 4.4. Доменные антипаттерны в пайплайнах Gaze Tracking

#### Антипаттерн 1: Переполнение внутреннего буфера OpenCV VideoCapture
*Симптом:* отставание координат курсора от реального движения глаз на 5–15 кадров (растущая задержка управления).  
*Физическая причина:* драйвер камеры (V4L2 в Linux, DirectShow/MediaFoundation в Windows) по умолчанию использует системное кольцевое буферирование на 5–10 кадров. Вызов `cap.read()` возвращает кадр из хвоста буфера, а не актуальный кадр матрицы.  
*Решение:* <a id="threadedcameracapture"></a>создание выделенного потока фонового чтения, который перезаписывает единственную атомарную ячейку текущего кадра, и аппаратная установка размера буфера в 1:

```python
import cv2, threading, time
from typing import Optional, Tuple
import numpy as np

class ThreadedCameraCapture:
    def __init__(self, camera_index: int = 0, target_width: int = 1280, target_height: int = 720, target_fps: int = 60) -> None:
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_timestamp_ns: int = 0
        self.is_running: bool = False
        self.lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None

    def start(self) -> "ThreadedCameraCapture":
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.worker_thread.start()
        return self

    def _capture_worker(self) -> None:
        while self.is_running:
            success, frame = self.cap.read()
            if success:
                ts = time.perf_counter_ns()
                with self.lock:
                    self.latest_frame = frame
                    self.latest_timestamp_ns = ts
            else:
                time.sleep(0.001)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int]:
        with self.lock:
            if self.latest_frame is None: return None, 0
            return self.latest_frame.copy(), self.latest_timestamp_ns

    def release(self) -> None:
        self.is_running = False
        if self.worker_thread: self.worker_thread.join(timeout=1.0)
        self.cap.release()
```

#### Антипаттерн 2: Domain Shift оптической системы камеры
*Симптом:* модель, обученная на публичных датасетах (MPIIGaze, GazeCapture), выдает ошибку $> 15^\circ$ на собственной камере (например, Samsung Galaxy S25 Ultra или веб-камере).  
*Физическая причина:* различие матриц внутренних параметров (Intrinsic Camera Matrix $K$) и дисторсии:
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
где $f_x, f_y$ — фокусные расстояния в пикселях, $(c_x, c_y)$ — главная точка оптической оси.  
*Решение:* нормализация головы (Head Pose Normalization):
1. Оценка позы головы в пространстве камеры через решение задачи Perspective-n-Point (PnP) по 6 3D-антропометрическим опорным точкам лица.
2. Построение матрицы поворота $R \in SO(3)$ и масштаба $S$, переводящих виртуальную оптическую ось камеры в нормализованную систему координат с фиксированным расстоянием $d_{\text{norm}} = 600 \text{ мм}$ и фокусным расстоянием $f_{\text{norm}} = 1000$.
3. Поворот вектора взгляда обратно в мировую систему координат: $\mathbf{g}_{\text{world}} = R^{-1} \mathbf{g}_{\text{norm}}$.

#### Антипаттерн 3: Рассинхрон кадров и неравномерный джиттер времени
*Симптом:* метки целевых точек на экране не соответствуют реальному направлению взгляда в обучающей выборке.  
*Физическая причина:* видеокамера захватывает кадры с частотой 60 Гц (шаг 16.66 мс), экран генерирует события отрисовки точки с частотой 60/120/144 Гц без аппаратной синхронизации.  
*Решение:* фиксация временных меток через высокоточный аппаратный таймер `time.perf_counter_ns()`. Для каждого кадра с меткой времени $t_{\text{cam}}$ вычисляются целевые экранные координаты путем линейной интерполяции между двумя ближайшими точками траектории движения $(t_1, \mathbf{p}_1)$ и $(t_2, \mathbf{p}_2)$:
$$\mathbf{p}(t_{\text{cam}}) = \mathbf{p}_1 + \frac{t_{\text{cam}} - t_1}{t_2 - t_1} (\mathbf{p}_2 - \mathbf{p}_1), \quad t_1 \le t_{\text{cam}} \le t_2$$

#### Антипаттерн 4: Нелинейное геометрическое искажение проекции «глаз — экран»
*Симптом:* в центре экрана ошибка минимальна ($< 2 \text{ см}$), а по краям и углам монитора возрастает экспоненциально ($> 10 \text{ см}$).  
*Физическая причина:* плоскость экрана $Z_{\text{screen}} = 0$ не ортогональна оптической оси камеры (угол наклона ноутбука/монитора $\alpha \ne 0$), расстояние от глаз до углов экрана $D(\phi, \theta) = \frac{D_0}{\cos\theta \cos\phi}$ существенно больше расстояния до центра. Линейная интерполяция вектора $\mathbf{g}$ некорректна.  
*Решение:* расчет точки пересечения 3D-луча с плоскостью монитора:
$$\mathbf{p}_{\text{screen}} = \mathbf{o}_{\text{eye}} + t \mathbf{g}, \quad t = \frac{\mathbf{n} \cdot (\mathbf{p}_{\text{screen\_0}} - \mathbf{o}_{\text{eye}})}{\mathbf{n} \cdot \mathbf{g}}$$
где $\mathbf{n}$ — нормаль к плоскости монитора, $\mathbf{p}_{\text{screen\_0}}$ — начало координат экрана в системе камеры.  
На уровне инференса пользовательской калибровки: применение 9-точечной полиномиальной регрессии 2-го порядка для компенсации остаточной дисторсии:
$$\begin{bmatrix} x_{\text{corr}} \\ y_{\text{corr}} \end{bmatrix} = \mathbf{W} \begin{bmatrix} 1 & x & y & x^2 & y^2 & x y \end{bmatrix}^T$$
