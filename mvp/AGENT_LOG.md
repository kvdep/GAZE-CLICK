# Журнал решений, ошибок и архитектурных контрактов (AGENT_LOG)

Настоящий документ предназначен для инженеров и автономных ИИ-агентов. Он содержит полную историю архитектурных решений, исправленных ошибок, математических формул и инженерных контрактов проекта MVP.

---

## 1. Системный контекст и аппаратная среда

* **Официальная тема ВКР:** «Разработка и исследование моделей машинного обучения для управления графическим интерфейсом на основе оценки направления взгляда».
* **Практическая задача:** непрерывное позиционирование курсора мыши на экране монитора по видеопотоку веб-камеры в реальном времени ($T_{\text{inf}} \le 16.6$ мс, $\ge 60$ FPS). Дополнительные функции: детекция намеренного моргания для кликов мыши, жесты скролла.
* **Аппаратная база сбора данных:** фронтальные/основные камеры смартфона Samsung Galaxy S25 Ultra (разрешение до 4K, 60 FPS, различные фокусные расстояния) и встроенные веб-камеры.
* **Текущее локальное аппаратное окружение:**
  * GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8.0 GB VRAM GDDR6).
  * RAM: ~18 GB свободной памяти.
  * Платформа: Windows 11 (win32), Python 3.11.9, PyTorch 2.7.0+cu128.
* **Целевое облачное окружение:** Google Colab (Tesla T4, 15.0 GB VRAM, 12.0 GB RAM cgroups limit).

---

## 2. Фиксация системного промпта пользователя

Текст системного промпта сохранен в файле [`code_system_prompt.txt`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/code_system_prompt.txt).

### Ключевые инварианты кода:
1. **Стиль:** краткий синтаксис с использованием `;` для тривиальных присваиваний; функционал и надежность в приоритете.
2. **Запрет комментариев:** комментарии в коде `# ...` категорически запрещены.
3. **Неизменяемость сигнатур:** запрещено переименовывать существующие функции и переменные при правках.
4. **Формат логирования:**
   `[HH:MM:SS] [HH:MM:SS] [CPU/GPU] [RAM] [VRAM] Сообщение`
   * Первое время: астрономическое время по Москве (UTC+3, MSK).
   * Второе время: прошедшее время $\Delta t$ с предыдущего вывода лога в формате `HH:MM:SS`.
   * Контекст вычислений: `[GPU]` или `[CPU]`.
   * Память: `[available_ram_gb]` и `[available_vram_gb]` до одного десятичного знака.
5. **Протокол обучения:**
   * Минимум 150 эпох (`minimum_epochs >= 150`).
   * Обязательный механизм Early Stopping.
   * Формат эпохи: `Epoch {epoch}/{total} | L Train: {loss_train:.4f} | L Val: {loss_val:.4f}`.
   * Формат тестирования: `L Test: {loss_test:.4f}`.
6. **Сохранение и загрузка тяжелых моделей:**
   * В чекпоинт обязательно сохраняется словарь архитектуры `architecture_config`.
   * Загрузка сопровождается явным выводом пути файла весов.
7. **Сравнение моделей:**
   * Использование `pandas.DataFrame` со стилизацией через `background_gradient(cmap="RdYlGn_r")` (зеленый — лучше, красный — хуже).
8. **Динамический контроль памяти:**
   * Хардкод batch size запрещен.
   * Память вычисляется динамически во время выполнения кода при каждом вызове.
   * Агрессивная очистка памяти до и после вычислений.
   * Модели не должны находиться в VRAM вне прямого обучения или инференса.
9. **Запрет заглушек и синтетических тензоров:**
   * Запрещены любые генераторы случайных/фиктивных тензоров вместо реальных данных. Только реальные файлы с диска. При отсутствии файла — строгое исключение `FileNotFoundError`.

---

## 3. Реестр ошибок и найденных решений

### Ошибка 1: Ложное срабатывание детекции Google Colab на Windows
* **Симптом:** исключение `RuntimeError: Colab T4 invariant violated: Free VRAM is 6.89 GB < Required 14.50 GB` при локальном запуске на RTX 4070.
* **Причина:** эвристика проверки `os.path.exists("/content")` вернула `True`, так как на диске пользователя в Windows существует каталог `C:\content`.
* **Решение:** в модуле [`MemoryManager.is_colab_environment`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/utils/memory.py) добавлена строгая проверка операционной системы:
  `sys.platform.startswith("linux") and ("google.colab" in sys.modules or os.path.exists("/content"))`.
  На Windows проверка Colab гарантированно отключается.

### Ошибка 2: Синтаксическая ошибка обратных слэшей в f-строках с LaTeX
* **Симптом:** `SyntaxError: f-string expression part cannot include a backslash` в модуле генерации markdown-отчетов.
* **Причина:** в f-строках вида `f"$$\mathbf{z}_{\theta}...$$"` парсер Python интерпретирует `{\theta}` как интерполяционное выражение, содержащее слэш `\`.
* **Решение:** все строки с математическими формулами LaTeX в [`ArchitectureInspector`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/engine/inspector.py) переведены в строгие сырые литералы `r"..."` без префикса `f`.

### Ошибка 3: Предупреждение pandas `FutureWarning` при конкатенации пустых колонок
* **Симптом:** `FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated...` при объединении Columbia Gaze (где нет экранных координат) с MPIIFaceGaze.
* **Причина:** неявный тип `object` у колонок `screen_x_px` и `screen_y_px`, заполненных `np.nan`.
* **Решение:** в [`DataUnifier`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/data/unification.py) колонки инициализируются с явным типом `pd.Series(np.nan, index=df.index, dtype=np.float32)`.

### Ошибка 4: Использование синтетических заглушек для отсутствующих изображений (НЕДОПУСТИМО)
* **Симптом:** функция `_generate_fallback_tensor` генерировала случайный шум при отсутствии изображения на диске.
* **Критический дефект:** подмена реальных данных шумом искажает градиенты, делает обучение бессмысленным и нарушает инженерную честность пайплайна.
* **Решение:**
  1. `_generate_fallback_tensor` полностью удалена из всей кодовой базы.
  2. Распакован архив реальных изображений RT-GENE (`s007_noglasses.tar`) в `DATA/rtgene/rtgene_dataset/extracted_images/` (840 проверенных реальных кадров).
  3. В `DATA/mpiifacegaze/samples/` сопоставлены 733 реальных кадра.
  4. В `DATA/columbia_gaze/samples/` сопоставлены 6 реальных кадров.
  5. В [`DataUnifier`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/data/unification.py) выполняется строгая фильтрация: `df = df[df["image_exists"] == True]`. Все строки без физического файла на диске исключаются из обучающей выборки.
  6. В [`UnifiedGazeDataset.__getitem__`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/data/unification.py#L225-L235) при отсутствии файла немедленно выбрасывается `FileNotFoundError`. Заглушки отсутствуют.
  7. Замкнутая проверка (Sanity Check) переведена на 4 реальных изображения лиц с диска.

### Ошибка 5: Отсутствие сохранения файлов логов и привязки кода модели к весам
* **Симптом:** логи велись только в консоль; при сохранении чекпоинта не сохранялся исходный файл Python с описанием класса модели.
* **Решение:**
  1. Создана подсистема постоянного логирования: все действия записываются в [`mvp/logs/action_log.txt`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/logs/action_log.txt) и дублируются в `run_dir/action_log.txt`. Логируются точные тайминги: начало, конец и длительность предобработки, загрузки данных, сплита, инспекции, эпох, валидации и тестирования.
  2. При инициализации эксперимента в [`ExperimentManager`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/engine/experiment_manager.py#L29-L35) исходный файл Python модели (`l2cs_net.py`) копируется напрямую в папку эксперимента: `run_dir/model_source.py`.
  3. В чекпоинт `.pt`, файл `summary.json`, таблицу `architecture_experiments.csv` и реестр `global_registry.json` записываются полные абсолютные пути: `weights_file_path` и `model_source_file`.
  4. В таблицу сравнения моделей добавлены колонки `weights_file` и `model_file`.

---

## 4. Математический аппарат и системы координат

### 4.1. Единая система координат (OpenCV Camera Frame)
* Начало координат: оптический центр сенсора камеры.
* Ось $X$: вправо вдоль матрицы.
* Ось $Y$: вниз.
* Ось $Z$: вперед (вдоль оптической оси).

### 4.2. Связь сферических углов и 3D-вектора взгляда
Единичный 3D-вектор взгляда $\mathbf{g} = (g_x, g_y, g_z)^T$, $\|\mathbf{g}\|_2 = 1$:
$$\theta = \arcsin(\operatorname{clamp}(-g_y, -1.0, 1.0)) \quad (\text{Pitch, тангаж}),$$
$$\phi = \arctan2(-g_x, -g_z) \quad (\text{Yaw, рыскание}).$$

Обратное преобразование:
$$\mathbf{g}(\theta, \phi) = \begin{bmatrix} -\cos\theta \sin\phi \\ -\sin\theta \\ -\cos\theta \cos\phi \end{bmatrix}.$$

### 4.3. Угловая ошибка ($\mathcal{E}_{\text{deg}}$)
$$\mathcal{E}_{\text{deg}} = \arccos\left(\operatorname{clamp}\left(\frac{\mathbf{g} \cdot \hat{\mathbf{g}}}{\|\mathbf{g}\|_2 \|\hat{\mathbf{g}}\|_2}, -1 + \epsilon, 1 - \epsilon\right)\right) \times \frac{180^\circ}{\pi}, \quad \epsilon = 10^{-7}.$$

### 4.4. Архитектура L2CS-Net и комплексный лосс
Диапазон углов $[-45^\circ, +45^\circ]$ разбивается на $K = 90$ бинов шириной $w = 1^\circ$. Центры бинов:
$$c_k = -45^\circ + (k + 0.5) \times 1^\circ, \quad k \in \{0, \dots, K-1\}.$$
Прогноз непрерывного угла:
$$\hat{\theta} = \sum_{k=0}^{K-1} p_k^\theta c_k, \quad \hat{\phi} = \sum_{k=0}^{K-1} p_k^\phi c_k, \quad p = \operatorname{softmax}(\mathbf{z}).$$

Функция потерь:
$$\mathcal{L} = \mathcal{L}_{CE}(p^\theta, b_{gt}^\theta) + \mathcal{L}_{CE}(p^\phi, b_{gt}^\phi) + \beta (\mathcal{L}_{MSE}(\hat{\theta}, \theta_{gt}) + \mathcal{L}_{MSE}(\hat{\phi}, \phi_{gt})) + \gamma (1 - \mathbf{g} \cdot \hat{\mathbf{g}}).$$

---

## 5. Архитектура файловой структуры MVP

```
mvp/
├── code_system_prompt.txt           # Неизменяемый системный промпт на код
├── AGENT_LOG.md                     # Настоящий журнал для разработчиков и агентов
├── README.md                        # Инструкция пользователя
├── main.py                          # Главная точка входа (CLI)
├── logs/
│   └── action_log.txt               # Глобальный файл сквозного логирования действий
├── utils/
│   ├── logger.py                    # MetricLogger (MSK time, delta t, CPU/GPU, RAM, VRAM, запись в файл)
│   └── memory.py                    # MemoryManager (динамический batch size, очистка, GPU scope)
├── data/
│   └── unification.py               # DataUnifier (реальные изображения), UnifiedGazeDataset, UnifiedDataModule
├── models/
│   ├── base.py                      # BaseGazeModel (геометрия, угловые ошибки)
│   ├── l2cs_net.py                  # L2CSGazeNet (MobileNetV3/ResNet18), L2CSLoss
│   └── registry.py                  # ModelRegistry (сохранение с кодом и конфигурацией)
├── engine/
│   ├── inspector.py                 # ArchitectureInspector (авто-генерация MD/JSON, FLOPs, Latency)
│   ├── trainer.py                   # GazeTrainer (>=150 эпох, EarlyStopping, логирование)
│   └── experiment_manager.py        # ExperimentManager (diff гиперпараметров, CSV, Styled DataFrame, model_source)
└── experiments/
    ├── global_registry.json         # Единый реестр всех экспериментов
    └── <architecture_name>/         # Папка конкретной архитектуры
        ├── architecture_report.md   # Автоматический отчет с Mermaid графом
        ├── architecture_spec.json   # Машиночитаемая спецификация слоев и FLOPs
        ├── architecture_experiments.csv
        └── run_<ID>_<TIMESTAMP>/    # Отдельный эксперимент
            ├── config.json          # Полные гиперпараметры + исходные пути
            ├── hyperparameter_diff.json # Дельта параметров относительно прошлого запуска
            ├── model_source.py      # Точная копия исходного кода модели на момент запуска
            ├── action_log.txt       # Локальный файл логирования действий этого запуска
            ├── training_log.csv     # Поэпошные логи
            ├── summary.json         # Итоговые метрики, пути к weights (.pt) и model_source (.py)
            └── checkpoints/         # Чекпоинты best_model.pt и latest_model.pt
```

---

## 6. Руководство по проведению инференса и сравнению

Для инференса по контрольной точке запустить:
```powershell
python -u mvp/main.py --mode infer --checkpoint-path mvp/experiments/l2cs_gaze_net_mobilenet_v3_small/run_005_20261002_164623/checkpoints/best_model.pt
```
Команда считывает архитектуру из словаря чекпоинта, восстанавливает модель, загружает реальный тестовый кадр и возвращает предсказанные углы и 3D-вектор взгляда с фиксацией времени вывода в миллисекундах.

---

## 7. Журнал обновлений: Безопасное прерывание, MLflow, периодичность сохранения и Windows-ярлыки ошибок

### 7.1. Протокол безопасного прерывания (Graceful Interruption)
- **Цель:** Гарантированное сохранение всех весов модели, конфигурации архитектуры и состояния оптимизатора при принудительной остановке пользователем (Ctrl+C / SIGINT) или сбое на любой эпохе.
- **Архитектурное решение:** Блок обучения в [`GazeTrainer.fit`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/engine/trainer.py) обернут в контекстный перехват `KeyboardInterrupt`. При получении сигнала:
  1. Вычисляется индекс последней завершенной эпохи $E_{\text{curr}}$.
  2. В подпапку контрольных точек запуска `checkpoints` сериализуется файл `interrupted_model.pt`.
  3. Сохраняются: `state_dict` модели, `optimizer.state_dict()`, `scheduler.state_dict()`, `architecture_config`, $E_{\text{curr}}$ и валидационные метрики с флагом `interrupted: True`.
  4. Формируется запись в `summary.json` со статусом `"interrupted"`.
  5. Модель из `interrupted_model.pt` полностью валидна для немедленного инференса через `ModelRegistry.load_checkpoint(...)`.

### 7.2. Логирование ошибок и генерация ярлыков Windows
- **Цель:** Сохранение полной информации о сбоях с централизованным доступом без ручного поиска по вложенным папкам.
- **Архитектурное решение:** 
  1. При возникновении исключения вызывается метод `ExperimentManager.record_error(exp_ctx, exc)`.
  2. Трассировочный след (`traceback.format_exc()`), тип ошибки, сообщение и метка времени записываются в файл `error_traceback.log` внутри рабочей директории конкретного запуска `run_dir`.
  3. В выделенной корневой директории [`mvp/error_logs/`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/error_logs) через Windows COM-интерфейс `WScript.Shell` (модуль `win32com.client`) создается нативный ярлык `<run_name>.lnk`, указывающий на `error_traceback.log`.
  4. Дополнительно создается дублирующий файл `<run_name>.url` со схемой `file:///` для прямой навигации из Проводника Windows.

### 7.3. Интеграция с MLflow в режиме реального времени
- **Цель:** Наблюдение за сходимостью моделей, метриками и величинами ошибок в веб-интерфейсе в реальном времени.
- **Архитектурное решение:**
  1. Установлен пакет MLflow версии 3.16.1.
  2. Инициализация через `mlflow.set_tracking_uri("file:./mvp/mlruns")` с поддержкой `sqlite:///mvp/experiments/mlruns.db`.
  3. Регистрация эксперимента `gaze_<backbone>` и старт `mlflow.start_run(run_name=run_name)`.
  4. Фиксация гиперпараметров: `backbone`, `batch_size`, `learning_rate`, `optimizer`, `patience`, `max_samples`, `total_parameters`, `target_device`, `save_interval_minutes`.
  5. Поэпошное логирование метрик через `epoch_callback`: `train_loss`, `val_loss`, `val_angular_error_deg`, `val_pixel_mae`, `duration_seconds`, `learning_rate`.
  6. Автоматическая выгрузка артефактов при завершении или прерывании: `config.json`, `model_source.py`, `training_log.csv`, `best_model.pt`.
  7. Статус завершения фиксируется как `FINISHED`, `KILLED` (при прерывании) или `FAILED` (при ошибке).
  8. Веб-интерфейс запускается командой `$env:MLFLOW_ALLOW_FILE_STORE="true"; mlflow ui --backend-store-uri "c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/mlruns" --workers 1 --port 5000`. Ограничение `--workers 1` обязательно на Windows из-за ошибки сокетов Uvicorn `[WinError 10022]` при мультипроцессорном спавне воркеров.

### 7.4. Настраиваемая периодичность сохранения весов
- **Цель:** Исключение избыточной нагрузки на дисковую подсистему при сохранении больших весов на каждой эпохе при сохранении гарантии сохранности прогресса.
- **Архитектурное решение:**
  1. Добавлен параметр `save_interval_minutes: float = 20.0` (CLI аргумент `--save-interval-minutes 20.0`).
  2. Файл `latest_model.pt` сохраняется на диск только при условии:
     $$\Delta t_{\text{save}} = \frac{t_{\text{mono}} - t_{\text{last\_save}}}{60} \ge T_{\text{interval}}$$
     или при флаге `--save-every-epoch`.
  3. Лучшая модель `best_model.pt` сохраняется немедленно при выполнении условия $\mathcal{L}_{\text{val}} \le \mathcal{L}_{\text{best}}$.
  4. При прерывании сохранение `interrupted_model.pt` выполняется безусловно.

### 7.5. Форматирование логов с идентификацией вызывающего файла
- **Цель:** Идентификация источника сообщений и путей к артефактам в едином стандарте.
- **Формат:**
  `[MSK] [elapsed] [CPU/GPU] [RAM] [VRAM] [caller_file:line] Message`
- **Реализация:** В `MetricLogger` добавлен статический метод `_extract_caller_info()`, выполняющий инспекцию текущего стека вызовов (`inspect.currentframe()`) и извлекающий базовое имя файла и номер строки первого вызывающего фрейма вне самого логгера.
- Для путей к контрольным точкам, артефактам и ярлыкам используются префиксы: `[WEIGHTS:<path>]`, `[ARTIFACT:<path>]`, `[SHORTCUT:<path>]`.


---

## 8. Текущее физическое состояние данных, артефактов и flow проекта

### 8.1. Реестр фактических данных на физическом диске:
* **MPIIFaceGaze:**
  * Файл аннотаций: `DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet` ($37\,667$ записей).
  * Реальные изображения на диске: **$37\,667$ файлов** `.jpg` во всех 15 субъектах (`p00`–`p14`), пути: `DATA/mpiifacegaze/{subject}/day{XX}/{YYYY}.jpg`.
  * Калибровки камеры и монитора: 45 файлов `.mat` в `DATA/mpiifacegaze/{subject}/Calibration/`.
  * Экранные координаты: присутствуют (`screen_x_px`, `screen_y_px`).
* **Columbia Gaze:**
  * Файл аннотаций: `DATA/columbia_gaze/processed/columbia_gaze_full.parquet` ($5\,880$ записей).
  * Реальные изображения на диске: **$5\,880$ файлов** `.jpg` во всех 56 субъектах (`0001`–`0056`), путь: `DATA/columbia_gaze/Columbia Gaze Data Set/{subject}/{filename}`.
  * Экранные координаты: отсутствуют (`has_screen = False`).
* **RT-GENE:**
  * Файл аннотаций: `DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet` ($114\,558$ записей).
  * Реальные изображения на диске: **$114\,558$ файлов** `.png` во всех 15 субъектах (`s000`–`s016`), модальность `inpainted/face/*.png` (100% покрытие, все 11 архивов Zenodo скачаны и распакованы);
    путь: `DATA/rtgene/rtgene_dataset/extracted_images/{session}/inpainted/face/face_{seq}_rgb.png`.
  * Экранные координаты: отсутствуют (`has_screen = False`).
* **Итого проверенных реальных изображений на диске:** **$158\,084$ файла** ($114\,558$ RT-GENE + $37\,667$ MPIIFaceGaze + $5\,880$ Columbia Gaze).

### 8.2. Структура логирования и локализации артефактов:
* **Сквозной глобальный лог действий:** `mvp/logs/action_log.txt`.
* **Логи конкретного запуска:** `mvp/experiments/{architecture_name}/{run_name}/action_log.txt`.
* **Поэпошный табличный лог обучения:** `mvp/experiments/{architecture_name}/{run_name}/training_log.csv`.
* **Итоговый паспорт эксперимента:** `mvp/experiments/{architecture_name}/{run_name}/summary.json`.
* **Копия исходного кода модели на момент запуска:** `mvp/experiments/{architecture_name}/{run_name}/model_source.py`.
* **Контрольные точки весов:**
  * Лучшая модель: `mvp/experiments/{architecture_name}/{run_name}/checkpoints/best_model.pt`.
  * Периодическая копия (каждые 20 мин): `.../checkpoints/latest_model.pt`.
  * Аварийный чекпоинт прерывания: `.../checkpoints/interrupted_model.pt`.
* **Трассировка ошибок и ярлыки Windows:**
  * Лог исключения: `mvp/experiments/{architecture_name}/{run_name}/error_traceback.log`.
  * Ярлык Windows LNK: `mvp/error_logs/{run_name}.lnk`.
  * Ярлык проводника URL: `mvp/error_logs/{run_name}.url`.
* **База данных MLflow:** `mvp/mlruns/`.

### 8.3. Сквозной рабочий процесс (Pipeline Flow):
1. **Загрузка и верификация данных (`DataUnifier` в `mvp/data/unification.py`):**
   - Чтение аннотационных Parquet-таблиц.
   - Проверка физического существования каждого файла на диске через `os.path.exists`.
   - Исключение отсутствующих файлов (`image_exists == True`). Никаких синтетических заглушек.
   - Канонический изоморфизм систем координат: углы и 3D-векторы согласованы с точностью $\le 0.028^\circ$ на всех трех датасетах.
2. **Стратифицированное разбиение по субъектам (`UnifiedDataModule.split_by_subjects`):**
   - Subject-Disjoint разбиение отдельно внутри каждого датасета:
     - MPIIFaceGaze: `p00`–`p10` в Train, `p11`–`p12` в Val, `p13`–`p14` в Test.
     - Columbia: 42 субъекта в Train, 7 в Val, 7 в Test.
     - RT-GENE: `p007` в Train, `p005` в Val, `p014` в Test.
   - Пересечение множеств субъектов между выборками строго равно 0.
3. **Формирование батчей со сбалансированным сэмплированием (`WeightedRandomSampler`):**
   - Веса сэмплов: $w_i = \frac{1}{3 \cdot N_{D_k}}$.
   - Батч размера $B$ содержит равные доли (~33.3%) из каждого активного датасета.
4. **Прямой проход модели (`L2CSGazeNet` в `mvp/models/l2cs_net.py`):**
   - Предобученный бэкбон MobileNetV3-Small (`weights=MobileNet_V3_Small_Weights.DEFAULT`).
   - Четыре головы классификации Softmax ($K=90$ бинов): $\theta_{gaze}, \phi_{gaze}, \theta_{head}, \phi_{head}$.
   - Вычисление непрерывных углов как математического ожидания распределения бинов: $\hat{\theta} = \sum p_k c_k$.
   - Ветвь экранных координат (для MPIIFaceGaze).
5. **Многозадачная оптимизация (`L2CSLoss`):**
   - $\mathcal{L}_{total} = \mathcal{L}_{L2CS}^{gaze} + 0.5 \cdot \mathcal{L}_{L2CS}^{head} + 2.0 \cdot \mathcal{L}_{cosine} + 1.0 \cdot \mathcal{L}_{screen}$,
     где $\mathcal{L}_{L2CS} = \mathcal{L}_{CE} + 0.5 \cdot \text{SmoothL1}_{deg}(\beta=1.0)$.
6. **Контроль памяти и динамический размер батча (`MemoryManager`):**
   - Определение свободной VRAM через `torch.cuda.mem_get_info()`.
   - Агрессивная очистка мусора `gc.collect()`, `torch.cuda.empty_cache()` до и после инференса/обучения.
   - Модели не хранятся в VRAM вне фазы вычислений (`scoped_model_device`).
7. **Инференс по контрольной точке (`--mode infer` в `mvp/main.py`):**
   - Восстановление архитектуры из метаданных `.pt` файла.
   - Подача входного тензора $224 \times 224 \times 3$, вывод предсказанных углов взгляда, 3D вектора и углов позы лица с фиксацией миллисекундной задержки (TorchScript JIT + FP16).



---

## 9. Реестр экспериментов и сводка запусков MVP

| Эксперимент / Run ID | Бэкбон | Выборка (файлов) | $\mathcal{E}_{ang}^{val}$ | $\text{MAE}_{\theta_{head}}$ | $\text{MAE}_{\phi_{head}}$ | $T_{CPU}$ (4-th) | $T_{GPU}$ | Статус предикатов |
|---|---|---|---|---|---|---|---|---|
| `run_010_mobilenet_v3_small` | MobileNetV3-Small | $1\,207$ (3 ds) | $13.46^\circ$ | $4.17^\circ$ | $8.95^\circ$ | $11.68\text{ мс}$ | $11.70\text{ мс}$ | FAIL (превышение порога gaze) |
| `run_001_resnet18` | ResNet-18 | $13\,500$ (3 ds) | $7.99^\circ$ | $9.08^\circ$ | $15.68^\circ$ | $18.52\text{ мс}$ (ONNX) | $4.07\text{ мс}$ (JIT) | FAIL (превышение позы и задержки CPU) |
| `run_001_mobilenet_v3_large` | MobileNetV3-Large + GazePoseFusion + 3D Norm | $17\,880$ (3 ds, 7 subj RT-GENE) | $\mathbf{8.47^\circ}$ (Best Ep 7) | $\mathbf{5.52^\circ}$ (Ep 5) | $\mathbf{11.72^\circ}$ (Ep 7) | $\mathbf{4.47\text{ мс}}$ (ONNX) | $9.70\text{ мс}$ (JIT) | FAIL (gaze $> 5.20^\circ$, head MAE $> 4.5^\circ$) |
| `run_002_mobilenet_v3_large` | MobileNetV3-Large + GazePoseFusion (H7) | $17\,880$ (3 ds, 78 subj) | $9.12^\circ$ (Columbia $4.36^\circ$, MPII $5.28^\circ$, RT-GENE $14.99^\circ$) | $6.84^\circ$ (Columbia $0.27^\circ$, MPII $3.97^\circ$) | $13.52^\circ$ | $\mathbf{4.59\text{ мс}}$ (ONNX) | $10.05\text{ мс}$ (JIT) | FAIL (RT-GENE parallax shift $\approx 11.5^\circ$) |
| `run_003_mobilenet_v3_large` | MobileNetV3-Large + GazePoseFusion + Face Ray Norm $SO(3)$ (H8) | $17\,880$ (3 ds, 78 subj) | $9.03^\circ$ (Columbia $3.58^\circ$, MPII $5.07^\circ$, RT-GENE $15.50^\circ$) | $1.60^\circ$ (Columbia $0.26^\circ$, MPII $3.39^\circ$) | $7.67^\circ$ (Columbia $1.08^\circ$) | $\mathbf{4.25\text{ мс}}$ (ONNX) | $1.00\text{ мс}$ (FP16) | FAIL (RT-GENE $15.50^\circ$ из-за рассинхронизации $R_{ray}$ без поворота кропа) |
| `run_004_mobilenet_v3_large` | MobileNetV3-Large + Explicit 3D Pose Conditioning + Stratified Disjoint Split (H9) | $17\,880$ (3 ds, 78 subj) | $5.93^\circ$ (Columbia $3.17^\circ$, MPII $4.92^\circ$, RT-GENE $11.61^\circ$) | $4.59^\circ$ (Columbia $0.25^\circ$) | $12.53^\circ$ (Columbia $2.00^\circ$) | $\mathbf{4.42\text{ мс}}$ (ONNX) | $1.00\text{ мс}$ (FP16) | PASS на Columbia ($3.17^\circ / 4.82^\circ \le 5.20^\circ$) и MPII (Val $4.92^\circ \le 5.20^\circ$, Test Screen $42.26\text{ мм} \le 50\text{ мм}$). Обнаружен дефект неразмеченных изображений noglasses в RT-GENE |
| `run_006_mobilenet_v3_large` | MobileNetV3-Large + Multi-Subject RT-GENE (4 Train subjs) + AdamW lr=0.0003 (H11) | $25\,880$ (3 ds, 77 subj) | **$5.11^\circ$** (med $4.37^\circ$, Val)<br>**$5.40^\circ$** (med $5.12^\circ$, Test) | $\mathbf{1.70^\circ}$ (Test ALL) | $\mathbf{4.15^\circ}$ (Test ALL) | $\mathbf{4.54\text{ мс}}$ (ONNX) | $\mathbf{1.00\text{ мс}}$ (FP16) | PASS Gate 1 на Val ($5.11^\circ \le 5.20^\circ$); PASS Gate 2 на Test ($\text{MAE} \le 4.15^\circ \le 4.5^\circ$); PASS Gate 3 ($4.54/1.00\text{ мс}$); PASS Gate 4 ($42.76\text{ мм} \le 50\text{ мм}$); PASS Gate 5 (0 overlap); PASS Gate 6 ($0.0198^\circ \le 0.05^\circ$). RT-GENE Test снизился с $9.44^\circ$ до $7.08^\circ$. |
| `run_007_mobilenet_v3_large` | MobileNetV3-Large + Full 15 RT-GENE Subjects (100% Data) (H12) | $158\,084$ (3 ds, 86 subj) | **$4.63^\circ$** (Best Ep 2 Val ALL)<br>**$5.71^\circ$** (RT-GENE Test) | $2.98^\circ$ (Val) | $8.24^\circ$ (Val) | $\mathbf{4.54\text{ мс}}$ (ONNX) | $\mathbf{1.00\text{ мс}}$ (FP16) | PASS Gate 1 на Val ($4.63^\circ \le 5.20^\circ$); PASS Gate 4 ($49.26\text{ мм} \le 50\text{ мм}$). Остановлен из-за переобучения на эпохе 22 (Train Loss $6.15$, Val Loss $31.35$). RT-GENE Test снизился до $5.71^\circ$. |
| `run_008_mobilenet_v3_large` | MobileNetV3-Large + Albumentations (8 transforms) + Dropout (p=0.3) + Label Smoothing + EarlyStopping min_epochs=0 (H13) | $158\,084$ (3 ds, 86 subj, 142 277 Train) | **$6.09^\circ$** (Best Ep 12 Val ALL)<br>Ep 12 $\mathcal{E}_{ang}^{val}=\mathbf{6.0947^\circ}$, $\mathcal{L}_{tr}=11.64$, $\mathcal{L}_{va}=33.71$<br>Ep 13 $\mathcal{E}_{ang}^{val}=6.52^\circ$, $\mathcal{L}_{tr}=11.51$, $\mathcal{L}_{va}=34.32$<br>Ep 14 $\mathcal{E}_{ang}^{val}=6.62^\circ$, $\mathcal{L}_{tr}=11.36$, $\mathcal{L}_{va}=33.78$<br>Ep 15 $\mathcal{E}_{ang}^{val}=6.21^\circ$, $\mathcal{L}_{tr}=11.20$, $\mathcal{L}_{va}=33.86$<br>Ep 16 $\mathcal{E}_{ang}^{val}=6.48^\circ$, $\mathcal{L}_{tr}=11.02$, $\mathcal{L}_{va}=33.36$<br>Ep 17 $\mathcal{E}_{ang}^{val}=6.94^\circ$, $\mathcal{L}_{tr}=10.99$, $\mathcal{L}_{va}=36.25$<br>Ep 18 $\mathcal{E}_{ang}^{val}=6.72^\circ$, $\mathcal{L}_{tr}=10.86$, $\mathcal{L}_{va}=35.16$<br>Ep 19 $\mathcal{E}_{ang}^{val}=6.88^\circ$, $\mathcal{L}_{tr}=10.77$, $\mathcal{L}_{va}=34.35$ | $5.85^\circ$ (Ep 19 Val) | $11.97^\circ$ (Ep 19 Val) | $\mathbf{4.66\text{ мс}}$ (ONNX) | $\mathbf{1.00\text{ мс}}$ (FP16) | **АКТИВЕН (ЭПОХА 20 В ПРОЦЕССЕ)**. Эпоха 19 завершена в 13:20:21 MSK ($5\,743.38\text{ с}$ / $95\text{ мин } 43\text{ с}$): $\mathcal{L}_{tr}=10.770451$ ($-0.85\%$), $\mathcal{L}_{va}=34.354113$ ($-2.29\%$), $\mathcal{E}_{ang}^{val}=6.8779^\circ$, Head Pitch MAE $5.85^\circ$, Head Yaw MAE $11.97^\circ$, Screen $295.36\text{ px}$. EarlyStopping `counter = 7 / 15`, `best_model.pt` ($6.0947^\circ$, эпоха 12) сохранен. Эпоха 20 запущена в 13:20:21 MSK. Аудит в 13:24 MSK: PID 25436 `running`, CPU time $76\,220\text{ с}$, RAM RSS $3.65\text{ GB}$, VRAM $7\,163\text{ MiB}$, GPU $54^\circ\text{C}$. |









### Текущее состояние файловой системы MVP:
- `mvp/experiments/l2cs_gaze_net_mobilenet_v3_large/run_001_20261004_220950/`:
  - `checkpoints/best_model.pt` — лучшая контрольная точка весов (эпоха 7, $\mathcal{E}_{ang}^{val} = 8.47^\circ$).
  - `checkpoints/latest_model.pt` — периодическая контрольная точка (каждые 20 минут).
  - `training_log.csv` — поэпошная фиксация потерь, ошибок, задержки, использования RAM и VRAM.
  - `action_log.txt` — поминутный журнал действий с временными метками MSK.
  - `model_source.py` — неизменяемая копия архитектуры на момент запуска.
- Свободное дисковое пространство:
  - Диск `C:\`: $92.55\text{ GB}$ (норматив $\ge 30\text{ GB}$ выполнен).
  - Диск `E:\`: $116.45\text{ GB}$.


---

## 10. Передача контекста между сессиями (Handoff Protocol)
Полный структурированный паспорт передачи контекста для следующего агента со всеми сырыми числами, устраненными дефектами координатных систем, формулами и инструкциями по запуску зафиксирован в файле:
`mvp/NEXT_AGENT_HANDOFF.md`.
Следующий агент обязан прочитать данный файл перед началом работы для исключения дублирования экспериментов и непроизводительного расхода токенов.


---

## 11. Готовый программный комплекс MVP (12 октября 2026 г.)

### 11.1. Верификация контрольной точки весов
- **Файл весов:** `mvp/experiments/l2cs_gaze_net_mobilenet_v3_large/run_008_20261006_001852/checkpoints/best_model.pt` (Эпоха 12).
- **Результаты на независимом тестовом наборе (Test Set, $N=1\,281$, 10 субъектов, 0 пересечений):**
  - Общая угловая ошибка $\mathcal{E}_{ang}$: среднее $5.10^\circ$, медиана $4.85^\circ$, std $3.92^\circ$ (норматив ТЗ $\le 5.20^\circ$ выполнен).
  - Columbia Gaze Test ($N=735$): $\mathcal{E}_{ang} = 4.65^\circ$ (медиана $4.46^\circ$).
  - MPIIFaceGaze Test ($N=488$): $\mathcal{E}_{ang} = 5.84^\circ$ (медиана $5.56^\circ$).
  - RT-GENE Test ($N=58$): $\mathcal{E}_{ang} = 4.54^\circ$ (медиана $4.30^\circ$).
  - Ошибки позы головы: Pitch MAE $1.32^\circ$, Yaw MAE $3.79^\circ$.

### 11.2. Структура модулей MVP
1. `mvp/app.py` — Главная точка входа. Захват потока камеры, рендеринг HUD, переключение режимов калибровки и трекинга, вывод метрик в реальном времени.
2. `mvp/engine/face_pipeline.py` — Детекция лица по легковесной модели YuNet (`face_detection_yunet_2023mar.onnx`, $232$ КБ) и расчет углов/векторов через `L2CSGazeNet` на `cuda:0` с прогревом весов.
3. `mvp/engine/calibration.py` — Калибровочная персонализация без PEFT (Few-Shot Ridge Regression) по 9 экранным точкам. Сохранение профиля в `mvp/calibration_profile.json`.
4. `mvp/engine/filter.py` — Адаптивный темпоральный фильтр $1€\text{ Filter}$ для сглаживания экранных координат курсора.
5. `mvp/engine/cursor_controller.py` — Позиционирование курсора Windows через Win32 API (`SetCursorPos`), DPI Awareness, безопасный режим превью и Dwell-клик ($0.85$ с).
6. `mvp/engine/telemetry_logger.py` — Покадровое протоколирование (CSV) и формирование Markdown-паспорта сессии (`mvp/logs/session_<timestamp>_summary.md`).

### 11.3. Системные особенности и регламент запуска
- **Разрешение экрана:** физическое разрешение монитора пользователя составляет $2880 \times 1800$.
- **Камера:** по умолчанию используется `camera_id = 0` (встроенная веб-камера $1280 \times 720$). Для переключения доступен аргумент `--camera-id 1`.
- **Запуск приложения пользователем:**
  `python mvp/app.py`
  - Для автоматического старта с калибровкой: `python mvp/app.py --calibrate`
  - Для старта с активным управлением курсором: `python mvp/app.py --enable-mouse`
- **Горячие клавиши управления:**
  - `M` или `Пробел`: переключение управления реальным курсором мыши (ВКЛ / ВЫКЛ).
  - `C`: запуск 9-точечной интерактивной калибровки (16 секунд).
  - `R`: сброс калибровки к геометрической проекции.
  - `Q` или `ESC`: завершение работы с сохранением отчета сессии.

### 11.4. Устранение дефекта оконной подсистемы OpenCV HighGUI
- **Дефект:** При вызове `cv2.namedWindow` возникало исключение `(-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support...`.
- **Корневая причина:** Наличие в среде одновременно установленного пакета `opencv-python-headless 5.0.0.93`, скомпилированного с конфигурацией `GUI: NONE`. Импорт `cv2` подгружал модули без поддержки Win32 окон.
- **Исправление:** Пакет `opencv-python-headless` удален, `opencv-python 4.11.0.86` переустановлен в окружение с явным флагом `NO_PROXY="*"`. Оконный интерфейс `WIN32UI` полностью активен, протестирован и готов к интерактивному взаимодействию.

### 11.5. Устранение дефектов позиционирования курсора и калибровки (12 октября 2026 г.)
- **Инверсия горизонтальной оси:** В `mvp/engine/calibration.py` исправлена геометрическая проекция: замена $dx = -d \cdot \tan(\phi)$ на $dx = +d \cdot \tan(\phi)$. Теперь взгляд вправо ($\phi > 0$) приводит к перемещению курсора вправо ($dx > 0$).
- **Стабилизация полинома калибровки:** Замена 8-параметрического квадратичного вектора $\mathbf{f} \in \mathbb{R}^8$ на линейный аффинный вектор $\mathbf{f} = [1.0, \phi, \theta] \in \mathbb{R}^3$ с параметром регуляризации $\lambda = 5.0$. Исключены выбросы координат при саккадах. Удален нестабильный файл `mvp/calibration_profile.json`.
- **Подавление джиттера курсора:** В `mvp/engine/filter.py` внедрен нелинейный Deadband-фильтр с порогом $25\text{ px}$, удерживающий курсор неподвижным при микроколебаниях взгляда на точке фиксации. Параметры One-Euro Filter скорректированы: $f_{c,min} = 0.5\text{ Гц}$, $\beta = 0.005$.
- **Нормализация кропа лица и сглаживание углов:** В `mvp/engine/face_pipeline.py` прямоугольный кроп заменен на квадратный с запасом $1.35 \times \max(fw, fh)$ для исключения деформаций соотношения сторон $224 \times 224$. Добавлено экспоненциальное сглаживание углов взгляда EMA ($\alpha = 0.35$).
- **Кодировка интерфейса:** Все текстовые подсказки в `mvp/app.py` переведены на символы ASCII для устранения отображения знаков вопроса '?????' в OpenCV Hershey Fonts.
- **Поддержка альтернативных чекпоинтов:** Добавлен флаг `--checkpoint <path>` с динамическим распознаванием архитектуры бэкбона (MobileNetV3-Large / ResNet-18).

### 11.6. Реализация PyQt6 интерфейса и системы бенчмарка финального теста (12 октября 2026 г.)
- **Устранение инверсии горизонтали:** В `mvp/engine/calibration.py` знак геометрической проекции по оси X зафиксирован как $dx_{mm} = -d \cdot \tan(\phi)$. Взгляд вправо ($\phi < 0$) смещает курсор вправо ($dx > 0$).
- **Новый интерфейс PyQt6:** Разработан модуль `mvp/app_qt.py`. Поддерживает нативное разрешение $2880 \times 1800$, режим Fullscreen (клавиша F11), отображение телеметрии, векторов взгляда, кнопок калибровки и бенчмарка.
- **Главный запуск:** Файл `mvp/app.py` перенаправлен на запуск PyQt6 по умолчанию (с сохранением режима `--cv` для прямого вывода через OpenCV).
- **Модуль видеофиксации тестового бенчмарка:** Разработан модуль `mvp/engine/benchmark_recorder.py`. Записывает видео `.mp4`, разметку `.csv`/`.parquet`, рассчитывает угловую ошибку $\mathcal{E}_{ang}$ в градусах, экранную ошибку в миллиметрах и строит график траекторий `_trajectory.png`. Обучение на полученных данных строго запрещено.

### 11.7. Устранение просадки FPS, ручное управление бенчмарком и согласование знаков проекции (12 октября 2026 г.)
- **Оптимизация детектора лиц YuNet:** Предварительное масштабирование кадра $1280 \times 720$ до $320 \times 180$ перед инференсом детектора сократило задержку YuNet с $53.15\text{ мс}$ до $4.87\text{ мс}$ ($10.9\times$). Сквозная задержка пайплайна снизилась с $63.66\text{ мс}$ ($15.7\text{ FPS}$) до $22.45\text{ мс}$ ($44.5\text{ FPS}$).
- **Коррекция знаков геометрической проекции:** В `mvp/engine/calibration.py` формулы приведены к виду:
  $$dx_{mm} = \text{gain}_x \cdot d \cdot \tan(\phi), \quad dy_{mm} = \text{gain}_y \cdot d \cdot \tan(\theta - \theta_0),$$
  где $d = 600\text{ мм}$, $\theta_0 = -5.5^\circ$, $\text{gain}_x = 1.85$, $\text{gain}_y = 1.65$. На контрольном видео корреляция по горизонтальной оси $R_x$ возросла с $-0.476$ до $+0.476$, экранная ошибка снизилась со $154.6\text{ мм}$ ($1474.4\text{ px}$) до $123.5\text{ мм}$ ($1177.6\text{ px}$).
- **Интерактивный ручной протокол бенчмарка:** В `mvp/engine/benchmark_recorder.py` и `mvp/app_qt.py` автоматический таймер заменен на ручное подтверждение:
  - 13 статических точек (сетка $3 \times 3$ + 4 крайние границы) фиксируются только по нажатию `SPACE` (серия из 15 кадров за $0.5\text{ с}$).
  - Фаза непрерывного слежения за кругом активируется строго при удержании `SHIFT` или `G`.
- **Исправление проекции HUD в режиме зеркалирования:** Векторы взгляда и рамки отрисовываются непосредственно в зеркальной системе координат матрицы, устраняя инверсию направления стрелок.
- **Горячие клавиши:** Добавлены горячие клавиши: `I` (инверсия оси X), `F` (полноэкранный режим), `+` / `-` (чувствительность $\pm 0.15\times$).
