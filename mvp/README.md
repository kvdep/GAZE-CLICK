# Gaze Tracking GUI Control: MVP Framework

Минимально жизнеспособный продукт (MVP) исследовательской системы оценки направления взгляда для управления графическим интерфейсом операционной системы в реальном времени.

Связанные документы:
- Системный промпт на код: [code_system_prompt.txt](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/code_system_prompt.txt)
- Журнал решений, ошибок и контрактов: [AGENT_LOG.md](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/AGENT_LOG.md)
- Главный журнал действий: [action_log.txt](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/logs/action_log.txt)
- Главная точка входа: [main.py](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/main.py)

---

## 1. Возможности каркаса

1. **Работа строго с реальными изображениями (без заглушек):**
   - Полное исключение любых синтетических тензоров и фиктивных сэмплов.
   - Загружаются реальные кадры с диска: 840 кадров RT-GENE (`extracted_images`), 733 кадра MPIIFaceGaze (`samples`), кадры Columbia Gaze (`samples`).
   - Если изображение отсутствует на диске — сэмпл отбрасывается при валидации; при попытке прямого чтения несуществующего пути выбрасывается `FileNotFoundError`.
2. **Безопасная остановка обучения в любой момент (Graceful Interruption):**
   - Перехват `KeyboardInterrupt` / Ctrl+C и системных прерываний в любой точке цикла обучения.
   - Мгновенная аварийная сериализация весов, конфигурации архитектуры и состояния оптимизатора в файл контрольной точки `interrupted_model.pt` в поддиректории `checkpoints`.
   - Запись статуса `interrupted` в `summary.json`, `architecture_experiments.csv` и реестр экспериментов.
3. **Логирование ошибок и генерация ярлыков Windows:**
   - При сбое или падении процесса обучения полный трассировочный след исключения записывается в `error_traceback.log` в папку эксперимента.
   - В выделенной директории [mvp/error_logs](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/error_logs) создаются Windows-ярлык `.lnk` (через `WScript.Shell`) и файл `.url`, ссылающиеся на конкретный лог ошибки.
4. **Интеграция с MLflow в реальном времени:**
   - Автоматическая передача гиперпараметров (`learning_rate`, `batch_size`, `patience`, `save_interval_minutes`), параметров архитектуры и системного устройства.
   - Поэпошное логирование метрик `train_loss`, `val_loss`, `val_angular_error_deg`, `val_pixel_mae`, `duration_seconds`, `learning_rate`.
   - Автоматическое прикрепление артефактов (`config.json`, `model_source.py`, `training_log.csv`, веса `.pt`).
5. **Настраиваемая периодичность сохранения весов:**
   - По умолчанию сохранение периодических контрольных точек на диск выполняется каждые 20 минут реального времени (`--save-interval-minutes 20.0`), предотвращая избыточный износ накопителя тяжелыми файлами.
   - Лучшая модель `best_model.pt` сохраняется всегда при улучшении ошибки валидации.
   - Доступен флаг `--save-every-epoch` для принудительного сохранения на каждой эпохе.
6. **Сквозное логирование всех действий ([`action_log.txt`](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/logs/action_log.txt)):**
   - Единый формат записи: `[MSK] [elapsed] [CPU/GPU] [RAM] [VRAM] [caller_file:line] Message`.
   - Для путей к контрольным точкам и артефактам используются префиксы: `[WEIGHTS:<path>]`, `[ARTIFACT:<path>]`, `[SHORTCUT:<path>]`.
7. **SOTA-архитектура L2CS-Net:**
   - Сверточный бэкбон (MobileNetV3-Small для CPU/мобильного реалтайма, ResNet-18 для повышенной точности).
   - Двухголовая классификационно-регрессионная схема с $K = 90$ бинами и Softmax-ожиданием.
   - Комплексная функция потерь: CrossEntropy + MSE + Cosine Proximity + Screen Coordinates MSE.
8. **Автоматическое описание и профилирование архитектуры:**
   - Измерение числа параметров, объема весов в памяти, послойных FLOPs и задержки вывода ($T_{\text{mean}}$, $P_{95}$, FPS).
   - Автоматическая генерация отчетов [architecture_report.md](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/experiments/l2cs_gaze_net_mobilenet_v3_small/architecture_report.md) с формулами и Mermaid-графом, а также JSON-спецификаций [architecture_spec.json](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/experiments/l2cs_gaze_net_mobilenet_v3_small/architecture_spec.json).
9. **Динамический контроль памяти:**
   - Расчет размера батча на лету по доступной видеопамяти и оперативной памяти без хардкода.
   - Агрессивная очистка памяти до и после операций.
   - Выгрузка моделей из VRAM вне прямого инференса или прохода обучения.

---

## 2. Команды запуска CLI

Все команды исполняются из корневой директории проекта `c:\Users\jack_\MAIN\ВУЗ\ВКР`:

### 2.1. Замкнутая проверка (Sanity Check) на реальных изображениях
Проверка градиентов и сходимости на 4 реальных кадрах лиц из датасета:
```bash
python -u mvp/main.py --mode sanity_check
```

### 2.2. Автоматическое профилирование архитектуры
Генерация архитектурного описания, расчет FLOPs, задержки и создание Markdown/JSON спецификаций:
```bash
python -u mvp/main.py --mode inspect --backbone mobilenet_v3_small
python -u mvp/main.py --mode inspect --backbone resnet18
```

### 2.3. Запуск эксперимента обучения
Запуск цикла обучения с MLflow, настраиваемым интервалом сохранения весов и возможностью остановки:
```bash
# Стандартный запуск с периодичностью сохранения раз в 20 минут:
python -u mvp/main.py --mode train --backbone mobilenet_v3_small --epochs 150 --patience 15 --save-interval-minutes 20.0

# Запуск с принудительным сохранением весов каждую эпоху:
python -u mvp/main.py --mode train --backbone mobilenet_v3_small --epochs 150 --save-every-epoch
```

### 2.4. Мониторинг в MLflow UI в реальном времени
Запуск веб-интерфейса MLflow в PowerShell (с флагом разрешения файлового бэкенда и ограничением `--workers 1` для предотвращения ошибки сокетов Uvicorn на Windows):
```powershell
$env:MLFLOW_ALLOW_FILE_STORE="true"; mlflow ui --backend-store-uri "c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/mlruns" --workers 1 --port 5000
```
Интерфейс доступен по адресу `http://127.0.0.1:5000`.

### 2.5. Сводное сравнение проведенных экспериментов
Вывод сравнительной таблицы метрик моделей с путями к `.pt` и исходному коду:
```bash
python -u mvp/main.py --mode compare
```

### 2.6. Прямой инференс по контрольной точке
Загрузка модели из `.pt` (включая `interrupted_model.pt`), восстановление архитектуры и прогон на реальном изображении:
```bash
python -u mvp/main.py --mode infer --checkpoint-path mvp/experiments/l2cs_gaze_net_mobilenet_v3_small/run_005_20261002_164623/checkpoints/best_model.pt
```

### 2.7. Объединение и экспорт датасетов
Формирование единого Parquet-файла из выбранных датасетов:
```bash
python -u mvp/main.py --mode unify_data
```
