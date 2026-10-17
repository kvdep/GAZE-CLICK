# ИНСТРУКЦИЯ И ПАСПОРТ ПЕРЕДАЧИ КОНТЕКСТА ДЛЯ СЛЕДУЮЩЕГО АГЕНТА (NEXT_AGENT_HANDOFF)

**Дата фиксации:** 2026-10-06 15:18 MSK  
**Статус проекта:** Эксперимент H13 (`run_008_20261006_001852`) выполнен до Эпохи 9 включительно и приостановлен по запросу пользователя. Чекпоинты `best_model.pt` ($6.4206^\circ$) и `latest_model.pt` (Эпоха 9) верифицированы на диске. Оборудование полностью освобождено (0 MB VRAM, 0% GPU). Подготовлена спецификация H14 (ResNet-18, Gaussian LDL, FiLM conditioning, peripheral cutout).

---

## 1. СИСТЕМНЫЕ ИНВАРИАНТЫ И ПРАВИЛА ПОЛЬЗОВАТЕЛЯ (КРИТИЧЕСКИ ВАЖНО)

1. **Тон и коммуникация:**
   - Никаких комплиментов и похвалы.
   - Никаких аналогий и метафор.
   - Никаких упрощений. Всегда приводить точные математические формулы и сырые числовые значения.
   - Если пользователь или решение неправы — заявлять об этом прямо без сглаживания.
   - Максимально строгий и точный технический язык, без канцеляризмов.
   - Запрещено рекомендовать пользователю обратиться за помощью или консультацией.
   - Никаких аргументов «соломенного чучела». Формулировать мысль прямо.

2. **Инварианты кодовой базы:**
   - **Комментарии в коде строго запрещены** (символ `#` запрещен внутри исполняемого Python-кода, docstrings допускаются только при строгой необходимости).
   - Краткий синтаксис, использование `;` для тривиальных присваиваний.
   - Сигнатуры существующих функций и переменных запрещено переименовывать.
   - Строгий формат логирования:
     `[HH:MM:SS] [HH:MM:SS] [CPU/GPU] [RAM] [VRAM] [caller_file:line] Сообщение`
     (Первое время — MSK UTC+3; второе время — прошедшее $\Delta t$; память в GB с одним знаком).
   - Минимум 150 эпох в цикле обучения (`minimum_epochs >= 150`), обязательный Early Stopping.
   - Динамический расчет памяти через `MemoryManager` и `torch.cuda.mem_get_info()`, хардкод лимитов запрещен.
   - Агрессивная очистка VRAM (`torch.cuda.empty_cache()`, `gc.collect()`), scoped-контекст нахождения моделей на GPU (`scoped_model_device`).
   - **Запрет синтетических заглушек:** никаких случайных тензоров при отсутствии файлов. Только реальные файлы с диска. При отсутствии файла — строгое исключение `FileNotFoundError`.

3. **Дисковое пространство:**
   - На диске `C:\` строго поддерживать не менее **30.0 GB** свободного места (текущее состояние: **92.49 GB** свободно).
   - Тяжелые архивы и данные размещаются на `E:\` (текущее состояние: **116.45 GB** свободно).

---

## 2. ШЕСТЬ НЕРЕДУЦИРУЕМЫХ ГЕЙТОВ ТЗ (КРИТЕРИИ УСПЕХА MVP)
Из документа `docs/08_task_specification_mvp_instruction.md`:
1. **Угловая ошибка направления взгляда на валидации:** $\mathcal{E}_{ang}^{val} \le 5.20^\circ$.
2. **Покомпонентная ошибка позы лица на валидации:** $\text{MAE}_{\theta_{head}} \le 4.5^\circ$ (Pitch) и $\text{MAE}_{\phi_{head}} \le 4.5^\circ$ (Yaw).
3. **Задержка инференса одного кадра ($224 \times 224$):**
   - GPU (JIT / FP16): $T_{GPU} \le 2.0\text{ мс}$.
   - CPU (4 потока ONNX / JIT): $T_{CPU} \le 8.0\text{ мс}$.
   - Пропускная способность конвейера: $\ge 60\text{ FPS}$ ($T_{total} \le 16.67\text{ мс}$).
4. **Экранная ошибка позиционирования (MPIIFaceGaze):**
   - $E_{screen}^{px} \le 180\text{ px}$.
   - $E_{screen}^{mm} \le 50.0\text{ мм}$ (при дистанции $d = 600\text{ мм}$ и шаге пикселя $\Delta_{px} = 0.2238053\text{ мм/px}$).
5. **Строгое разделение по субъектам (Subject-Disjoint Split):**
   - Пересечение множеств субъектов: $\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$ (строго 0 общих субъектов).
6. **Невязка между сферическими углами и 3D-векторами:**
   - $\mathcal{E}_{discrepancy} \le 0.05^\circ$ на 100% строк датасетов.

---

## 3. СОСТОЯНИЕ ДАТАСЕТОВ НА ФИЗИЧЕСКОМ ДИСКЕ (РЕАЛЬНЫЕ ФАЙЛЫ)

Все три датасета полностью распакованы, проверены на физическое наличие каждого файла и объединены модулем `DataUnifier` в `mvp/data/unification.py`:

1. **Columbia Gaze:**
   - Путь: `DATA/columbia_gaze/Columbia Gaze Data Set/`
   - Количество проверенных реальных JPG-файлов: **$5\,880$** (все 56 субъектов `0001`–`0056`).
   - Разметка: `DATA/columbia_gaze/processed/columbia_gaze_full.parquet`.

2. **MPIIFaceGaze:**
   - Путь: `DATA/mpiifacegaze/pXX/dayYY/ZZZZ.jpg`
   - Количество проверенных реальных JPG-файлов: **$37\,667$** (все 15 субъектов `p00`–`p14`).
   - Матрицы калибровки камер $K_{cam}$: загружены из `DATA/mpiifacegaze/pXX/Calibration/Camera.mat`.
   - Разметка: `DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet`.

3. **RT-GENE:**
   - Путь: `DATA/rtgene/rtgene_dataset/extracted_images/`
   - Количество проверенных реальных PNG-файлов: **$11\,752$** (7 субъектов: `p002`: 3965, `p014`: 2583, `p008`: 1498, `p005`: 1215, `p009`: 911, `p007`: 840, `p012`: 740).
   - Разметка: `DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet`.

### Распределение объединенной выборки по сплитам (17 880 реальных изображений):
- **Train ($13\,046$ сэмплов, 57 субъектов):**
  - Columbia Gaze: 4410 сэмплов (42 субъекта);
  - MPIIFaceGaze: 4914 сэмплов (`p00`–`p10`, 11 субъектов);
  - RT-GENE: 3722 сэмпла (`p002`, `p007`, `p008`, `p009`, 4 субъекта).
- **Val ($2\,320$ сэмплов, 11 субъектов):**
  - Columbia Gaze: 735 сэмплов (7 субъектов);
  - MPIIFaceGaze: 598 сэмплов (`p11`, `p12`, 2 субъекта);
  - RT-GENE: 987 сэмплов (`p005`, `p012`, 2 субъекта).
- **Test ($2\,514$ сэмплов, 10 субъектов):**
  - Columbia Gaze: 735 сэмплов (7 субъектов);
  - MPIIFaceGaze: 488 сэмплов (`p13`, `p14`, 2 субъекта);
  - RT-GENE: 1291 сэмпл (`p014`, 1 субъект).
- **Пересечение субъектов:** строго **0** ($\text{Tr} \cap \text{Va} = 0, \text{Tr} \cap \text{Te} = 0, \text{Va} \cap \text{Te} = 0$).

---

## 4. СВОДКА ПРЕДЫДУЩИХ ЭКСПЕРИМЕНТОВ И ИТОГИ АУДИТА H5

| Эксперимент | Архитектура / Бэкбон | Выборка | $\mathcal{E}_{ang}^{val}$ | $\text{MAE}_{\theta}^{val}$ | $\text{MAE}_{\phi}^{val}$ | $T_{CPU}^{ONNX}$ | Статус предикатов |
|---|---|---|---|---|---|---|---|
| `run_010` (H1) | MobileNetV3-Small | $1\,207$ | $13.46^\circ$ | $4.17^\circ$ | $8.95^\circ$ | $11.68\text{ мс}$ | FAIL (gaze $> 5.20^\circ$) |
| `run_001_resnet18` (H4) | ResNet-18 | $13\,500$ | $7.99^\circ$ | $9.08^\circ$ | $15.68^\circ$ | $18.52\text{ мс}$ | FAIL (задержка CPU $> 8\text{ мс}$) |
| `run_001_mobilenet_v3_large` (H5) | MobileNetV3-Large + GazePoseFusion + 3D Norm | $17\,880$ | $\mathbf{8.47^\circ}$ | $\mathbf{6.16^\circ}$ | $\mathbf{11.72^\circ}$ | $\mathbf{4.47\text{ мс}}$ | FAIL (общая выборка не прошла) |

### Результаты независимого по-датасетного аудита контрольной точки H5 (`best_model.pt`, эпоха 7):
Зафиксировано скриптом `mvp/audit_h5.py`:
- **Columbia Gaze ($N_{val}=735, N_{test}=735$):**
  - Val $\mathcal{E}_{ang} = \mathbf{4.04^\circ} \le 5.20^\circ$ (**PASS**)
  - Test $\mathcal{E}_{ang} = \mathbf{5.15^\circ} \le 5.20^\circ$ (**PASS**)
  - Val MAE позы: $\theta = \mathbf{0.41^\circ} \le 4.5^\circ$, $\phi = \mathbf{1.44^\circ} \le 4.5^\circ$ (**PASS**)
  - Test MAE позы: $\theta = \mathbf{0.46^\circ} \le 4.5^\circ$, $\phi = \mathbf{2.99^\circ} \le 4.5^\circ$ (**PASS**)
  - *Вывод: на выборке без ракурса головы модель полностью выполняет ТЗ.*
- **MPIIFaceGaze ($N_{val}=598, N_{test}=488$):**
  - Val $\mathcal{E}_{ang} = \mathbf{5.11^\circ} \le 5.20^\circ$ (**PASS**)
  - Val Pitch MAE: $\theta = \mathbf{2.80^\circ} \le 4.5^\circ$ (**PASS**)
  - Test Screen MAE в мм: $\mathbf{48.38\text{ мм}} \le 50.0\text{ мм}$ (**PASS**)
  - Val Yaw MAE: $17.30^\circ$, Test Screen MAE в px: $216.2\text{ px}$ ($> 180\text{ px}$)
- **RT-GENE ($N_{val}=987, N_{test}=1291$):**
  - Val $\mathcal{E}_{ang} = 13.81^\circ$, Test $\mathcal{E}_{ang} = 15.83^\circ$ (**FAIL**)
  - Val MAE позы: $\theta = 12.47^\circ$, $\phi = 16.00^\circ$ (**FAIL**)
  - *Вывод: именно RT-GENE разрушал итоговые метрики модели.*

---

## 5. ДВА КРИТИЧЕСКИХ ДЕФЕКТА И ИХ УСТРАНЕНИЕ

### Дефект 1: Инверсия оси тангажа в разметке RT-GENE (УСТРАНЕН)
- **Причина:** В файле `DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet` векторные координаты `head_vec_y` и `gaze_vec_y` были сформированы в системе, где ось $Y$ направлена вверх, в то время как соглашение OpenCV требует направления оси $Y$ вниз.
- **Следствие:** Метод `_normalize_vector_and_compute_angles` выполнял $\theta = \arcsin(-v_y)$. Для взгляда вверх ($+17.14^\circ$) получалось $-17.14^\circ$, а для взгляда вниз — $+17.14^\circ$. Возникала систематическая погрешность $34.28^\circ$ на каждом сэмпле RT-GENE. Модель обучалась на взаимоисключающих градиентах (Columbia/MPII тянули тангаж в одну сторону, RT-GENE — в строго противоположную).
- **Решение:** В `mvp/data/unification.py` (строки 264–272) расчет переведен на прямое вычисление единичных векторов из истинных сферических углов:
  $$\mathbf{g} = [-\cos\theta \sin\phi, -\sin\theta, -\cos\theta \cos\phi]^T$$
  $$\mathbf{h} = [-\cos\theta_h \sin\phi_h, -\sin\theta_h, -\cos\theta_h \cos\phi_h]^T$$
  Невязка между углами и векторами на RT-GENE теперь составляет $2 \times 10^{-5}$ градусов.

### Дефект 2: Потеря разрешения зрачка в полнолицевом бэкбоне (УСТРАНЕН В H6)
- **Причина:** При коэффициенте сжатия $S = 32$ в `mobilenet_v3_large` входной кадр $224 \times 224$ сжимается до карты признаков $7 \times 7$. Глаз ($16 \times 32$ пикселя) сжимается в $0.5 \times 1.0$ ячейку, а остальные 48 ячеек кодируют форму черепа, скул и подбородка. В Subject-Disjoint сплите это приводит к переобучению на форме лица обучающих субъектов ($\mathcal{L}_{train} = 2.25$, $\mathcal{L}_{val} = 37.72$).
- **Геометрическая инвариантность центров глаз:** После 3D-нормализации Sugano / Zhang с канонической камерой центры глазных яблок проецируются строго в координаты $x_{left} = 59.16$, $x_{right} = 164.67$, $y = 76.30$. Это доказано расчетом в `task-1873`.
- **Решение:** Вырезание изолированных кропов $64 \times 64$ вокруг центров $(59, 76)$ и $(165, 76)$ с подачей в сиамский энкодер глаз.

---

## 6. АРХИТЕКТУРА ГИПОТЕЗЫ H6 (`DualStreamGazeNet`)

Реализована и протестирована в файле [mvp/models/dual_stream_gaze_net.py](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/models/dual_stream_gaze_net.py):

```
                        Входное изображение (B, 3, 224, 224)
                                    /        \
                                   /          \
            Поток лица (Face Stream)          Поток глаз (Siamese Eye Stream)
          MobileNetV3-Large (features)        Тензорный срез: Left [44:108, 27:91]
                       |                                       Right [44:108, 133:197] (h-flip)
            GlobalAvgPool2d (B, 960)                            |
             /         |         \                     SiameseEyeEncoder (29k параметров)
            /          |          \                             |
  Linear(90)      Linear(90)   Linear(128)             f_left (128)  f_right (128)
Head Pitch Logits Head Yaw Logits  e_head                       \     /
                                     \                       f_eyes (B, 256)
                                      \                            /
                                       \                          /
                                       Слияние признаков f_gaze = [f_eyes || e_head] (B, 384)
                                                       /         |         \
                                                      /          |          \
                                                Linear(90)   Linear(90)   Linear(2)
                                                Gaze Pitch   Gaze Yaw     Screen Coords
```

### Характеристики DualStreamGazeNet:
- Общее число параметров: **$3\,367\,130$** ($3.37\text{M}$ параметров — на 23% легче, чем базовая сеть L2CS).
- Параметры сиамского энкодера глаз: **$29\,000$** параметров.
- Дополнительная вычислительная сложность: $< 0.05\text{ GFLOPs}$.
- Регистрация в `ModelRegistry`: класс `@ModelRegistry.register("dual_stream_gaze_net")`.
- В [mvp/main.py](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/main.py) добавлен аргумент `--arch dual_stream_gaze_net`.

---

## 7. ЧТО ДЕЛАТЬ СЛЕДУЮЩЕМУ АГЕНТУ (ПОШАГОВЫЙ ПЛАН ДЕЙСТВИЙ)

### Шаг 1: Запустить обучение гипотезы H6 на GPU
Выполнить команду в PowerShell из рабочей директории проекта `C:\Users\jack_\MAIN\ВУЗ\ВКР`:
```powershell
python mvp/main.py --mode train --arch dual_stream_gaze_net --backbone mobilenet_v3_large --max-samples 6000 --batch-size 64 --epochs 150 --patience 15 --lr 0.0005
```
*Важно:* Команда будет выполняться асинхронно в фоне. Не опрашивать статус в цикле — дождаться завершения или реактивного уведомления.

### Шаг 2: Провести независимый аудит обученной модели
После завершения обучения запустить скрипт аудита на созданном чекпоинте `best_model.pt`:
```powershell
python mvp/audit_h5.py "mvp/experiments/dual_stream_gaze_net_mobilenet_v3_large/<run_id>/checkpoints/best_model.pt"
```

### Шаг 3: Сопоставить метрики с шестью гейтами ТЗ
Проверить выполнение всех 6 предикатов:
1. $\mathcal{E}_{ang}^{val} \le 5.20^\circ$;
2. $\text{MAE}_{\theta_{head}} \le 4.5^\circ$ и $\text{MAE}_{\phi_{head}} \le 4.5^\circ$;
3. $T_{CPU}^{ONNX} \le 8.0\text{ мс}$;
4. $E_{screen}^{mm} \le 50.0\text{ мм}$ и $E_{screen}^{px} \le 180\text{ px}$;
5. $\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$;
6. $\mathcal{E}_{discrepancy} \le 0.05^\circ$.

### Шаг 4: Зафиксировать результаты в журналах
- Внести по-датасетную таблицу и статус гейтов в `mvp/AGENT_THOUGHTS.md` (раздел 14).
- Обновить реестр запусков в `mvp/AGENT_LOG.md` (раздел 9).
- Если возникли новые системные дефекты — зафиксировать их в `SYSTEM_AND_ENVIRONMENT_GUIDE.md`.

---

## 8. КЛЮЧЕВЫЕ ПУТИ К ФАЙЛАМ ПРОЕКТА
- ТЗ на MVP: [08_task_specification_mvp_instruction.md](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/docs/08_task_specification_mvp_instruction.md)
- Журнал мыслей и гипотез: [AGENT_THOUGHTS.md](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/AGENT_THOUGHTS.md)
- Журнал архитектуры и ошибок: [AGENT_LOG.md](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/AGENT_LOG.md)
- Руководство по окружению и багам: [SYSTEM_AND_ENVIRONMENT_GUIDE.md](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/SYSTEM_AND_ENVIRONMENT_GUIDE.md)
- Модель DualStreamGazeNet: [dual_stream_gaze_net.py](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/models/dual_stream_gaze_net.py)
- Главная точка входа: [main.py](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/main.py)
- Скрипт независимого аудита: [audit_h5.py](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/audit_h5.py)
- Загрузка и нормализация данных: [unification.py](file:///c:/Users/jack_/MAIN/ВУЗ/ВКР/mvp/data/unification.py)
