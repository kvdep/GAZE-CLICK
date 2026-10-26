# Gaze-Click: Система трекинга взгляда для бесконтактного управления курсором ОС

## 1. Назначение проекта

Проект представляет собой научно-исследовательскую и программную разработку выпускной квалификационной работы (ВКР), направленную на создание высокоточной системы бесконтактного позиционирования курсора мыши операционной системы в реальном времени с использованием видеопотока потребительской веб-камеры ноутбука.

---

## 2. Официальный онлайн-лидерборд решений (Google Sheets)

Результаты всех протестированных и исследованных моделей консолидированы в едином онлайн-лидерборде:

- **Таблица Google Sheets:** [https://docs.google.com/spreadsheets/d/1J4LHhZXiZYnBFjymL6ayOt2jAafnt4sISfhsUsNiKLo/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1J4LHhZXiZYnBFjymL6ayOt2jAafnt4sISfhsUsNiKLo/edit?usp=sharing)
- **Локальные файлы синхронизации:**
  - [`docs/leaderboard.csv`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/leaderboard.csv) (UTF-8 BOM для прямого импорта через File $\to$ Import)
  - [`docs/leaderboard.tsv`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/leaderboard.tsv) (Табулированный формат для вставки через буфер обмена в ячейку A1)
- **Скрипты выгрузки и обновления:**
  - [`scripts/export_leaderboard.py`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/scripts/export_leaderboard.py) — генерация CSV и TSV с ранжированием 31 модели.
  - [`scripts/sync_google_sheet_leaderboard.py`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/scripts/sync_google_sheet_leaderboard.py) — копирование TSV в буфер обмена Windows и вебхук-синхронизация.

### Схема колонок лидерборда:
`Ранг | Название модели / Архитектура | Ошибка 1 (Mean deg) | Ошибка 2 (Median deg) | Ошибка 3 (Screen px) | Ошибка 4 (R2) | Ошибка 5 (ETH-XGaze deg) | Ошибка 6 (MPII deg) | Ошибка 7 (Gaze360/EVE deg) | Задержка CPU (мс) | Задержка GPU (мс) | Потребление RAM (МБ) | Потребление VRAM (МБ) | Год создания | Место, где лежат веса | Источник (HF / GitHub / ArXiv)`

### Критерий сортировки:
$$\text{Метрика качества (наименьшая угловая ошибка } \mathcal{E}_{ang}\text{)} \longrightarrow \text{Год создания (2026 } \to 2024\text{)} \longrightarrow \text{Размер обучающего датасета}.$$

---

## 3. Топ-4 архитектуры по результатам эмпирического бенчмарка (9 сессий пользователя, 3344 кадра)

| Ранг | Архитектура | Mean Error (°) | Median (°) | Экранная дельта (px) | $R^2$ Score | $T_{CPU}$ (мс) | $T_{GPU}$ (мс) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **PureGaze (ResNet-50 + ResGazeEs)** | **$1.85^\circ \pm 0.44^\circ$** | **$1.70^\circ$** | **$185.6 \pm 44.4$** | **$+0.821$** | $13.27$ | $3.20$ |
| **2** | **UniGaze-Base (ViT-Base 12L)** | **$1.97^\circ \pm 0.60^\circ$** | **$1.64^\circ$** | **$198.5 \pm 60.4$** | **$+0.792$** | $35.00$ | $11.75$ |
| **3** | **ptgaze ETH-XGaze (ResNet-18)** | **$2.21^\circ \pm 0.34^\circ$** | **$1.92^\circ$** | **$222.7 \pm 34.2$** | **$+0.769$** | **$8.50$** | **$6.51$** |
| **4** | **GazeTR-Hybrid (ResNet-18 + ViT-6L)** | **$2.26^\circ \pm 0.47^\circ$** | **$2.03^\circ$** | **$227.4 \pm 47.0$** | **$+0.764$** | $14.75$ | $11.40$ |

---

## 4. Быстрый запуск

Запуск интерактивного конвейера с калибровкой и графическим интерфейсом:

```powershell
# Лидер по точности: PureGaze (1.85 deg, 185 px)
python mvp/app.py --model_name puregaze --test

# Включение прямого управления курсором мыши Windows:
python mvp/app.py --model_name puregaze --enable-mouse

# Запуск самой быстрой архитектуры: ptgaze ETH-XGaze (6.51 мс)
python mvp/app.py --model_name ptgaze_eth --test

# Запуск трансформера UniGaze (1.97 deg, 11.75 мс GPU)
python mvp/app.py --model_name unigaze --test
```

---

## 5. Документация проекта

- [`docs/00_system_and_agent_guidelines.md`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/00_system_and_agent_guidelines.md) — системные регламенты, архитектурные контракты взаимодействия агентов и протокол ведения лидерборда.
- [`docs/09_ready_models_and_aggregators.md`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/09_ready_models_and_aggregators.md) — каталог готовых моделей SOTA, статистика датасетов и реестр агрегаторов данных.
- [`docs/10_architectural_approaches_gaze_and_facial_gestures.md`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/10_architectural_approaches_gaze_and_facial_gestures.md) — сравнительный анализ 7 архитектурных парадигм и детекции мимических жестов.
- [`docs/11_detailed_model_architectures.md`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/11_detailed_model_architectures.md) — математические спецификации, бэкбоны, тензоры и функции потерь 27 моделей.
- [`docs/models/00_model_discrepancies_and_adapters_report.md`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/docs/models/00_model_discrepancies_and_adapters_report.md) — инженерный отчет по калибровке, анализу адаптеров и бенчмарку 9 сессий.
- [`AGENT_HANDOVER.md`](file:///D:/%D0%9F%D0%A0%D0%9E%D0%A4/%D0%92%D0%A3%D0%97/%D0%92%D0%9A%D0%A0/AGENT_HANDOVER.md) — памятка передачи контекста и регламенты разработки.
