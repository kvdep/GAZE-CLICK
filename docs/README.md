# Оглавление документации проекта (Master Table of Contents)

Данный документ является единой точкой навигации по всей исследовательской и инженерной документации проекта `GAZE-CLICK`. Документация строго декомпозирована по смысловым каталогам во избежание дублирования информации.

---

## Структура каталогов и правила размещения информации

```
docs/
├── README.md                                   # Настоящее оглавление
├── guidelines/                                 # Системные регламенты, правила агентов, протоколы запретов
│   ├── system_and_agent_guidelines.md          # Системное и техническое руководство по исследованию
│   ├── prohibition_synthetic_metrics.md        # Протокол запрета синтетических данных и поддельных метрик
│   └── meta_loop_verification.md               # Протокол замкнутой верификации мета-лупа
├── specifications/                             # Постановка задачи, гипотезы, метрики, требования ТЗ
│   ├── research_topic_and_scope.md             # Математическая и инженерная постановка темы ВКР
│   ├── metrics_and_losses.md                   # Математические метрики, функции потерь и пороги ТЗ
│   ├── hypotheses_registry.md                  # Реестр научных и инженерных гипотез
│   └── mvp_task_specification.md               # Техническое задание и регламент разработки MVP
├── literature/                                 # Научно-аналитический обзор литературы и открытых проектов
│   ├── sota_review.md                          # Аналитический обзор SOTA 2019–2026 гг.
│   ├── bibliography_papers.md                  # Каталог 40+ ключевых научных публикаций с аннотациями
│   └── github_repositories.md                  # Анализ существующих открытых репозиториев
├── datasets/                                   # Каталоги датасетов, геометрические параметры и загрузка
│   ├── catalog.md                              # Исчерпывающий каталог датасетов (MPII, EVE, Gaze360 и др.)
│   └── registry_and_downloads.md               # Статус локальных загрузок на E:\, хеши и команды скачивания
├── models/                                     # Архитектуры нейросетей, адаптеры и отчеты об ошибках
│   ├── detailed_model_architectures.md         # Детальная спецификация архитектур моделей
│   ├── architectural_approaches_gaze_gestures.md # Спецификация 7 архитектурных подходов (CNN, ViT, GazeTR)
│   ├── adapters_and_discrepancies.md           # Инженерный отчет: аудит адаптеров, расхождения форматов
│   ├── ready_models_and_aggregators.md         # Реестр готовых моделей, весов и агрегаторов
│   ├── autoregressive_gaze_models.md           # Реестр и анализ авторегрессионных моделей (ARGaze, EyeTAG, PRECOG)
│   ├── gazetr/                                 # Кодовая база и чекпоинты GazeTR
│   ├── idiap/                                  # Кодовая база Idiap (Gaze360 LSTM, MTGS)
│   ├── l2cs/                                   # Кодовая база L2CS-Net
│   ├── mobileone/                              # Кодовая база MobileOne / Qualcomm EyeGaze
│   └── puregaze/                               # Кодовая база PureGaze (Cheng et al. / Akgül et al.)
├── benchmarks/                                 # Сводные таблицы точности и лидерборды
│   ├── leaderboard.csv                         # Машиночитаемая таблица лидерборда 19 моделей
│   ├── leaderboard.tsv                         # Таблица лидерборда с табуляцией
│   └── README.md                               # Методология формирования лидерборда и условия тестов
└── papers/                                     # Первоисточники научных статей (PDF, TXT, HTML)
    └── SOTA_GAZE_RESEARCH_AND_SYNTHESIS.md     # Научный анализ и декомпозиция литературы
```

---

## 1. [Регламенты и протоколы (guidelines/)](file:///D:/ПРОФ/ВУЗ/ВКР/docs/guidelines)
В данном каталоге содержатся правила ведения разработки, протоколы верификации и запретов:
* [`system_and_agent_guidelines.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/guidelines/system_and_agent_guidelines.md) — базовые системные правила, стек инструментов, управление памятью и требования к воспроизводимости.
* [`prohibition_synthetic_metrics.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/guidelines/prohibition_synthetic_metrics.md) — регламент полного запрета использования синтетических данных, симулированных метрик и хардкода.
* [`meta_loop_verification.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/guidelines/meta_loop_verification.md) — протокол замкнутого цикла проверки гипотез и запуска бенчмарков.

---

## 2. [Спецификации и математическая база (specifications/)](file:///D:/ПРОФ/ВУЗ/ВКР/docs/specifications)
В данном каталоге содержатся документы, формализующие теоретическую основу ВКР:
* [`research_topic_and_scope.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/specifications/research_topic_and_scope.md) — научная новизна, границы применимости, формулировка темы и ключевые аспекты ВКР.
* [`metrics_and_losses.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/specifications/metrics_and_losses.md) — формулы угловой ошибки $\mathcal{L}_{\text{ang}}$, экранной ошибки $\Delta x$, функций потерь (Pinball, L1, Angular Cosine) и пороговые нормативы ТЗ.
* [`hypotheses_registry.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/specifications/hypotheses_registry.md) — реестр верифицируемых научных и инженерных гипотез (H1–H12).
* [`mvp_task_specification.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/specifications/mvp_task_specification.md) — пошаговое техническое задание на реализацию программного прототипа (MVP).

---

## 3. [Анализ литературы и SOTA (literature/)](file:///D:/ПРОФ/ВУЗ/ВКР/docs/literature)
В данном каталоге собраны аналитические обзоры предметной области:
* [`sota_review.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/literature/sota_review.md) — аналитический разбор передовых подходов в оценке взгляда (2019–2026 гг.).
* [`bibliography_papers.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/literature/bibliography_papers.md) — систематизированный каталог 40+ публикаций с аннотациями, формулами и ссылками на первоисточники.
* [`github_repositories.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/literature/github_repositories.md) — каталог открытых реализаций, архитектур и инструментов оценки взгляда.

---

## 4. [Каталог и реестр датасетов (datasets/)](file:///D:/ПРОФ/ВУЗ/ВКР/docs/datasets)
В данном каталоге содержится информация о наборах данных:
* [`catalog.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/datasets/catalog.md) — детальные паспорта датасетов: MPIIGaze, MPIIFaceGaze, GazeCapture, Columbia Gaze, EYEDIAP, EVE, RT-GENE, ETH-XGaze, Gaze360, OpenEDS. Содержит точные оптические параметры, разметку и ограничения.
* [`registry_and_downloads.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/datasets/registry_and_downloads.md) — статус локального размещения датасетов на диске `E:\`, контрольные суммы, команды `curl` и ссылки на зеркала.

---

## 5. [Архитектуры моделей и адаптеры (models/)](file:///D:/ПРОФ/ВУЗ/ВКР/docs/models)
* [`detailed_model_architectures.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/models/detailed_model_architectures.md) — детальная спецификация архитектур моделей (ранжирование по угловой ошибке, стадии, тензоры, лоссы).
* [`architectural_approaches_gaze_gestures.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/models/architectural_approaches_gaze_gestures.md) — подробные схемы 7 архитектурных подходов (CNN, ViT, GazeTR, MobileOne, Conformer).
* [`adapters_and_discrepancies.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/models/adapters_and_discrepancies.md) — инженерный отчет об аудите адаптеров инференса, расхождениях форматов углов (Pitch/Yaw vs Yaw/Pitch) и нормализации.
* [`ready_models_and_aggregators.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/models/ready_models_and_aggregators.md) — реестр доступных весов, ссылок на HuggingFace, GitHub, ArXiv и агрегаторов.
* [`autoregressive_gaze_models.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/models/autoregressive_gaze_models.md) — реестр и детальный анализ 14 авторегрессионных моделей взгляда и сканирующих траекторий (2016–2026 гг.).
* Подкаталоги моделей (`gazetr/`, `idiap/`, `l2cs/`, `mobileone/`, `puregaze/`) содержат исходный код и сопроводительные README авторов.

---

## 6. [Лидерборды и бенчмарки (benchmarks/)](file:///D:/ПРОФ/ВУЗ/ВКР/docs/benchmarks)
В данном каталоге агрегированы результаты экспериментальных испытаний моделей:
* [`leaderboard.csv`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/benchmarks/leaderboard.csv) / [`leaderboard.tsv`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/benchmarks/leaderboard.tsv) — сводный рейтинг моделей по угловой ошибке, экранной ошибке, коэффициенту детерминации $R^2$, задержке инференса (CPU/GPU) и потреблению памяти.
* [`README.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/benchmarks/README.md) — описание методики проведения замеров, структуры тестовых сессий и критериев ранжирования.

---

## 7. [Первоисточники статей (papers/)](file:///D:/ПРОФ/ВУЗ/ВКР/docs/papers)
Каталог содержит оригинальные статьи в формате PDF, текстовые выжимки TXT, HTML-страницы и сводный аналитический документ [`SOTA_GAZE_RESEARCH_AND_SYNTHESIS.md`](file:///D:/ПРОФ/ВУЗ/ВКР/docs/papers/SOTA_GAZE_RESEARCH_AND_SYNTHESIS.md).
