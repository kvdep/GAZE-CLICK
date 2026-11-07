import os

def update_models():
    file_path = r'D:\ПРОФ\ВУЗ\ВКР\docs\models\ready_models_and_aggregators.md'
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\r\n', '\n')

    # 1. Update 2026 section
    target_2026 = "10. **`JiananWang624/head_gaze_analysis`** (GitHub, 2026)\n    - **Репозиторий:** `https://github.com/JiananWang624/head_gaze_analysis`\n    - **Веса:** Google Drive `https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing`.\n    - **Специфика:** Интеграция L2CS-Net и 6DRepNet для совместной оценки позы головы и направления взгляда."
    
    # Let's search loosely by unique header
    header_2026_10 = "10. **`JiananWang624/head_gaze_analysis`**"
    header_2025 = "#### [2025] Модели 2025 года"
    header_2024 = "#### [2024] Модели 2024 года"
    header_rank = "### 2.3. Приоритетный рейтинг моделей"

    assert header_2026_10 in text
    assert header_2025 in text
    assert header_2024 in text
    assert header_rank in text

    # Add after JiananWang
    idx_2025 = text.index(header_2025)
    
    extra_2026 = """11. **`EMC-Gaze` (Equivariant Meta-Calibrated Gaze)** (arXiv:2603.12388, Март 2026)
   - **Препринт:** `https://arxiv.org/abs/2603.12388`
   - **Архитектура:** E(3)-эквивариантный графовый энкодер ландмарок (EGNN) с замкнутым гребневым калибратором (closed-form ridge calibrator) через эпизодическое мета-обучение.
   - **Вход:** 161 ключевая точка MediaPipe FaceMesh без обработки сырого пиксельного изображения лица.
   - **Параметры и производительность:** $1.8\\text{ млн}$ параметров, задержка $14.0\\text{ мс}$ на CPU (браузерный инференс).
   - **Метрика качества:** Угловая ошибка RMSE $\\mathbf{5.79^\\circ \\pm 1.81^\\circ}$ после 9-точечной калибровки на 33 интерактивных сессиях; на статических позах головы — $\\mathbf{2.92^\\circ \\pm 0.75^\\circ}$.
   - **Обучающий датасет:** Сессии веб-камер EyeTrax (33 участника, $\\approx 200\\,000$ кадров).
12. **`Narrow-Range Screen Viewing / PRECOG`** (USC SAIL, arXiv:2610.11036, Октябрь 2026)
   - **Препринт:** `https://arxiv.org/abs/2610.11036`
   - **Архитектура:** Дифференциальная регрессия взгляда (Differential Gaze Estimation) с временным контекстом (TCN / GRU) для длительного взгляда на монитор с компенсацией дрейфа калибровки.
   - **Параметры и задержка:** Бэкбон ResNet-18 ($11.7\\text{M}$), задержка $8.2\\text{ мс}$ на CPU.
   - **Метрика качества:** Рекордная угловая ошибка $\\mathbf{1.598^\\circ}$ на длительном мониторном просмотре.
   - **Обучающий датасет:** PRECOG ($140$ видеозаписей длительных сессий, $190$ часов видео, FLIR Blackfly S 195 Гц + EyeLink 1000 250 Гц).
13. **`FIUD` (Factor-Informed Uncertainty Distillation)** (arXiv:2607.20072, Июль 2026)
   - **Препринт:** `https://arxiv.org/abs/2607.20072`
   - **Архитектура:** Дистилляция представлений взгляда под гетероскедастической неопределенностью и вариацией позы головы.
   - **Метрика качества:** Ошибка $3.78^\\circ$ на ETH-XGaze, $4.22^\\circ$ на MPIIFaceGaze.

"""

    text = text[:idx_2025] + extra_2026 + text[idx_2025:]

    idx_2024 = text.index(header_2024)
    extra_2025 = """9. **`quhongyu/OmniGaze`** (NeurIPS 2025 / arXiv:2510.13660, Октябрь 2025)
   - **Репозиторий:** `https://github.com/quhongyu/OmniGaze`
   - **Препринт:** `https://arxiv.org/abs/2510.13660`
   - **Архитектура:** Полуконтролируемый фреймворк (Semi-supervised Learning) с моделью наград (Reward Model) на базе визуального энкодера и мультимодальной LLM (MLLM) для фильтрации псевдо-разметки неразмеченных лиц.
   - **Параметры:** $86.5\\text{ млн}$ параметров (бэкбон ViT-Base). Задержка: $18.5\\text{ мс}$ на GPU.
   - **Метрика качества:** Превосходит все существующие SOTA на 10-15% в zero-shot кросс-доменном тестировании на 4 неизвестных датасетах.
   - **Обучающий массив:** $2.65\\text{ млн}$ изображений (ETH-XGaze $1.08\\text{M}$, Gaze360 $172\\text{k}$ + $1.4\\text{ млн}$ неразмеченных лиц: VGGFace2, WebFace, VFHQ, CelebA).
10. **`MAGE` (Multi-task Architecture for Gaze Estimation)** (IEEE 2025 / arXiv:2505.16384, Май 2025)
   - **Препринт:** `https://arxiv.org/abs/2505.16384`
   - **Архитектура:** Единый энкодер и 3 специализированных декодера для 6-DoF прогноза: вектор взгляда $\\mathbf{g} \\in \\mathbb{R}^3$, 3D-центр глазного яблока $\\mathbf{o} \\in \\mathbb{R}^3$, экранная точка фиксации PoG $\\mathbf{s} \\in \\mathbb{R}^2$ + модуль быстрой калибровки.
   - **Метрика качества:** Ошибка $3.91^\\circ$ на MPIIFaceGaze, экранная ошибка $1.34\\text{ см}$ на EYEDIAP.
   - **Обучающие датасеты:** MPIIFaceGaze ($45\\text{k}$), EYEDIAP ($94$ сессии).
11. **`MAC-Gaze` (Motion-Aware Continual Calibration)** (arXiv:2505.22769, Май 2025)
   - **Препринт:** `https://arxiv.org/abs/2505.22769`
   - **Архитектура:** Непрерывная калибровка, адаптирующаяся к динамике движения пользователя при ходьбе с использованием синхронизированных данных IMU смартфона и TrueDepth RGB-D сенсора.
   - **Обучающие датасеты:** MotionGaze ($481\\text{k}$ кадров + $800\\text{k}$ записей IMU $100\\text{ Гц}$) и RGBDGaze ($320\\text{k}$ кадров, 45 субъектов).
12. **`Cheng et al. 3D Prior Few-Shot 2D Gaze`** (CVPR 2025 / arXiv:2502.04074, Февраль 2025)
   - **Препринт:** `https://arxiv.org/abs/2502.04074`
   - **Архитектура:** Межзадачный перенос 3D-априорных геометрических представлений (3D Gaze Prior) в 2D-экранные координаты через легковесный адаптер всего по $K \\in \\{1, 3, 5, 9\\}$ калибровочным точкам.
   - **Метрика качества:** Снижение экранной ошибки на $34.8\\%$ по сравнению с прямой регрессией.

"""
    text = text[:idx_2024] + extra_2025 + text[idx_2024:]

    # Add historical seminal models before Section 2.3
    idx_rank = text.index(header_rank)
    seminal_models = """#### [2016–2022] Основополагающие архитектуры (Seminal Baselines)

1. **`Park et al. EVE Spatial-Temporal Baseline (SCPT)`** (ECCV 2020)
   - **Репозиторий:** `https://github.com/swook/EVE`
   - **Веса:** `https://github.com/swook/EVE/releases/tag/v0.0`
   - **Архитектура:** ResNet-18 + пространственно-временная свертка (TCN / Bi-GRU) для непрерывного видеопотока.
   - **Метрика качества:** $\\mathbf{2.49^\\circ}$ на датасете EVE.
   - **Обучающий датасет:** EVE ($12.3\\text{ млн}$ кадров, 54 субъекта).

2. **`NVlabs/few_shot_gaze (FAZE)`** (ICCV 2019)
   - **Репозиторий:** `https://github.com/NVlabs/few_shot_gaze`
   - **Архитектура:** Disentangling Autoencoder с мета-обучением MAML для быстрой адаптации всего по $K=3..9$ кадрам.
   - **Метрика качества:** Снижение ошибки с $5.2^\\circ$ (zero-shot) до $\\mathbf{3.18^\\circ}$ (few-shot).
   - **Обучающий датасет:** GazeCapture ($2.45\\text{ млн}$) + MPIIFaceGaze ($45\\text{k}$).

3. **`CA-Net (Coarse-to-fine Attention Network)`** (AAAI 2020)
   - **Архитектура:** Двухстадийное иерархическое внимание: грубая оценка позы лица + точечное внимание к областям зрачков.
   - **Метрика качества:** $4.15^\\circ$ на MPIIFaceGaze.

4. **`iTracker`** (Krafka et al., CVPR 2016)
   - **Репозиторий:** `https://gazecapture.csail.mit.edu/`
   - **Архитектура:** Четырехпоточная CNN: кроп лица ($224\\times 224$), левый глаз ($224\\times 224$), правый глаз ($224\\times 224$), маска сетки лица ($25\\times 25$).
   - **Метрика качества:** Средняя экранная ошибка $1.71\\text{ см}$ на экранах мобильных телефонов и планшетов.
   - **Обучающий датасет:** GazeCapture ($2.45\\text{ млн}$ кадров).

5. **`Full-Face Spatial Weights CNN`** (Zhang et al., CVPRW 2017)
   - **Архитектура:** Сверточная сеть с пространственной картой весов (Spatial Weighting Layer), подавляющей влияние фона.
   - **Метрика качества:** $4.8^\\circ$ на MPIIFaceGaze.

---

"""
    text = text[:idx_rank] + seminal_models + text[idx_rank:]

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("ready_models_and_aggregators.md updated with all models successfully!")

if __name__ == '__main__':
    update_models()
