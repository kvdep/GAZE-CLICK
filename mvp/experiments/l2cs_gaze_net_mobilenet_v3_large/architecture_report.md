# Автоматический архитектурный отчет: l2cs_gaze_net

## 1. Сводные метрики вычислительной сложности

| Метрика | Значение | Описание |
| :--- | :--- | :--- |
| **Архитектура** | `l2cs_gaze_net` | Идентификатор архитектуры |
| **Бэкбон** | `mobilenet_v3_large` | Сверточный экстрактор признаков |
| **Число бинов (K)** | `90` | Дискретизация углового пространства |
| **Всего параметров** | `4,245,914` | Суммарное число параметров |
| **Обучаемые параметры** | `4,245,914` | Параметры с градиентом |
| **Объем весов в памяти** | `16.292 MB` | Память в формате float32 |
| **Вычислительная сложность** | `2.1535 GFLOPs` | Число операций с плавающей точкой |
| **Средняя задержка (Mean)** | `17.167 ms` | Устройство: `cuda:0` |
| **95-й перцентиль задержки (P95)** | `21.333 ms` | Гарантированное время кадра |
| **Пропускная способность** | `58.3 FPS` | Кадров в секунду |

## 2. Математическая декомпозиция слоев

1. **Извлечение признаков:** $\mathbf{F} = \operatorname{Backbone}(\mathbf{I}) \in \mathbb{R}^{C \times H \times W}$.
2. **Глобальное усреднение:** $\mathbf{f} = \operatorname{AdaptiveAvgPool}(\mathbf{F}) \in \mathbb{R}^{D}$, где $D = 960$.
3. **Классификационные логиты тангажа и рыскания:**
   $$\mathbf{z}_{\theta} = \mathbf{W}_{\theta} \mathbf{f} + \mathbf{b}_{\theta} \in \mathbb{R}^K, \quad \mathbf{z}_{\phi} = \mathbf{W}_{\phi} \mathbf{f} + \mathbf{b}_{\phi} \in \mathbb{R}^K$$
4. **Распределение вероятностей (Softmax):**
   $$p_k = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}, \quad k \in \{1, \dots, K\}.$$
5. **Математическое ожидание углов (Регрессия):**
   $$\hat{\theta} = \sum_{k=1}^K p_k^\theta c_k, \quad \hat{\phi} = \sum_{k=1}^K p_k^\phi c_k.$$
6. **Единичный 3D-вектор направления взгляда:**
   $$\hat{\mathbf{g}} = \begin{bmatrix} -\cos\hat{\theta}\sin\hat{\phi} \\ -\sin\hat{\theta} \\ -\cos\hat{\theta}\cos\hat{\phi} \end{bmatrix}, \quad \|\hat{\mathbf{g}}\|_2 = 1.$$

## 3. Граф вычислительного конвейера (Mermaid)

```mermaid
graph TD
    A["Входной кадр (B, 3, 224, 224)"] --> B["Backbone Features"]
    B --> C["AdaptiveAvgPool (B, 960)"]
    C --> D["Head Pitch Bins (B, 90)"]
    C --> E["Head Yaw Bins (B, 90)"]
    D --> F["Softmax Expectation -> Pitch theta"]
    E --> G["Softmax Expectation -> Yaw phi"]
    F --> H["3D Vector Transform -> g (B, 3)"]
    G --> H
```

## 4. Послойный профиль архитектуры

| Слой | Тип | Размерность входа | Размерность выхода | Параметры | FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `feature_extractor.0.0` | `Conv2d` | `[1, 3, 224, 224]` | `[1, 16, 112, 112]` | 432 | 10,838,016 |
| `feature_extractor.0.1` | `BatchNorm2d` | `[1, 16, 112, 112]` | `[1, 16, 112, 112]` | 32 | 0 |
| `feature_extractor.0.2` | `Hardswish` | `[1, 16, 112, 112]` | `[1, 16, 112, 112]` | 0 | 0 |
| `feature_extractor.1.block.0.0` | `Conv2d` | `[1, 16, 112, 112]` | `[1, 16, 112, 112]` | 144 | 3,612,672 |
| `feature_extractor.1.block.0.1` | `BatchNorm2d` | `[1, 16, 112, 112]` | `[1, 16, 112, 112]` | 32 | 0 |
| `feature_extractor.1.block.0.2` | `ReLU` | `[1, 16, 112, 112]` | `[1, 16, 112, 112]` | 0 | 0 |
| `feature_extractor.1.block.1.0` | `Conv2d` | `[1, 16, 112, 112]` | `[1, 16, 112, 112]` | 256 | 6,422,528 |
| `feature_extractor.1.block.1.1` | `BatchNorm2d` | `[1, 16, 112, 112]` | `[1, 16, 112, 112]` | 32 | 0 |
| `feature_extractor.2.block.0.0` | `Conv2d` | `[1, 16, 112, 112]` | `[1, 64, 112, 112]` | 1,024 | 25,690,112 |
| `feature_extractor.2.block.0.1` | `BatchNorm2d` | `[1, 64, 112, 112]` | `[1, 64, 112, 112]` | 128 | 0 |
| `feature_extractor.2.block.0.2` | `ReLU` | `[1, 64, 112, 112]` | `[1, 64, 112, 112]` | 0 | 0 |
| `feature_extractor.2.block.1.0` | `Conv2d` | `[1, 64, 112, 112]` | `[1, 64, 56, 56]` | 576 | 3,612,672 |
| `feature_extractor.2.block.1.1` | `BatchNorm2d` | `[1, 64, 56, 56]` | `[1, 64, 56, 56]` | 128 | 0 |
| `feature_extractor.2.block.1.2` | `ReLU` | `[1, 64, 56, 56]` | `[1, 64, 56, 56]` | 0 | 0 |
| `feature_extractor.2.block.2.0` | `Conv2d` | `[1, 64, 56, 56]` | `[1, 24, 56, 56]` | 1,536 | 9,633,792 |
| `feature_extractor.2.block.2.1` | `BatchNorm2d` | `[1, 24, 56, 56]` | `[1, 24, 56, 56]` | 48 | 0 |
| `feature_extractor.3.block.0.0` | `Conv2d` | `[1, 24, 56, 56]` | `[1, 72, 56, 56]` | 1,728 | 10,838,016 |
| `feature_extractor.3.block.0.1` | `BatchNorm2d` | `[1, 72, 56, 56]` | `[1, 72, 56, 56]` | 144 | 0 |
| `feature_extractor.3.block.0.2` | `ReLU` | `[1, 72, 56, 56]` | `[1, 72, 56, 56]` | 0 | 0 |
| `feature_extractor.3.block.1.0` | `Conv2d` | `[1, 72, 56, 56]` | `[1, 72, 56, 56]` | 648 | 4,064,256 |
| `feature_extractor.3.block.1.1` | `BatchNorm2d` | `[1, 72, 56, 56]` | `[1, 72, 56, 56]` | 144 | 0 |
| `feature_extractor.3.block.1.2` | `ReLU` | `[1, 72, 56, 56]` | `[1, 72, 56, 56]` | 0 | 0 |
| `feature_extractor.3.block.2.0` | `Conv2d` | `[1, 72, 56, 56]` | `[1, 24, 56, 56]` | 1,728 | 10,838,016 |
| `feature_extractor.3.block.2.1` | `BatchNorm2d` | `[1, 24, 56, 56]` | `[1, 24, 56, 56]` | 48 | 0 |
| `feature_extractor.4.block.0.0` | `Conv2d` | `[1, 24, 56, 56]` | `[1, 72, 56, 56]` | 1,728 | 10,838,016 |
| `feature_extractor.4.block.0.1` | `BatchNorm2d` | `[1, 72, 56, 56]` | `[1, 72, 56, 56]` | 144 | 0 |
| `feature_extractor.4.block.0.2` | `ReLU` | `[1, 72, 56, 56]` | `[1, 72, 56, 56]` | 0 | 0 |
| `feature_extractor.4.block.1.0` | `Conv2d` | `[1, 72, 56, 56]` | `[1, 72, 28, 28]` | 1,800 | 2,822,400 |
| `feature_extractor.4.block.1.1` | `BatchNorm2d` | `[1, 72, 28, 28]` | `[1, 72, 28, 28]` | 144 | 0 |
| `feature_extractor.4.block.1.2` | `ReLU` | `[1, 72, 28, 28]` | `[1, 72, 28, 28]` | 0 | 0 |
| `feature_extractor.4.block.2.avgpool` | `AdaptiveAvgPool2d` | `[1, 72, 28, 28]` | `[1, 72, 1, 1]` | 0 | 0 |
| `feature_extractor.4.block.2.fc1` | `Conv2d` | `[1, 72, 1, 1]` | `[1, 24, 1, 1]` | 1,752 | 3,456 |
| `feature_extractor.4.block.2.activation` | `ReLU` | `[1, 24, 1, 1]` | `[1, 24, 1, 1]` | 0 | 0 |
| `feature_extractor.4.block.2.fc2` | `Conv2d` | `[1, 24, 1, 1]` | `[1, 72, 1, 1]` | 1,800 | 3,456 |
| `feature_extractor.4.block.2.scale_activation` | `Hardsigmoid` | `[1, 72, 1, 1]` | `[1, 72, 1, 1]` | 0 | 0 |
| ... | ... | ... | ... | ... | (Всего слоев: 935) |