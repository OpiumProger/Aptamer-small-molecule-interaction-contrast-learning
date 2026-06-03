# Contrastive Learning Model README


Описанные файлы:

- `Contrast Learning.py`
- `DataPrepare.py`
- `Embeddings.py`
- `FinalTrainer.py`
- `Loss.py`
- `Model.py`
- `load_data_and_visual_data.py`
- `wasserstein_utils.py`

## Общая Идея

Цель пайплайна - обучить модель, которая переводит молекулы и аптамеры в общее contrastive-пространство. В этом пространстве:

- interacting / positive пары должны иметь высокую cosine similarity;
- non-interacting / negative пары должны иметь низкую cosine similarity;
- качество модели оценивается по разделению positive и negative пар.

Схематично:

```text
SMILES embedding  -> molecule encoder -> z_molecule
aptamer embedding -> aptamer encoder   -> z_aptamer

cosine(z_molecule, z_positive_aptamer) должен быть высоким
cosine(z_molecule, z_negative_aptamer) должен быть низким
```

## Данные

Основной рабочий CSV:

```text
aptabench_with_embeddings.csv
```

Ожидаемые колонки:

- `sequence` или колонка с названием, содержащим `sequence`;
- `canonical_smiles` или другая колонка со `smiles` в названии;
- `label`;
- `seq_emb_*` - embedding аптамера;
- `smi_emb_*` - embedding молекулы.

Смысл `label`:

- `1` - positive / interacting pair;
- `0` - negative / non-interacting pair.

## Файл `Embeddings.py`

Этот файл отвечает за первичное получение embedding-ов из сырых последовательностей и SMILES.

Основные функции:

- `data_prepare(...)`
- `aptamer_encode(...)`
- `smiles_encode(...)`
- `create_combined_dataset(...)`

### `data_prepare`

Читает исходный CSV, проверяет наличие колонок с последовательностями аптамеров и SMILES, фильтрует некорректные значения:

- `NaN`;
- нестроковые значения;
- пустые строки;
- слишком короткие строки.

Возвращает очищенные последовательности, SMILES и индексы строк, которые прошли фильтр.

### `aptamer_encode`

Кодирует последовательности аптамеров через модель:

```text
AIRI-Institute/gena-lm-bert-base
```

Последовательности приводятся к верхнему регистру, `U` заменяется на `T`.

Embedding получается усреднением hidden states по токенам с учетом attention mask.

### `smiles_encode`

Кодирует SMILES через:

```text
seyonec/ChemBERTa-zinc-base-v1
```

В качестве embedding-а берется CLS-представление:

```python
outputs.last_hidden_state[:, 0, :]
```

### `create_combined_dataset`

Объединяет исходный DataFrame с embedding-ами:

- `seq_emb_0`, `seq_emb_1`, ...
- `smi_emb_0`, `smi_emb_1`, ...

Результат сохраняется в CSV, обычно:

```text
aptabench_with_embeddings.csv
```

## Файл `load_data_and_visual_data.py`

Этот файл отвечает за загрузку подготовленного CSV, базовую оценку модели и визуализации embedding-пространства.

Основные функции:

- `load_data(...)`
- `analyze_results(...)`
- `visualize_embeddings_correct(...)`
- `visualize_embeddings_2d_simple(...)`

### `load_data`

Читает `aptabench_with_embeddings.csv`, находит:

- все `seq_emb_*`;
- все `smi_emb_*`;
- positive строки по `label == 1`;
- negative строки по `label == 0`.

Возвращает четыре массива:

```text
apt_pos
smi_pos
apt_neg
smi_neg
```

А также размерности embedding-ов.

### `analyze_results`

Прогоняет модель по `DataLoader` и считает:

- positive cosine similarities;
- negative cosine similarities;
- accuracy;
- separation;
- mean/std для positive и negative similarity.

Accuracy считается как доля случаев, где positive aptamer оказался ближе к molecule embedding, чем sampled negatives.

### `visualize_embeddings_correct`

Собирает embedding-и:

- molecule anchors;
- positive aptamers;
- hard negative aptamers.

Затем строит:

- t-SNE;
- UMAP;
- PCA 2D;
- распределения similarity;
- boxplot similarity;
- PCA 3D.

Сохраняет изображение, например:

```text
embedding_visualization_correct.png
```

## Файл `DataPrepare.py`

Этот файл содержит dataset-класс:

```python
FinalContrastiveDataset
```

Он строит структуру данных для contrastive learning.

### Внутренняя Логика

При инициализации создаются словари:

```python
self.smi_to_pos
self.smi_to_neg
```

Ключ - embedding молекулы, преобразованный в `tuple`.

Значения:

- список positive aptamer embeddings для этой молекулы;
- список negative aptamer embeddings для этой молекулы.

Dataset хранит список уникальных молекул:

```python
self.smis
```

### `__getitem__`

Для одной молекулы возвращается:

```python
{
    'anchor_smi': anchor_smi,
    'positive_apts': positive_apts,
    'negative_apts': negative_apts
}
```

Где:

- `anchor_smi` - embedding молекулы;
- `positive_apts` - случайный positive aptamer для этой молекулы;
- `negative_apts` - несколько negative aptamers для этой же молекулы.

Количество negative aptamers задается параметром:

```python
negative_ratio=3
```

Если для молекулы нет negative aptamers, используется zero-vector fallback. Это важно учитывать при интерпретации результатов: такие примеры могут быть слишком легкими для модели.

### Аугментация

Dataset добавляет небольшой Gaussian noise:

- иногда к molecule embedding;
- иногда к positive aptamer embedding.

Это работает как простая регуляризация.

## Файл `Model.py`

Содержит основную contrastive-модель:

```python
MicroContrastiveModel
```

Модель имеет две encoder-ветки:

- `apt_encoder`;
- `mol_encoder`.

Обе ветки имеют похожую структуру:

```text
Linear
LayerNorm
GELU
Dropout
Linear
LayerNorm
GELU
Dropout
```

По умолчанию размерность:

```text
768 -> 768
```

### `encode_aptamer`

Кодирует aptamer embedding и нормализует результат:

```python
F.normalize(z, dim=-1)
```

### `encode_molecule`

Кодирует molecule embedding и тоже нормализует результат:

```python
F.normalize(z, dim=-1)
```

Нормализация нужна, потому что дальше используется cosine similarity / dot product на единичной сфере.

### `forward`

Принимает:

```python
anchor_smi
positive_apts
negative_apts
```

Возвращает:

```python
{
    'z_anchor': z_anchor,
    'z_positive': z_positive,
    'z_negatives': z_negatives
}
```

Где:

- `z_anchor` - molecule embedding в contrastive-пространстве;
- `z_positive` - positive aptamer embedding;
- `z_negatives` - batch negative aptamer embeddings.

## Файл `Loss.py`

Содержит loss:

```python
TemperatureScaledLoss
```

Это cross-entropy loss по positive и negative similarity.

### Temperature

Temperature хранится как trainable parameter:

```python
self.log_temperature
```

Фактическая температура:

```python
torch.exp(self.log_temperature).clamp(0.07, 0.5)
```

Temperature управляет резкостью logits: чем ниже temperature, тем сильнее различия similarity влияют на loss.

### Forward

На вход:

```python
z_anchor
z_positive
z_negatives
```

Считаются:

```text
positive similarity = dot(z_anchor, z_positive) / temperature
negative similarity = dot(z_anchor, z_negative) / temperature
```

Logits имеют вид:

```text
[positive_sim, negative_sim_1, negative_sim_2, ...]
```

Правильный класс всегда индекс `0`, то есть positive pair.

Loss:

```python
F.cross_entropy(logits, labels)
```

Метрики:

- `accuracy` - positive logit выше всех negative logits;
- `top3_acc` - positive входит в top-3;
- `temperature` - текущее значение temperature.

Важно: loss обучает relative ranking. Он требует, чтобы positive был выше negatives, но сам по себе не задает фиксированный абсолютный порог similarity.

## Файл `FinalTrainer.py`

Содержит training loop:

```python
FinalTrainer
```

### Инициализация

Создает:

- модель;
- train/validation loaders;
- `TemperatureScaledLoss`;
- optimizer `AdamW`;
- scheduler `StepLR`;
- историю метрик.

Optimizer использует разные learning rates:

```python
model parameters: 4e-4
temperature parameter: 1e-2
```

Temperature обучается быстрее, чем остальные веса.

### `train_epoch`

Для каждого batch:

1. Переносит данные на device.
2. Прогоняет модель.
3. Считает contrastive loss.
4. Добавляет L2 regularization.
5. Делает backward.
6. Применяет gradient clipping.
7. Обновляет optimizer.
8. Собирает loss, accuracy, top-3 accuracy.

### `validate`

Работает аналогично, но без backward и optimizer step.

### `train`

Основной цикл обучения.

На каждой эпохе:

- train epoch;
- validation;
- scheduler step;
- запись history;
- печать метрик;
- сохранение лучшей модели по validation accuracy.

Checkpoint сохраняется в:

```text
final_micro_model.pth
```

В checkpoint входят:

- epoch;
- `model_state_dict`;
- `optimizer_state_dict`;
- validation accuracy;
- top-3 accuracy;
- train accuracy;
- temperature;
- history.

## Файл `Contrast Learning.py`

Это главный orchestrator для contrastive-части.

До генерации аптамеров он делает:

1. Загружает данные через `load_data`.
2. Делит positive и negative пары на train/val/test.
3. Создает `FinalContrastiveDataset`.
4. Создает `DataLoader`.
5. Создает `MicroContrastiveModel`.
6. Обучает модель через `FinalTrainer`.
7. Строит графики обучения.
8. Оценивает модель на test set.
9. Загружает лучший checkpoint.
10. Строит визуализацию embedding-пространства.
11. Запускает Wasserstein/pair-distance analysis.
12. Сохраняет адаптированные embeddings.

### Split Данных

Positive и negative пары делятся отдельно:

```text
70% train
15% validation
15% test
```

Это сделано отдельно для positive и negative массивов.

### DataLoader

Используется custom `collate_fn`, который собирает:

```python
anchor_smi
positive_apts
negative_apts
```

Batch size:

```python
32
```

### Обучение

Модель создается с размерностями:

```python
input_dim_apt=768
input_dim_mol=768
latent_dim=768
projection_dim=768
```

Device:

```python
cuda if available else cpu
```

Обучение:

```python
trainer.train(n_epochs=15, save_path='final_micro_model.pth')
```

### Test Evaluation

После обучения считается:

- accuracy;
- separation;
- positive mean similarity;
- negative mean similarity.

Хороший результат выглядит так:

```text
positive mean similarity: high
negative mean similarity: low
separation: positive_mean - negative_mean high
accuracy: high
```

### Сохраняемые Графики

`plot_final_results` сохраняет:

```text
final_training_results.png
```

`plot_similarity_distributions` сохраняет:

```text
final_similarity_distributions.png
```

`visualize_embeddings_correct` сохраняет:

```text
embedding_visualization_correct.png
```

### Сохранение Adapted Embeddings

Функция `save_embeddings` прогоняет исходные embeddings через обученные encoders:

```text
apt_pos -> model.encode_aptamer
smi_pos -> model.encode_molecule
apt_neg -> model.encode_aptamer
smi_neg -> model.encode_molecule
```

И сохраняет:

```text
apt_pos_final.npy
smi_pos_final.npy
apt_neg_final.npy
smi_neg_final.npy
```

Эти файлы содержат embeddings уже в contrastive-пространстве модели.

## Файл `wasserstein_utils.py`

Этот файл анализирует качество разделения positive и negative пар через расстояния:

```text
distance = 1 - cosine_similarity
```

Основной класс:

```python
PairDistanceAnalyzer
```

Основные функции:

- `get_pair_distances_from_loader`
- `analyze_model_with_loader`
- `analyze_model_pair_distances`
- `create_pair_distance_heatmap`

### `get_pair_distances_from_loader`

Берет trained model и `DataLoader`, затем считает:

- positive distances;
- negative distances;
- positive similarities;
- negative similarities.

Для positive:

```text
distance = 1 - cosine(z_molecule, z_positive_aptamer)
```

Для negative:

```text
distance = 1 - cosine(z_molecule, z_negative_aptamer)
```

### `PairDistanceAnalyzer`

Получает массивы positive и negative distances.

Считает:

- mean/std distances;
- Wasserstein distance;
- Earth Mover's Distance;
- outliers;
- ambiguous pairs;
- theoretical best threshold;
- theoretical accuracy.

### Wasserstein Distance

Используется библиотека `ot` для optimal transport.

Идея:

```text
если распределения positive distances и negative distances далеко друг от друга,
то модель хорошо разделяет классы.
```

### Ambiguous Pairs

`find_ambiguous_pairs` ищет пары в зоне перекрытия распределений.

Такие пары потенциально:

- шумные;
- плохо размеченные;
- биологически неоднозначные;
- сложные для модели.

### Визуализации

Может сохранять:

```text
pair_distance_distributions.png
pair_transport_plan.png
pair_distance_heatmap.png
```

## Основной Поток Выполнения

Полный поток contrastive-части:

```text
raw dataset
    |
    v
Embeddings.py
    |
    v
aptabench_with_embeddings.csv
    |
    v
load_data_and_visual_data.load_data
    |
    v
positive arrays + negative arrays
    |
    v
FinalContrastiveDataset
    |
    v
DataLoader
    |
    v
MicroContrastiveModel
    |
    v
TemperatureScaledLoss + FinalTrainer
    |
    v
final_micro_model.pth
    |
    v
test evaluation + visualization + Wasserstein analysis
    |
    v
adapted contrastive embeddings
```

## Основные Артефакты

После запуска contrastive-части могут появиться:

```text
final_micro_model.pth
final_training_results.png
final_similarity_distributions.png
embedding_visualization_correct.png
pair_distance_distributions.png
apt_pos_final.npy
smi_pos_final.npy
apt_neg_final.npy
smi_neg_final.npy
```

## Как Интерпретировать Метрики

### Accuracy

Показывает, как часто positive aptamer ближе к molecule embedding, чем sampled negative aptamers.

### Positive Mean Similarity

Среднее cosine similarity для interacting пар.

Ожидается высоким.

### Negative Mean Similarity

Среднее cosine similarity для non-interacting пар.

Ожидается низким или отрицательным.

### Separation

```text
positive_mean_similarity - negative_mean_similarity
```

Чем выше, тем лучше разделение.

### Distance

```text
distance = 1 - cosine_similarity
```

Для хорошей модели:

- positive distances маленькие;
- negative distances большие.

### Wasserstein Distance

Показывает, насколько далеко друг от друга распределения positive и negative distances.

Чем выше, тем лучше разделены классы.

## Минимальный Запуск

Если `aptabench_with_embeddings.csv` уже существует:

```bash
python "Contrast Learning.py"
```

Если embedding CSV еще не создан, сначала нужно подготовить его через `Embeddings.py`.
