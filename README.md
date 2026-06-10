# Contrast Learning — генерация non-interacting аптамеров

Пайплайн для генерации DNA/RNA-последовательностей аптамеров, которые **не должны** связываться с заданной малой молекулой. Основан на contrastive learning, кластеризации negative latent seeds и условном GRU-декодере.

## Схема пайплайна

```text
AptaBench CSV
    │
    ▼
Embeddings.py ──► aptabench_with_embeddings_v2.csv
    │               (GENA-LM + ChemBERTa, 768d)
    ▼
Contrast Learning.py
    ├─ MicroContrastiveModel (768→768)
    ├─ decoder.py (KMeans → negative cluster seeds)
    ├─ GRU.py (mol_emb + negative_latent → DNA)
    └─ generated_pairs_molecule_aptamer.txt
    │
    ▼
rerank_generated_aptamers.py ──► rerank_*.csv
    │
    ▼
Внешняя валидация (опционально)
    ├─ RSAPred (pKd)
    ├─ boltz_script.py (Boltz-2 affinity)
    └─ Vina / SimRNA (Colab, см. докинг_fixed.ipynb)
```

## Структура проекта

| Файл | Назначение |
|------|------------|
| `Contrast Learning.py` | Главный скрипт: contrastive → кластеры → GRU → генерация |
| `Embeddings.py` | Построение `seq_emb_*` / `smi_emb_*` из сырых sequence + SMILES |
| `Model.py` | `MicroContrastiveModel` |
| `DataPrepare.py` | `FinalContrastiveDataset` |
| `FinalTrainer.py` | Обучение contrastive-модели |
| `Loss.py` | `TemperatureScaledLoss` |
| `load_data_and_visual_data.py` | Загрузка CSV, метрики, t-SNE/UMAP/PCA |
| `wasserstein_utils.py` | Wasserstein-анализ разделения pos/neg |
| `decoder.py` | Кластеризация 768d embeddings → negative seeds для GRU |
| `GRU.py` | `ConditionalGRUDecoder`, обучение и генерация последовательностей |
| `rerank_generated_aptamers.py` | Пост-фильтрация по `sequence_sim` |
| `boltz_script.py` | Batch-прогон Boltz-2 по парам из AptaBench |
| `CVAE.py` | Альтернативный декодер (экспериментальный, не в основном пайплайне) |

### Данные и артефакты

| Файл | Описание |
|------|----------|
| `AptaBench_dataset_v2.csv` | Исходные пары (sequence, SMILES, label) |
| `aptabench_with_embeddings_v2.csv` | Датасет с эмбеддингами (6413 строк) |
| `final_micro_model.pth` | Веса contrastive-модели |
| `best_conditional_decoder.pth` | Веса GRU-декодера |
| `cluster_embeddings_768d.npy` | 768d эмбеддинги для кластеризации |
| `cluster_labels_768d.npy` | Метки KMeans |
| `cluster_types.npy` | positive / negative для каждой точки |
| `generated_pairs_molecule_aptamer.txt` | Результат генерации |

Дополнительная документация: `GRU_ARCHITECTURE_AND_PIPELINE.md`, `PIPELINE_HANDOFF.md`, `VALIDATION_PAIRS_CAFFEINE.md`.

---

## Установка окружения

### Вариант A — Conda (рекомендуется)

```bash
cd "C:\Users\USER\Contrast Learning"
conda env create -f environment.yaml
conda activate contrast-learning
```

### Вариант B — pip

```bash
cd "C:\Users\USER\Contrast Learning"
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### HuggingFace-модели (скачаются при первом запуске)

- Аптамеры: `AIRI-Institute/gena-lm-bert-base`
- Молекулы: `seyonec/ChemBERTa-zinc-base-v1`

Нужен интернет и ~2–4 GB на диске для кэша моделей.

### Boltz-2 (отдельное окружение)

Boltz ставится отдельно и вызывается как CLI `boltz predict`. Для batch-скрининга используйте `boltz_script.py` в окружении, где установлен Boltz (часто отдельный conda env).

---

## Быстрый старт

### 1. Подготовить эмбеддинги (если ещё нет v2 CSV)

```bash
python Embeddings.py
```

Ожидаемый результат: `aptabench_with_embeddings_v2.csv` с колонками `seq_emb_*`, `smi_emb_*`.

### 2. Полный пайплайн (contrastive + кластеры + GRU + генерация)

```bash
python "Contrast Learning.py"
```

При `USE_PRETRAINED = True` (по умолчанию):
- contrastive загружается из `final_micro_model.pth`
- GRU загружается из `best_conditional_decoder.pth`
- кластеры загружаются из `cluster_*.npy` (если есть)

Результат: `generated_pairs_molecule_aptamer.txt`.

### 3. Rerank сгенерированных аптамеров

```bash
python rerank_generated_aptamers.py ^
  --molecule-index 0 ^
  --data aptabench_with_embeddings_v2.csv ^
  --output rerank_molecule_0.csv ^
  --diagnostics
```

Фильтр по SMILES (например, caffeine):

```bash
python rerank_generated_aptamers.py ^
  --smiles-contains "Cn1c(=O)c2" ^
  --data aptabench_with_embeddings_v2.csv ^
  --output rerank_caffeine.csv ^
  --diagnostics
```

### 4. Boltz-2 batch (опционально)

```bash
python boltz_script.py aptabench ^
  --data aptabench_with_embeddings_v2.csv ^
  --n-positive 100 ^
  --n-negative 100
```

Dry-run без запуска Boltz:

```bash
python boltz_script.py aptabench --data aptabench_with_embeddings_v2.csv --n-positive 5 --n-negative 5 --dry-run
```

---

## Настройки в `Contrast Learning.py`

```python
DATA_FILE = "aptabench_with_embeddings_v2.csv"
MODEL_CHECKPOINT = "final_micro_model.pth"
USE_PRETRAINED = True   # False = переобучить contrastive + GRU с нуля
```

### GEN_CONFIG (генерация аптамеров)

| Параметр | По умолчанию | Смысл |
|----------|--------------|-------|
| `target_smiles` | `None` | Фильтр мишени; подстрока SMILES или `None` = все |
| `max_generation_targets` | `10` | Top-N молекул по contrastive separation |
| `sequence_sim_filter` | `True` | Оставлять только `sequence_sim < positive_mean` |
| `max_latent_similarity` | `0.15` | Порог cosine(mol_z, negative_seed) |
| `n_latent_points` | `128` | Сколько negative seeds пробовать |
| `n_keep` | `50` | Сколько аптамеров сохранить на молекулу |

Пример — только caffeine:

```python
"target_smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
```

---

## Интерпретация метрик

### `latent_similarity` (до декодирования)

Cosine между `mol_z` и negative latent seed.  
Хорошо: ≤ 0.15, лучше отрицательное значение (например, −0.5).

### `sequence_sim` (после декодирования)

Cosine между `mol_z` и эмбеддингом сгенерированной последовательности.  
Используйте **относительный** фильтр: `sequence_sim < baseline_positive_mean` для целевой молекулы.  
Типичный диапазон даже для known negatives: 0.83–0.96.

### RSAPred pKd (AptaBench v2)

| label | mean pKd |
|-------|----------|
| positive | ~6.24 |
| negative | ~3.34 |

Порог: pKd < 4.5 → negative-like.

### Boltz `affinity_pred_value`

log₁₀(IC50, µM). **Выше = слабее связывание.** +2 ≈ слабый/декой.  
Не путать со знаком Vina!

### Vina (kcal/mol)

Для калибровки на caffeine: positives ~−8.5, negatives ~−5.6.  
Текущий SimRNA+Vina пайплайн в Colab **не пригоден как основная метрика** (coarse-grained PDB ~95 атомов).

---

## Переобучение с нуля

```python
USE_PRETRAINED = False
```

Удалите или переименуйте перед запуском (опционально):
- `final_micro_model.pth`
- `best_conditional_decoder.pth`
- `cluster_*.npy`

Запуск займёт значительно больше времени (contrastive 15 эпох + GRU).

---

## Требования к данным

CSV должен содержать:

- `sequence` (или колонка с `sequence` в имени)
- `canonical_smiles` (или колонка со `smiles` в имени)
- `label` — `1` = interacting, `0` = non-interacting
- `seq_emb_*` — эмбеддинги аптамеров (768 колонок)
- `smi_emb_*` — эмбеддинги молекул (768 колонок)

---

## Известные ограничения

1. Contrastive-модель разделяет pos/neg в latent-пространстве, но абсолютный `sequence_sim` остаётся высоким даже у negatives.
2. GRU может запоминать скэффолды из negative-датасета (Specificity) — проверяйте Levenshtein / rerank.
3. `decoder.py` содержит только кластеризацию; генерацию последовательностей делает `GRU.py`.
4. Внешняя валидация (Boltz, RSAPred, Vina) запускается отдельно.

---

## Полезные команды

```bash
# Диагностика эмбеддингов последовательностей
python diagnose_seq_embedding_pipeline.py --help

# Активация окружения (если уже создано вручную)
conda activate contrast-learning
```

---

