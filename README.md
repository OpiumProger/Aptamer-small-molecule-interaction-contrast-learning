# Contrast Learning — генерация non-interacting аптамеров

Пайплайн для генерации DNA/RNA-последовательностей аптамеров, которые **не должны** связываться с заданной малой молекулой. Основан на contrastive learning, кластеризации negative latent seeds и условном GRU-декодере.

**Внешняя валидация (июнь 2026):** 19 diverse-пар (rerank, `motif_penalty=0`) проверены в RSAPred — **13/19 (68%)** с pKd < 4.5 (negative-like). Подробности — [раздел RSAPred](#rsapred-валидация).

## Схема пайплайна

```text
AptaBench CSV
    │
    ▼
Embeddings.py ──► aptabench_with_embeddings_v2.csv
    │               (GENA-LM + ChemBERTa, 768d)
    ▼
Contrast Learning.py                    run_rsapred_generation.py
    │ (test split, крупные молекулы)         │ (RSAPred pKd, малые молекулы ≤40 atoms)
    ├─ MicroContrastiveModel (768→768)       └─ rsapred_* outputs
    ├─ decoder.py (KMeans → negative seeds)
    ├─ GRU.py (mol_emb + negative_latent → DNA)
    └─ generation_summary.csv
    │
    ▼
rerank_generated_aptamers.py ──► rerank_*.csv / rsapred_rerank_all.csv
    │   (sequence_sim, motif_penalty, rank_diverse)
    ▼
Внешняя валидация
    ├─ RSAPred (pKd) — основная метрика для малых молекул
    ├─ boltz_script.py (Boltz-2 affinity)
    └─ Vina / SimRNA (Colab, см. докинг_fixed.ipynb)
```

## Структура проекта

| Файл | Назначение |
|------|------------|
| `Contrast Learning.py` | Главный скрипт: contrastive → кластеры → GRU → генерация (test split) |
| `run_rsapred_generation.py` | Отдельный прогон для молекул RSAPred с pKd (малые мишени) |
| `Embeddings.py` | Построение `seq_emb_*` / `smi_emb_*` из сырых sequence + SMILES |
| `Model.py` | `MicroContrastiveModel` |
| `DataPrepare.py` | `FinalContrastiveDataset` (global/hard negatives, molecule split) |
| `FinalTrainer.py` | Обучение contrastive-модели |
| `Loss.py` | `TemperatureScaledLoss` |
| `load_data_and_visual_data.py` | Загрузка CSV, метрики, t-SNE/UMAP/PCA |
| `wasserstein_utils.py` | Wasserstein-анализ разделения pos/neg |
| `decoder.py` | Кластеризация 768d embeddings → negative seeds для GRU |
| `GRU.py` | `ConditionalGRUDecoder`, обучение, генерация, `motif_penalty` |
| `rerank_generated_aptamers.py` | Пост-фильтрация по `sequence_sim` и diverse rank |
| `export_rsapred_diverse_pairs.py` | Экспорт diverse-пар для ручной проверки в RSAPred |
| `boltz_script.py` | Batch-прогон Boltz-2 по парам из AptaBench |
| `CVAE.py` | Альтернативный декодер (экспериментальный) |

### Данные и артефакты

| Файл | Описание |
|------|----------|
| `AptaBench_dataset_v2.csv` | Исходные пары (sequence, SMILES, label, source, pKd) |
| `aptabench_with_embeddings_v2.csv` | Датасет с эмбеддингами (6413 строк) |
| `final_micro_model.pth` | Веса contrastive-модели |
| `best_conditional_decoder.pth` | Веса GRU-декодера |
| `cluster_embeddings_768d.npy` | 768d эмбеддинги для кластеризации |
| `cluster_labels_768d.npy` | Метки KMeans |
| `cluster_types.npy` | positive / negative для каждой точки |
| `generation_summary.csv` | Сводка генерации (default run) |
| `rsapred_generation_summary.csv` | Сводка RSAPred-прогона (20 молекул) |
| `rsapred_rerank_all.csv` | Rerank RSAPred-пар |
| `rsapred_validation_results.csv` | RSAPred pKd для 19 diverse-пар (ручная проверка) |
| `rsapred_pairs_to_submit.csv` | Diverse-кандидаты для отправки в RSAPred |

Дополнительная документация: `GRU_ARCHITECTURE_AND_PIPELINE.md`, `PIPELINE_HANDOFF.md`, `VALIDATION_PAIRS_CAFFEINE.md`.

---

## Установка окружения

### Вариант A — Conda (рекомендуется)

```bash
cd "C:\Users\USER\Contrast Learning"
conda env create -f environment.yaml
conda activate contrast-learning
# или: conda activate new_chemberta_env
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

---

## Быстрый старт

### 1. Подготовить эмбеддинги (если ещё нет v2 CSV)

```bash
python Embeddings.py
```

### 2a. Генерация (default — test split, top-10 молекул)

```bash
python "Contrast Learning.py"
```

Выход: `generation_summary.csv`, `top_candidates_for_tools.csv`, `generated_pairs_molecule_aptamer.txt`.

> **Внимание:** молекулы из test split часто крупные (аминогликозиды, ~50 heavy atoms) — **вне зоны калибровки RSAPred**. Для проверки в RSAPred используйте п. 2b.

### 2b. Генерация для RSAPred (малые молекулы с pKd)

```bash
python run_rsapred_generation.py --max-targets 20 --max-heavy-atoms 40
```

Выход: `rsapred_target_molecules.csv`, `rsapred_generation_summary.csv`, `rsapred_top_candidates_for_tools.csv`.

Приоритетные мишени: caffeine, xanthine, 2-aminopyrimidine.

### 3. Rerank

**RSAPred-прогон (рекомендуется перед внешней валидацией):**

```bash
python rerank_generated_aptamers.py ^
  --use-summary ^
  --summary rsapred_generation_summary.csv ^
  --all-molecules ^
  --output rsapred_rerank_all.csv ^
  --max-motif-penalty 0.05
```

**Default-прогон:**

```bash
python rerank_generated_aptamers.py ^
  --use-summary ^
  --summary generation_summary.csv ^
  --all-molecules ^
  --output rerank_all.csv ^
  --max-motif-penalty 0.05
```

**Экспорт diverse-пар для RSAPred:**

```bash
python export_rsapred_diverse_pairs.py
```

Создаёт `rsapred_pairs_to_submit.csv` (19 пар с `motif_penalty=0`).

### 4. Boltz-2 batch (опционально)

```bash
python boltz_script.py aptabench --data aptabench_with_embeddings_v2.csv --n-positive 100 --n-negative 100
```

---

## Настройки в `Contrast Learning.py`

```python
DATA_FILE = "aptabench_with_embeddings_v2.csv"
MODEL_CHECKPOINT = "final_micro_model.pth"
USE_PRETRAINED = False   # False = переобучить; True = загрузить готовые веса
```

### GEN_CONFIG (ключевые параметры)

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `target_smiles` | `None` | Подстрока SMILES или `None` = test split |
| `max_generation_targets` | `10` | Top-N молекул по `contrastive_separation` |
| `min_contrastive_separation` | `0.05` | Минимальный зазор pos/neg для мишени |
| `max_latent_sim_for_decode` | `-0.10` | Порог cosine(mol_z, decoded_seq) при отборе |
| `sequence_sim_filter` | `True` | Фильтр `decoded_sim < positive_mean` |
| `motif_penalty_weight` | `0.15` | Штраф за шаблоны CTTACGAC/GGGACGAC |
| `max_motif_penalty_for_tools` | `0.05` | Макс. motif для top-K в tools CSV |
| `n_keep` | `50` | Сколько аптамеров сохранить на молекулу |

Пример — только caffeine:

```python
"target_smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
```

---

## Интерпретация метрик

Метрики ниже относятся к **переобученной** модели (`USE_PRETRAINED=False`). После contrastive + GRU в latent-пространстве cosine для negatives становится **отрицательным** — это нормально и ожидаемо.

### На этапе генерации (`generation_summary.csv`)

| Метрика | Хорошо | Плохо | Смысл |
|---------|--------|-------|-------|
| `seed_sim` | < 0 | > 0.15 | Cosine(mol_z, negative latent seed) |
| `decoded_sim` | < −0.15 | > 0 | Cosine(mol_z, эмбеддинг сгенерированной последовательности) |
| `motif_penalty` | **0.00** | **0.12** | Штраф за префиксы `CTTACGAC` / `GGGACGAC` |
| `contrastive_separation` | ≥ 0.05 | < 0.05 | Зазор pos/neg mean для мишени в датасете |

Типичный RSAPred-прогон: mean `decoded_sim` ≈ **−0.18**; ~5% пар без шаблонов (`motif=0`).

### На этапе rerank (`rerank_*.csv`)

| Метрика | Фильтр | Смысл |
|---------|--------|-------|
| `sequence_sim` | ≤ 0.15 (absolute) | Cosine(mol_z, aptamer_emb) после декодирования |
| `passes_relative_filter` | `sequence_sim < baseline_positive_mean` | Относительный фильтр к позитивам мишени |
| `motif_penalty` | ≤ 0.05 для diverse | Без доминирующих шаблонов |
| `rank_diverse` | 1 = лучший diverse | Ранг среди кандидатов с `motif ≤ 0.05` |
| `composite_score` | min лучше | `sequence_sim + 0.15 × motif_penalty` |

> **Устаревшее:** диапазон `sequence_sim` 0.83–0.96 для negatives относился к **сырым** ChemBERTa/GENA эмбеддингам без contrastive-проекции. В текущем пайплайне `sequence_sim` и `decoded_sim` **отрицательные** (−0.10 … −0.22).

### RSAPred pKd (основная внешняя метрика)

| Класс (AptaBench v2) | mean pKd |
|----------------------|----------|
| positive (binding) | ~6.24 |
| negative (non-binding) | ~3.34 |

**Рабочий порог:** pKd **< 4.5** → negative-like (millimolar, слабое/нет связывания).

RSAPred калиброван на **малых молекулах** (median ~32 heavy atoms в RSAPred-парах). Крупные аминогликозиды из test split — **out-of-domain**; для них RSAPred даёт завышенный pKd (~5.0).

### Boltz `affinity_pred_value`

log₁₀(IC50, µM). **Выше = слабее связывание.** +2 ≈ слабый/декой.

### Vina (kcal/mol)

Для калибровки на caffeine: positives ~−8.5, negatives ~−5.6. SimRNA+Vina в Colab **не пригоден как основная метрика** (coarse-grained PDB).

---

## RSAPred валидация

Проверены **все 19 diverse-пар** из `rsapred_rerank_all.csv` (`motif_penalty=0`, без шаблонов GGGACGAC/CTTACGAC). Полная таблица: `rsapred_validation_results.csv`.

### Сводка

| Метрика | Значение |
|---------|----------|
| PASS (pKd < 4.5) | **13 / 19 (68%)** |
| FAIL (pKd ≥ 4.5) | 6 / 19 (32%) |
| Лучший результат | mol7_diverse_3 → **pKd 2.8** |
| Худший результат | mol10_diverse_1 → **pKd 6.37** (µM, предсказанное связывание) |
| Средний pKd (PASS) | ~3.7 |
| Средний pKd (FAIL) | ~5.2 |

### Лучшие пары (для демо / публикации)

| pair_id | SMILES | pKd |
|---------|--------|-----|
| mol7_diverse_3 | `Nc1ncc2[nH]cnc2n1` | **2.8** |
| mol7_diverse_1 | `Nc1ncc2[nH]cnc2n1` | **3.1** |
| mol7_diverse_2 | `Nc1ncc2[nH]cnc2n1` | **3.13** |

Молекула mol7: **3/3** diverse-кандидата прошли RSAPred.

### Провальные мишени

| Молекула | PASS/total | Комментарий |
|----------|------------|-------------|
| mol10 `Cc1ccc(NCCN)nc1` | 0/2 | Оба аптамера — binding (5.01 и 6.37 µM) |
| mol4 `NCCNc1ccccn1` | 0/2 | pKd 4.92 и 5.48 |
| mol8, mol11 | 0/1 | Borderline (4.67, 4.78) |

### Универсальная последовательность `CCUUACGACACAUUUGGG...`

Одна RNA на 6 молекулах: **5 PASS** (pKd 3.55–4.12), **1 FAIL** (mol10, pKd 5.01). Провал связан с мишенью, а не с последовательностью.

### Оговорки

- 68% — на **отфильтрованных** diverse-кандидатах, не на случайной выборке.
- Молекулы 9–13 heavy atoms (in-domain для RSAPred).
- Порог 4.5 — эвристика; официального cutoff у RSAPred нет.
- `sequence_sim` из rerank **слабо коррелирует** с RSAPred pKd (mol8: лучший sim, FAIL; mol7: хуже sim, PASS).

---

## Переобучение с нуля

```python
USE_PRETRAINED = False
```

Опционально удалите перед запуском: `final_micro_model.pth`, `best_conditional_decoder.pth`, `cluster_*.npy`.

Запуск займёт значительно больше времени (contrastive 15 эпох + GRU).

---

## Требования к данным

CSV должен содержать:

- `sequence` (или колонка с `sequence` в имени)
- `canonical_smiles` (или колонка со `smiles` в имени)
- `label` — `1` = interacting, `0` = non-interacting
- `seq_emb_*` — эмбеддинги аптамеров (768 колонок)
- `smi_emb_*` — эмбеддинги молекул (768 колонок)

Для RSAPred-прогона дополнительно: `source`, `pKd_value`.

---

## Известные ограничения

1. GRU склонен к mode collapse на шаблоны `CTTACGAC` / `GGGACGAC` (`motif_penalty=0.12`) — используйте rerank с `--max-motif-penalty 0.05`.
2. Contrastive `sequence_sim` не предсказывает RSAPred pKd надёжно — нужна внешняя валидация.
3. Некоторые мишени (аминопиридины mol4, mol10) устойчиво дают FAIL в RSAPred.
4. Default test split (крупные молекулы) не подходит для RSAPred — используйте `run_rsapred_generation.py`.
5. Boltz, RSAPred, Vina запускаются отдельно от основного скрипта.

---

## Полезные команды

```bash
# Диагностика contrastive-обучения
python diagnose_contrastive_training.py

# Диагностика эмбеддингов последовательностей
python diagnose_seq_embedding_pipeline.py --help

# Превью RSAPred-мишеней
python preview_rsapred_targets.py
```

