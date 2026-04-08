# import os
#
# os.environ["USE_TF"] = "0"
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from tqdm import tqdm


def data_prepare(data, sequence_column='sequence', smiles_column='canonical_smiles'):
    df = pd.read_csv(data)

    if sequence_column not in df.columns:
        print(f"Колонка '{sequence_column}' не найдена в файле!")
        return None
    if smiles_column not in df.columns:
        print(f"Колонка '{smiles_column}' не найдена в файле!")
        return None

    sequence = df[sequence_column].tolist()
    smiles = df[smiles_column].tolist()

    clean_sequences = []
    clean_smiles = []
    clean_indices = []

    # Проверка аптамеров
    print("Проверка аптамеров...")
    for idx, seq in enumerate(sequence):
        if pd.isna(seq):
            print(f"Пропуск строки {idx}: NaN значение")
            continue
        if not isinstance(seq, str):
            print(f"Пропуск строки {idx}: не строковое значение (тип: {type(seq).__name__})")
            continue
        seq = str(seq).strip()
        if not seq:
            print(f"Пропуск строки {idx}: пустая строка")
            continue
        if len(seq) < 3:
            print(f"Пропуск строки {idx}: слишком короткая последовательность ({len(seq)} символов)")
            continue

        clean_sequences.append(seq)
        clean_indices.append(idx)

    print(f"    Оригинальных последовательностей аптамеров: {len(sequence)}")
    print(f"    Очищенных последовательностей аптамеров: {len(clean_sequences)}")
    print(f"    Удалено последовательностей аптамеров: {len(sequence) - len(clean_sequences)}")

    if len(clean_sequences) == 0:
        print("Нет валидных последовательностей для обработки!")
        return None

    # Проверка SMILES
    print("Проверка SMILES...")
    for idx, smi in enumerate(smiles):
        if pd.isna(smi):
            print(f"Пропуск строки {idx}: NaN значение")
            continue
        if not isinstance(smi, str):
            print(f"Пропуск строки {idx}: не строковое значение (тип: {type(smi).__name__})")
            continue
        smi = str(smi).strip()
        if not smi:
            print(f"Пропуск строки {idx}: пустая строка")
            continue
        if len(smi) < 3:
            print(f"Пропуск строки {idx}: слишком короткая последовательность ({len(smi)} символов)")
            continue

        clean_smiles.append(smi)

    print(f"    Оригинальных последовательностей SMILES: {len(smiles)}")
    print(f"    Очищенных последовательностей SMILES: {len(clean_smiles)}")
    print(f"    Удалено последовательностей SMILES: {len(smiles) - len(clean_smiles)}")

    if len(clean_smiles) == 0:
        print("Нет валидных последовательностей SMILES для обработки!")
        return None

    # Возвращаем DataFrame, очищенные последовательности и индексы
    return df, clean_sequences, clean_smiles, clean_indices


def aptamer_encode(clean_sequences):
    tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
    model = AutoModelForMaskedLM.from_pretrained(
        'AIRI-Institute/gena-lm-bert-base',
        trust_remote_code=True,
        output_hidden_states=True
    )
    model.eval()

    max_length = 32
    batch_size = 16
    all_embeddings = []

    print("Начинаем кодирование аптамеров...")
    for i in tqdm(range(0, len(clean_sequences), batch_size), desc="Обработка батчей"):
        batch = clean_sequences[i:i + batch_size]

        try:
            # Конвертируем U -> T
            batch = [seq.upper().replace('U', 'T') for seq in batch]

            # Токенизация
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            # Получение выходов с hidden_states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            last_hidden_state = outputs.hidden_states[-1]

            # Усредняем по последовательности
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            sum_emb = (last_hidden_state * attention_mask).sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            embeddings = sum_emb / sum_mask

            all_embeddings.append(embeddings.numpy())

        except Exception as e:
            print(f"Ошибка при обработке батча {i}-{i + batch_size}: {e}")
            continue

    if len(all_embeddings) == 0:
        print("Не удалось получить эмбеддинги!")
        return None

    all_embeddings = np.vstack(all_embeddings)
    print(f"Получено эмбеддингов аптамеров: {all_embeddings.shape}")

    return all_embeddings


def smiles_encode(clean_smiles):
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model.eval()

    print("\nНачинаем кодирование SMILES...")
    batch_size = 32
    all_embeddings = []

    for i in tqdm(range(0, len(clean_smiles), batch_size), desc="Обработка SMILES"):
        batch = clean_smiles[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            all_embeddings.append(embeddings.cpu().numpy())

    embeddings_array = np.vstack(all_embeddings)
    print(f"Получено эмбеддингов SMILES: {embeddings_array.shape}")

    return embeddings_array


def create_combined_dataset(original_df, seq_embeddings, smi_embeddings, clean_indices,
                            output_file='combined_dataset.csv'):
    """
    Создает объединенный датасет с оригинальными колонками и эмбеддингами

    Parameters:
    -----------
    original_df : pandas.DataFrame
        Исходный DataFrame с данными
    seq_embeddings : numpy.ndarray
        Эмбеддинги последовательностей (аптамеров)
    smi_embeddings : numpy.ndarray
        Эмбеддинги SMILES
    clean_indices : list
        Индексы строк, которые были успешно обработаны
    output_file : str
        Имя выходного файла
    """

    print("\n" + "=" * 60)
    print("Создание объединенного датасета")
    print("=" * 60)

    # Создаем DataFrame для эмбеддингов последовательностей
    seq_emb_df = pd.DataFrame(
        seq_embeddings,
        columns=[f'seq_emb_{i}' for i in range(seq_embeddings.shape[1])]
    )

    # Создаем DataFrame для эмбеддингов SMILES
    smi_emb_df = pd.DataFrame(
        smi_embeddings,
        columns=[f'smi_emb_{i}' for i in range(smi_embeddings.shape[1])]
    )

    # Создаем копию исходного DataFrame для выбранных индексов
    result_df = original_df.iloc[clean_indices].copy().reset_index(drop=True)

    # Добавляем эмбеддинги
    result_df = pd.concat([result_df, seq_emb_df, smi_emb_df], axis=1)

    # Проверяем результат
    print(f"\nРазмер итогового датасета: {result_df.shape}")
    print(f"Количество строк: {len(result_df)}")
    print(f"Количество колонок: {len(result_df.columns)}")

    # Показываем первые несколько колонок
    print("\nПервые 10 колонок:")
    for i, col in enumerate(result_df.columns[:10]):
        print(f"  {i + 1}. {col}")

    print("\nПоследние 10 колонок:")
    for i, col in enumerate(result_df.columns[-10:]):
        print(f"  {len(result_df.columns) - 9 + i}. {col}")

    # Сохраняем результат
    result_df.to_csv(output_file, index=False)
    print(f"\n  Датасет сохранен в '{output_file}'")

    # Статистика
    print("\n   Статистика:")
    print(f"  - Оригинальных строк: {len(original_df)}")
    print(f"  - Успешно обработано: {len(result_df)}")
    print(f"  - Пропущено: {len(original_df) - len(result_df)}")
    print(f"  - Размерность эмбеддингов последовательностей: {seq_embeddings.shape[1]}")
    print(f"  - Размерность эмбеддингов SMILES: {smi_embeddings.shape[1]}")

    return result_df


if __name__ == "__main__":
    # Загружаем данные и получаем очищенные последовательности
    result = data_prepare('AptaBench_dataset_v2.csv')

    if result is not None:
        original_df, clean_sequences, clean_smiles, clean_indices = result

        # Кодируем аптамеры
        seq_embeddings = aptamer_encode(clean_sequences)

        # Кодируем SMILES
        smi_embeddings = smiles_encode(clean_smiles)

        # Создаем объединенный датасет
        if seq_embeddings is not None and smi_embeddings is not None:
            combined_dataset = create_combined_dataset(
                original_df=original_df,
                seq_embeddings=seq_embeddings,
                smi_embeddings=smi_embeddings,
                clean_indices=clean_indices,
                output_file='aptabench_with_embeddings.csv'
            )

            # Дополнительно: сохраняем эмбеддинги отдельно (опционально)
            np.save('seq_embeddings.npy', seq_embeddings)
            np.save('smi_embeddings.npy', smi_embeddings)
            print("\n   Эмбеддинги также сохранены отдельно:")
            print("  - seq_embeddings.npy")
            print("  - smi_embeddings.npy")
        else:
            print(" Ошибка при получении эмбеддингов!")