import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm



def simple_gena_encode(csv_file, sequence_column='aptamer', output_file='embeddings.csv'):

    tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')

    # Используем AutoModelForMaskedLM
    model = AutoModelForMaskedLM.from_pretrained(
        'AIRI-Institute/gena-lm-bert-base',
        trust_remote_code=True,
        output_hidden_states=True
    )
    model.eval()

    df = pd.read_csv(csv_file)

    if sequence_column not in df.columns:
        print(f"Колонка '{sequence_column}' не найдена в файле!")
        print(f"Доступные колонки: {list(df.columns)}")
        return None

    # Получаем последовательности
    sequences = df[sequence_column].tolist()

    # Очищаем последовательности
    clean_sequences = []
    clean_indices = []

    print(" Проверка последовательностей...")
    for idx, seq in enumerate(sequences):
        # Проверка на NaN
        if pd.isna(seq):
            print(f" Пропуск строки {idx}: NaN значение")
            continue

        # Проверка типа данных
        if not isinstance(seq, str):
            print(f" Пропуск строки {idx}: не строковое значение (тип: {type(seq).__name__})")
            continue

        # Проверка на пустую строку
        seq = str(seq).strip()
        if not seq:
            print(f" Пропуск строки {idx}: пустая строка")
            continue

        # Проверка минимальной длины
        if len(seq) < 3:
            print(f" Пропуск строки {idx}: слишком короткая последовательность ({len(seq)} символов)")
            continue

        clean_sequences.append(seq)
        clean_indices.append(idx)

    print(f"    Оригинальных последовательностей: {len(sequences)}")
    print(f"    Очищенных последовательностей: {len(clean_sequences)}")
    print(f"    Удалено последовательностей: {len(sequences) - len(clean_sequences)}")

    if len(clean_sequences) == 0:
        print(" Нет валидных последовательностей для обработки!")
        return None

    # Настройки
    max_length = 32
    batch_size = 16
    all_embeddings = []

    # Обработка батчами
    print(" Начинаем кодирование...")
    for i in tqdm(range(0, len(clean_sequences), batch_size), desc="Обработка батчей"):
        batch = clean_sequences[i:i + batch_size]

        try:
            # Конвертируем U -> T
            batch = [seq.upper().replace('U', 'T') for seq in batch]

            # Проверяем после конвертации
            for j, seq in enumerate(batch):
                if not all(c in 'ATCGN' for c in seq):
                    print(f" Батч {i}, последовательность {j}: содержит недопустимые символы: {seq}")

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
            print(f"   Ошибка при обработке батча {i}-{i + batch_size}: {e}")
            print(f"   Размер батча: {len(batch)}")
            print(f"   Пример последовательности: {batch[0] if batch else 'нет'}")
            continue

    if len(all_embeddings) == 0:
        print("Не удалось получить эмбеддинги!")
        return None

    all_embeddings = np.vstack(all_embeddings)

    print(f" Получено эмбеддингов: {all_embeddings.shape}")

    # Проверяем на NaN в эмбеддингах
    nan_count = np.isnan(all_embeddings).sum()
    if nan_count > 0:
        print(f" В эмбеддингах найдено {nan_count} NaN значений")
        # Заменяем NaN на 0
        all_embeddings = np.nan_to_num(all_embeddings, nan=0.0)
        print(" NaN значения заменены на 0")


    inf_count = np.isinf(all_embeddings).sum()
    if inf_count > 0:
        print(f" В эмбеддингах найдено {inf_count} inf значений")
        # Заменяем inf на максимальные/минимальные значения
        all_embeddings = np.where(np.isinf(all_embeddings), 0.0, all_embeddings)
        print(" Inf значения заменены на 0")

    # Создаем DataFrame с эмбеддингами
    embed_df = pd.DataFrame(all_embeddings)
    embed_df.columns = [f'emb_{i}' for i in range(all_embeddings.shape[1])]

    embed_df['original_index'] = clean_indices[:len(embed_df)]

    # Создаем DataFrame с оригинальными данными и индексом
    original_df = df.copy()
    original_df['original_index'] = original_df.index

    # Объединяем с оригинальными данными по индексу
    result_df = pd.merge(original_df, embed_df, on='original_index', how='left')

    # Удаляем вспомогательную колонку
    result_df = result_df.drop('original_index', axis=1)

    rows_with_emb = result_df[[f'emb_{i}' for i in range(all_embeddings.shape[1])]].notna().all(axis=1).sum()
    print(f"Строк с эмбеддингами: {rows_with_emb} из {len(result_df)}")

    result_df.to_csv(output_file, index=False)

    print(f"Готово! Сохранено в {output_file}")
    print(f"Размер эмбеддингов: {all_embeddings.shape}")
    print(f"Размер итогового DataFrame: {result_df.shape}")

    return result_df


# Функция для проверки CSV файла перед обработкой
def check_csv_file(csv_file, sequence_column='aptamer'):

    print(f"🔍 Проверка файла: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
        print(f"Файл загружен. Размер: {df.shape}")
        print(f"Колонки: {list(df.columns)}")

        if sequence_column not in df.columns:
            print(f"Колонка '{sequence_column}' не найдена!")
            return None

        sequences = df[sequence_column]

        print("\n  Анализ данных в колонке '{}':".format(sequence_column))
        print(f"   Всего строк: {len(sequences)}")
        print(f"   Уникальных значений: {sequences.nunique()}")
        print(f"   Пустых значений (NaN): {sequences.isna().sum()}")
        print(f"   Не строковых значений: {(sequences.apply(type) != str).sum()}")

        # Примеры проблемных строк
        print("\n Примеры данных:")
        for i in range(min(5, len(sequences))):
            val = sequences.iloc[i]
            print(f"   Строка {i}: '{val}' (тип: {type(val).__name__})")

        # Ищем проблемные строки
        print("\nПроблемные строки:")
        problem_count = 0
        for idx, val in enumerate(sequences):
            if pd.isna(val):
                print(f"   Строка {idx}: NaN")
                problem_count += 1
            elif not isinstance(val, str):
                print(f"   Строка {idx}: не строка ({type(val).__name__}): {val}")
                problem_count += 1
            elif not val.strip():
                print(f"   Строка {idx}: пустая строка")
                problem_count += 1

        if problem_count == 0:
            print("Проблемных строк не найдено!")
        else:
            print(f"Найдено {problem_count} проблемных строк")

        return df

    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None


if __name__ == "__main__":
    csv_file = "AptaBench_dataset_v2.csv"
    sequence_column = "sequence"
    df_checked = check_csv_file(csv_file, sequence_column)

    if df_checked is not None:

        # Запускаем кодирование
        result = simple_gena_encode(
            csv_file=csv_file,
            sequence_column=sequence_column,
            output_file="rna_pos_embeddings.csv"
        )

        if result is not None:
            print(f"Исходных строк: {len(df_checked)}")
            print(f"Результирующих строк: {len(result)}")

            # Проверяем результат
            emb_columns = [col for col in result.columns if col.startswith('emb_')]
            if emb_columns:
                print(f"Колонок с эмбеддингами: {len(emb_columns)}")
                print(f"Размерность эмбеддинга: {len(emb_columns)}")

                # Проверяем первые эмбеддинги
                print("\nПервые 5 эмбеддингов (первые 5 значений):")
                for i in range(min(5, len(result))):
                    emb_sample = result[emb_columns].iloc[i].values[:5]
                    print(f"   Строка {i}: {emb_sample}")
    else:
        print("Не удалось проверить файл. Кодирование отменено.")