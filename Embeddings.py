import os

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel,T5EncoderModel
from tqdm import tqdm

def data_prepare(data, sequence_column='sequence', smiles_column='canonical_smiles'):
    df = pd.read_csv(data)

    if sequence_column not in df.columns:
        print(f"Колонка '{sequence_column}' не найдена в файле!")
        return None
    if smiles_column not in df.columns:
        print(f"Колонка '{smiles_column}' не найдена в файле!")
        return None

    clean_sequences = []
    clean_smiles = []
    clean_indices = []

    print("Проверка строк dataset (sequence + SMILES)...")
    for idx, row in df.iterrows():
        seq = row[sequence_column]
        smi = row[smiles_column]

        if pd.isna(seq) or pd.isna(smi):
            print(f"Пропуск строки {idx}: NaN в sequence или SMILES")
            continue
        if not isinstance(seq, str):
            print(f"Пропуск строки {idx}: sequence не строка (тип: {type(seq).__name__})")
            continue
        if not isinstance(smi, str):
            print(f"Пропуск строки {idx}: SMILES не строка (тип: {type(smi).__name__})")
            continue

        seq = str(seq).strip()
        smi = str(smi).strip()
        if not seq or not smi:
            print(f"Пропуск строки {idx}: пустой sequence или SMILES")
            continue
        if len(seq) < 3:
            print(f"Пропуск строки {idx}: слишком короткая sequence ({len(seq)} символов)")
            continue
        if len(smi) < 3:
            print(f"Пропуск строки {idx}: слишком короткий SMILES ({len(smi)} символов)")
            continue

        clean_sequences.append(seq.upper().replace('U', 'T'))
        clean_smiles.append(smi)
        clean_indices.append(idx)

    print(f"    Оригинальных строк: {len(df)}")
    print(f"    Очищенных строк: {len(clean_sequences)}")
    print(f"    Удалено строк: {len(df) - len(clean_sequences)}")

    if len(clean_sequences) == 0:
        print("Нет валидных строк для обработки!")
        return None

    # Возвращаем DataFrame, очищенные последовательности и индексы
    return df, clean_sequences, clean_smiles, clean_indices




def aptamer_encode(
    clean_sequences,
    encoder_type='auto_model',
    pooling='mean',
    max_length=128,
    batch_size=16,
    device=None,
):
    tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
    if encoder_type == 'masked_lm':
        model_cls = AutoModelForMaskedLM
    elif encoder_type == 'auto_model':
        model_cls = AutoModel
    else:
        raise ValueError("encoder_type должен быть 'auto_model' или 'masked_lm'")

    model = model_cls.from_pretrained(
        'AIRI-Institute/gena-lm-bert-base',
        trust_remote_code=True,
        output_hidden_states=True,
    )
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    model.eval()

    all_embeddings = []

    print(
        "Начинаем кодирование аптамеров "
        f"(GENA-LM {encoder_type}, pooling={pooling}, max_length={max_length}, device={device})..."
    )
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
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Получение выходов с hidden_states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            last_hidden_state = (
                outputs.hidden_states[-1]
                if outputs.hidden_states is not None
                else outputs.last_hidden_state
            )

            if pooling == 'mean':
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                sum_emb = (last_hidden_state * attention_mask).sum(dim=1)
                sum_mask = attention_mask.sum(dim=1)
                embeddings = sum_emb / sum_mask.clamp(min=1)
            elif pooling == 'cls':
                embeddings = last_hidden_state[:, 0, :]
            else:
                raise ValueError("pooling должен быть 'mean' или 'cls'")

            all_embeddings.append(embeddings.detach().cpu().numpy())

        except Exception as e:
            print(f"Ошибка при обработке батча {i}-{i + batch_size}: {e}")
            continue

    if len(all_embeddings) == 0:
        print("Не удалось получить эмбеддинги!")
        return None

    all_embeddings = np.vstack(all_embeddings)
    print(f"Получено эмбеддингов аптамеров: {all_embeddings.shape}")

    return all_embeddings


def smiles_encode(clean_smiles, device=None):
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    model.eval()

    print(f"\nНачинаем кодирование SMILES (ChemBERTa, device={device})...")
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
        inputs = {key: value.to(device) for key, value in inputs.items()}

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
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n  Датасет сохранен в '{output_file}'")

    # Статистика
    print("\n   Статистика:")
    print(f"  - Оригинальных строк: {len(original_df)}")
    print(f"  - Успешно обработано: {len(result_df)}")
    print(f"  - Пропущено: {len(original_df) - len(result_df)}")
    print(f"  - Размерность эмбеддингов последовательностей: {seq_embeddings.shape[1]}")
    print(f"  - Размерность эмбеддингов SMILES: {smi_embeddings.shape[1]}")

    return result_df


def build_parser():
    parser = argparse.ArgumentParser(description="Create AptaBench embeddings dataset.")
    parser.add_argument("--input", default="AptaBench_dataset_v2.csv")
    parser.add_argument("--output", default="aptabench_with_embeddings_v2.csv")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--aptamer-encoder", choices=["auto_model", "masked_lm"], default="auto_model")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-npy-prefix", default="v2")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    # Загружаем данные и получаем очищенные последовательности
    result = data_prepare(
        args.input,
        sequence_column=args.sequence_column,
        smiles_column=args.smiles_column,
    )

    if result is not None:
        original_df, clean_sequences, clean_smiles, clean_indices = result

        # Кодируем аптамеры
        seq_embeddings = aptamer_encode(
            clean_sequences,
            encoder_type=args.aptamer_encoder,
            pooling=args.pooling,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
        )

        # Кодируем SMILES
        smi_embeddings = smiles_encode(clean_smiles, device=args.device)

        # Создаем объединенный датасет
        if seq_embeddings is not None and smi_embeddings is not None:
            combined_dataset = create_combined_dataset(
                original_df=original_df,
                seq_embeddings=seq_embeddings,
                smi_embeddings=smi_embeddings,
                clean_indices=clean_indices,
                output_file=args.output,
            )

            # Дополнительно: сохраняем эмбеддинги отдельно (опционально)
            seq_npy = f"seq_embeddings_{args.save_npy_prefix}.npy"
            smi_npy = f"smi_embeddings_{args.save_npy_prefix}.npy"
            np.save(seq_npy, seq_embeddings)
            np.save(smi_npy, smi_embeddings)
            print("\n   Эмбеддинги также сохранены отдельно:")
            print(f"  - {seq_npy}")
            print(f"  - {smi_npy}")
        else:
            print(" Ошибка при получении эмбеддингов!")
