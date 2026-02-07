import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import time


df = pd.read_csv('neg_pairs.csv')
smiles_list = df['canonical_smiles'].fillna('').tolist()

# Удаляем пустые SMILES
smiles_list = [s.strip() for s in smiles_list if isinstance(s, str) and s.strip()]
print(f"Found {len(smiles_list)} valid SMILES")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка модели
print("Loading ChemBERTa...")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model.to(device)
model.eval()


print(f"Encoding {len(smiles_list)} SMILES...")
batch_size = 32
all_embeddings = []

start_time = time.time()

for i in range(0, len(smiles_list), batch_size):
    batch = smiles_list[i:i+batch_size]

    # Токенизация
    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    # Энкодинг
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings.cpu().numpy())

    # Прогресс
    processed = min(i + batch_size, len(smiles_list))
    if (i // batch_size) % 10 == 0:
        print(f"  Processed: {processed}/{len(smiles_list)}", end='\r')

# Объединение результатов
embeddings_array = np.vstack(all_embeddings)
total_time = time.time() - start_time

print(f"\n  Done in {total_time:.1f}s ({len(smiles_list)/total_time:.1f} mol/s)")
print(f"   Shape: {embeddings_array.shape}")

# 5. Создаем DataFrame с эмбеддингами
print("\nCreating DataFrame...")

embedding_dim = embeddings_array.shape[1]
embedding_columns = [f'emb_{i}' for i in range(embedding_dim)]

# Создаем DataFrame
embeddings_df = pd.DataFrame(embeddings_array, columns=embedding_columns)
embeddings_df.insert(0, 'canonical_smiles', smiles_list)  # Добавляем SMILES в первую колонку

# Сохраняем в CSV
output_file = 'SMILES_negative.csv'
embeddings_df.to_csv(output_file, index=False)

# Доп информация
info_file = 'aptabench_embeddings_info.txt'
with open(info_file, 'w') as f:
    f.write(f"Embeddings generated: {time.ctime()}\n")
    f.write(f"Total molecules: {len(smiles_list)}\n")
    f.write(f"Embedding dimension: {embedding_dim}\n")
    f.write(f"Model: seyonec/ChemBERTa-zinc-base-v1\n")
    f.write(f"Processing time: {total_time:.2f} seconds\n")
    f.write(f"Speed: {len(smiles_list)/total_time:.2f} molecules/second\n")

print(f"\n  Saved to CSV:")
print(f"  - {output_file} ({len(embeddings_df)} rows, {len(embeddings_df.columns)} columns)")

# 8. Проверка сохранения
print(f"\n  Loading back for verification...")
loaded_df = pd.read_csv(output_file)
print(f"  Loaded: {len(loaded_df)} rows, {len(loaded_df.columns)} columns")
print(f"  First 5 columns: {list(loaded_df.columns[:5])}")

print(f"\n  Sample row:")
print(f"  SMILES: {loaded_df['canonical_smiles'].iloc[0][:50]}...")
print(f"  First 5 embedding values:")
for i in range(5):
    col_name = f'emb_{i}'
    print(f"    {col_name}: {loaded_df[col_name].iloc[0]:.6f}")
