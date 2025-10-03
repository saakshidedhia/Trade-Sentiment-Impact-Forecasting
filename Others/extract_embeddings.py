import os

# 1. Limit parallelism to avoid backend crashes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# 2. Load all articles
df = pd.read_excel("data/scraped_articles.xlsx")

# Fill Text field with primary or fallback, but DON'T drop any row
df["Text"] = df["Article_Text"].fillna(df["Article body (partial)"])

# 3. Create Timestamp (combine Publish_date and Publish_time)
df["Timestamp"] = pd.to_datetime(df["Publish_date"].astype(str) + " " + df["Publish_time"].astype(str))

# 4. Load FinBERT model on CPU
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to("cpu")
model.eval()

# 5. Extract embeddings (batch processing)
batch_size = 16
embeddings = []

for i in tqdm(range(0, len(df), batch_size), desc="Embedding batches"):
    batch_texts = df["Text"].iloc[i : i + batch_size].fillna("").tolist()  # Fill empty text to ""
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    batch_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    embeddings.extend(batch_embs)

# 6. Create Embedding DataFrame
num_dims = embeddings[0].shape[0]
embed_cols = [f"finbert_{i+1}" for i in range(num_dims)]
embedding_df = pd.DataFrame(embeddings, columns=embed_cols)

# 7. Combine embeddings with original df
df_final = pd.concat([
    df.reset_index(drop=True),
    embedding_df
], axis=1)

# 8. Save to output folder
os.makedirs("output", exist_ok=True)
csv_path = "output/articles_with_finbert_no_filter.csv"
xlsx_path = "output/articles_with_finbert_no_filter.xlsx"

df_final.to_csv(csv_path, index=False)
df_final.to_excel(xlsx_path, index=False)

print(f"✅ All embeddings extracted and saved to {csv_path} and {xlsx_path}")
