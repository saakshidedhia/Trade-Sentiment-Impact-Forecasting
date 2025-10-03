import os
# 1. Limit parallelism to avoid backend crashes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# 2. Load and preprocess articles
df = pd.read_excel("data/scraped_articles.xlsx")

# Fill Text field with primary or fallback, then drop missing
df["Text"] = df["Article_Text"].fillna(df["Article body (partial)"])
df = df.dropna(subset=["Text"])

# 3. Combine duplicate texts but merge source titles
df_unique = df.groupby("Text", as_index=False).agg({
    "Publish date": "first",
    "Publish time": "first",
    "Article title": "first",  # assumes title same for duplicates
    "Sentiment": "mean",
    "Source title": lambda s: "; ".join(sorted(s.unique()))
})

# Assign new article numbers and timestamp
df_unique = df_unique.sort_values(by=["Publish date", "Publish time"]).reset_index(drop=True)
df_unique["Article_Number"] = range(1, len(df_unique) + 1)
df_unique["Timestamp"] = pd.to_datetime(df_unique["Publish date"].astype(str) + " " +
                                        df_unique["Publish time"].astype(str))

# 4. Load FinBERT on CPU
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to("cpu")
model.eval()

# 5. Batch embedding extraction
batch_size = 16
embeddings = []
for i in tqdm(range(0, len(df_unique), batch_size), desc="Embedding batches"):
    batch_texts = df_unique["Text"].iloc[i : i + batch_size].tolist()
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

# 6. Build DataFrame of embeddings
num_dims = embeddings[0].shape[0]
embed_cols = [f"finbert_{i+1}" for i in range(num_dims)]
embedding_df = pd.DataFrame(embeddings, columns=embed_cols)

# 7. Combine with metadata
df_final = pd.concat([
    df_unique[["Article_Number", "Timestamp", "Article title", "Text", "Sentiment", "Source title"]].reset_index(drop=True),
    embedding_df
], axis=1)

# 8. Save to CSV and Excel
os.makedirs("output", exist_ok=True)
csv_path = "output/articles_with_finbert.csv"
xlsx_path = "output/articles_with_finbert.xlsx"
df_final.to_csv(csv_path, index=False)
df_final.to_excel(xlsx_path, index=False)

print(f"✅ Embeddings extracted and saved to {csv_path} and {xlsx_path}")
