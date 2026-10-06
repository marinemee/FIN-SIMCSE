from datasets import load_dataset
import spacy
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load the 2019 dataset
# -----------------------------
ds = load_dataset(
    "eloukas/edgar-corpus",
    name="year_2019",
    split="train",
    trust_remote_code=True
)

# Start small for testing
ds = ds.select(range(100))
print(f"Loaded {len(ds)} filings")

# -----------------------------
# 2. Load spaCy
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# 3. Helper functions
# -----------------------------
def mostly_numeric(text, threshold=0.40):
    text_clean = re.sub(r"\s+", "", text)
    if not text_clean:
        return True
    digits = sum(c.isdigit() for c in text_clean)
    return (digits / len(text_clean)) > threshold

def clean_sentence(text):
    return re.sub(r"\s+", " ", text).strip()

# -----------------------------
# 4. Extract sentences
# -----------------------------
records = []

for i, row in enumerate(ds):
    company_id = str(row["cik"])
    year = row["year"]

    sections = {
        "1A": row.get("section_1A", ""),
        "7": row.get("section_7", "")
    }

    for section, text in sections.items():
        if not text or not isinstance(text, str):
            continue

        doc = nlp(text)

        for sent in doc.sents:
            sentence = clean_sentence(sent.text)

            if len(sentence) < 30 or len(sentence) > 500:
                continue
            if mostly_numeric(sentence):
                continue

            records.append({
                "sentence": sentence,
                "company_id": company_id,
                "section": section,
                "year": year
            })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(ds)} filings")

# -----------------------------
# 5. Create DataFrame + Deduplicate
# -----------------------------
df = pd.DataFrame(records)
print(f"\nSentences before deduplication: {len(df)}")

df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
print(f"Sentences after deduplication: {len(df)}")

# -----------------------------
# 6. Split by company (prevent leakage)
# -----------------------------
unique_companies = np.array(df["company_id"].unique())

train_comps, temp_comps = train_test_split(unique_companies, test_size=0.30, random_state=42)
val_comps, test_comps = train_test_split(temp_comps, test_size=0.50, random_state=42)

train_comps = set(train_comps)
val_comps = set(val_comps)
test_comps = set(test_comps)

def assign_split(cid):
    if cid in train_comps:
        return "train"
    elif cid in val_comps:
        return "val"
    else:
        return "test"

df["split"] = df["company_id"].apply(assign_split)

print("\nSplit distribution:")
print(df["split"].value_counts())
print(f"\nNumber of companies → Train: {len(train_comps)}, Val: {len(val_comps)}, Test: {len(test_comps)}")

# -----------------------------
# 7. Save
# -----------------------------
os.makedirs("data", exist_ok=True)
output_path = "data/sentences_2019.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")
print(df.head())