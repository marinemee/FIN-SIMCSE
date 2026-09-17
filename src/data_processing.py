from datasets import load_dataset
import spacy
import pandas as pd
import re
import os


# -----------------------------
# 1. Load the 2019 dataset
# -----------------------------

ds = load_dataset(
    "eloukas/edgar-corpus",
    name="year_2019",
    split="train",
    trust_remote_code=True
)

# Small test: only 100 filings
ds = ds.select(range(100))

print(f"Loaded {len(ds)} filings")


# -----------------------------
# 2. Load spaCy
# -----------------------------

nlp = spacy.load("en_core_web_sm")


# -----------------------------
# 3. Helper function
# -----------------------------

def mostly_numeric(text):
    text_no_spaces = re.sub(r"\s+", "", text)

    if not text_no_spaces:
        return True

    digits = sum(char.isdigit() for char in text_no_spaces)

    return digits / len(text_no_spaces) > 0.40


# -----------------------------
# 4. Extract and split
# -----------------------------

records = []

for i, row in enumerate(ds):

    company_id = row["cik"]
    year = row["year"]

    sections = {
        "1A": row["section_1A"],
        "7": row["section_7"]
    }

    for section, text in sections.items():

        if not text:
            continue

        doc = nlp(text)

        for sent in doc.sents:

            sentence = sent.text.strip()

            # Remove excessive whitespace
            sentence = re.sub(r"\s+", " ", sentence)

            # Remove very short sentences
            if len(sentence) < 30:
                continue

            # Remove very long sentences
            if len(sentence) > 500:
                continue

            # Remove mostly-numeric sentences
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
# 5. Create DataFrame
# -----------------------------

df = pd.DataFrame(records)

print(f"\nSentences before deduplication: {len(df)}")


# -----------------------------
# 6. Remove exact duplicates
# -----------------------------

df = df.drop_duplicates(subset=["sentence"])

print(f"Sentences after deduplication: {len(df)}")


# -----------------------------
# 7. Save CSV
# -----------------------------

os.makedirs("data", exist_ok=True)

output_path = "data/sentences_2019.csv"

df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")