# Fin-SimCSE: Contrastive Sentence Embeddings for Financial Disclosure Retrieval

## What is this project?

Financial reports (like 10-K filings) often describe the same risks or events using very different wording. Traditional keyword search fails in these cases.

This project fine-tunes a language model using **contrastive learning** (SimCSE-style) so that sentences with similar financial meaning end up close together in embedding space — even if they share almost no words.

## Goal

Build domain-specific sentence embeddings for financial text and test whether they improve retrieval and clustering compared to:
- Simple keyword methods (TF-IDF)
- General-purpose embeddings (MiniLM)
- Existing financial models (FinBERT)

## Method (Simple Version)

1. Take sentences from real company filings (EDGAR-CORPUS)
2. Train a model so that two slightly different views of the same sentence become similar
3. Push unrelated sentences apart
4. Evaluate how well the model can find related financial statements

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- sentence-transformers
- scikit-learn
- EDGAR-CORPUS dataset

## Project Status

Currently in development.
## Structure

```text
├── data/               # Processed sentence data
├── src/
│   ├── data_processing.py
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/          # Experiments & analysis
├── results/            # Metrics & plots
└── README.md
```


## Evaluation Metrics

- Recall@K
- Mean Reciprocal Rank (MRR)
- Silhouette Score
- Normalized Mutual Information (NMI)

## Notes

This is an applied undergraduate research project.  
It combines existing methods (SimCSE + financial text) and carefully evaluates them — it does not claim a new architecture.
It combines existing methods (SimCSE + financial text) and carefully evaluates them — it does not claim a new architecture.
