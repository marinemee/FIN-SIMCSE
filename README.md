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


## Results

We evaluated three approaches on the test set using section-level weak labels (same company + same section):

| Model               | Recall@1 | Recall@5 | Recall@10 | MRR    |
|---------------------|----------|----------|-----------|--------|
| TF-IDF              | 0.5677   | 0.4741   | 0.4192    | 0.6854 |
| MiniLM (off-the-shelf) | 0.5636 | 0.4745   | 0.4229    | 0.6796 |
| SimCSE (fine-tuned) | 0.5352   | 0.4258   | 0.3736    | 0.6513 |

### Findings

- TF-IDF performed strongly, indicating high lexical overlap within the same filing section.
- Off-the-shelf MiniLM performed almost identically to TF-IDF.
- Unsupervised SimCSE fine-tuning (batch size 16, 1 epoch, CPU) underperformed both baselines.

### Limitations

- Small batch size limited the quality of in-batch negatives.
- Training was constrained to CPU, which restricted model size and batch size.
- Weak labels based on section membership are noisy proxies for true semantic equivalence.
- Loss collapsed rapidly, suggesting the contrastive task became too easy under the chosen settings.

### Conclusion

Under the current experimental constraints, domain-specific contrastive fine-tuning did not yield gains over simpler baselines. The result is still informative: it shows that careful evaluation can reveal when a popular method fails to transfer under limited compute and weak supervision.
## Future Work

Several directions could improve upon the current results:

1. **Larger batch sizes**  
   Training with bigger batches (or gradient accumulation) would provide harder in-batch negatives and may prevent the rapid loss collapse observed in this experiment.

2. **Hard negative mining**  
   Instead of relying only on in-batch negatives, retrieving lexically similar but semantically different sentences (e.g., via BM25) could make the contrastive task more informative.

3. **Weakly supervised positives**  
   Using same-section sentences as positive pairs (in addition to the dropout-based unsupervised signal) could be tested as an ablation.

4. **Better evaluation labels**  
   The current weak labels (same company + same section) are noisy. Creating a small set of human-verified paraphrase pairs, or using an external dataset such as FiQA, would allow cleaner measurement of semantic retrieval quality.

5. **Domain-adapted backbone**  
   Starting from a finance-specific encoder (e.g., FinBERT) instead of general-domain MiniLM may give the contrastive fine-tuning a stronger initialization.

6. **Scale and compute**  
   Running the same experiment with a larger model and more data on GPU would clarify whether the current underperformance is mainly due to compute and batch-size limitations.
