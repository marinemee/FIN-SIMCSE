import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("data/sentences_2019.csv")
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Total sentences: {len(df)}")
print(f"Test sentences:  {len(test_df)}")
print(f"Unique companies in test: {test_df['company_id'].nunique()}")
print(f"Sections: {test_df['section'].value_counts().to_dict()}")


# -----------------------------
# 2. Build weak labels
# A sentence is relevant if it comes from the same company + same section
# -----------------------------
def build_relevance_dict(df):
    """
    Returns a dict:
    query_idx → set of relevant indices (same company + same section)
    """
    relevance = {}
    grouped = df.groupby(["company_id", "section"]).indices

    for (company, section), indices in grouped.items():
        indices = list(indices)
        for idx in indices:
            # all other sentences in the same group are relevant
            relevance[idx] = set(indices) - {idx}

    return relevance


relevance_dict = build_relevance_dict(test_df)
print(f"\nBuilt relevance labels for {len(relevance_dict)} queries")


# -----------------------------
# 3. Metrics
# -----------------------------
def recall_at_k(ranked_indices, relevant_set, k=5):
    if not relevant_set:
        return 0.0
    top_k = ranked_indices[:k]
    hits = sum(1 for i in top_k if i in relevant_set)
    return hits / min(k, len(relevant_set))   # or just / len(relevant_set)


def mrr(ranked_indices, relevant_set):
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(similarity_matrix, relevance_dict, k_list=[1, 5, 10]):
    recalls = {k: [] for k in k_list}
    mrr_scores = []

    n = similarity_matrix.shape[0]

    for q in tqdm(range(n), desc="Evaluating"):
        relevant = relevance_dict.get(q, set())
        if not relevant:
            continue

        scores = similarity_matrix[q].copy()
        scores[q] = -np.inf          # exclude self

        ranked = np.argsort(scores)[::-1]   # highest similarity first

        for k in k_list:
            recalls[k].append(recall_at_k(ranked, relevant, k))

        mrr_scores.append(mrr(ranked, relevant))

    results = {
        f"Recall@{k}": np.mean(recalls[k]) for k in k_list
    }
    results["MRR"] = np.mean(mrr_scores)
    return results


# -----------------------------
# 4. TF-IDF Baseline
# -----------------------------
print("\n=== TF-IDF Baseline ===")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(test_df["sentence"])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Full similarity matrix (ok for ~7k sentences)
similarity_matrix = cosine_similarity(tfidf_matrix)

results = evaluate_retrieval(similarity_matrix, relevance_dict)

print("\nTF-IDF Results:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}")


# -----------------------------
# 5. MiniLM Baseline
# -----------------------------
print("\n=== MiniLM Baseline ===")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode all test sentences
embeddings = model.encode(
    test_df["sentence"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("MiniLM embeddings shape:", embeddings.shape)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)

results = evaluate_retrieval(similarity_matrix, relevance_dict)

print("\nMiniLM Results:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}")    