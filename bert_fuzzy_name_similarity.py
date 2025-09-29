import pandas as pd
import re
from itertools import combinations
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------
# 1. Normalize function
# ----------------------
STOPWORDS = {"hotel", "tokyo", "ginza", "apartment", "inn", "guesthouse", "west", "east"}


def normalize_name(name):
    """Otel ismini normalize eder (küçük harf, noktalama sil, stopword çıkar)."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    tokens = [t for t in name.split() if t not in STOPWORDS]
    return " ".join(tokens)


def main(input_file="hotels_with_prices.xlsx", threshold=0.75):
    df = pd.read_excel(input_file)
    names_raw = df["name"].astype(str).tolist()
    names = [normalize_name(n) for n in names_raw]

    # ----------------------
    # 2. BERT embeddings
    # ----------------------
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(names, convert_to_numpy=True, normalize_embeddings=True)
    bert_sim_matrix = cosine_similarity(embeddings)

    # ----------------------
    # 3. TF-IDF n-gram similarity
    # ----------------------
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
    X_tfidf = tfidf.fit_transform(names)
    tfidf_sim_matrix = cosine_similarity(X_tfidf)

    # ----------------------
    # 4. Pairwise similarity hesapla
    # ----------------------
    pairs = []
    for i, j in combinations(range(len(names)), 2):
        name1, name2 = names[i], names[j]

        # BERT similarity
        bert_sim = float(bert_sim_matrix[i, j])

        # Fuzzy similarity
        fuzzy_sim = fuzz.token_set_ratio(name1, name2) / 100.0

        # TF-IDF similarity
        tfidf_sim = float(tfidf_sim_matrix[i, j])

        # Hibrit skor (ağırlıklar ayarlanabilir)
        final_score = 0.5 * bert_sim + 0.3 * fuzzy_sim + 0.2 * tfidf_sim

        # Kontrollü substring bonus
        tokens1 = name1.split()
        tokens2 = name2.split()
        if set(tokens1) & set(tokens2):
            final_score = min(1.0, final_score + 0.05)

        if final_score >= threshold:
            pairs.append({
                "id1": df.loc[i, "hotelId"],
                "id2": df.loc[j, "hotelId"],
                "name_score": round(final_score, 3)
            })

    return pd.DataFrame(pairs)


if __name__ == "__main__":
    df = main()
    print(df.head())
