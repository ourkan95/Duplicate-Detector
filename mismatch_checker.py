import pandas as pd
import re
from urllib.parse import urlparse, unquote
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
import numpy as np

# Load multilingual embedding model
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Stopwords (generic + Japan locations for now)
STOPWORDS = {
    "hotel", "hotels", "apartments", "apartment", "entire", "house", "room",
    "hostel", "resort", "resorts", "inn", "guesthouse", "bnb", "bed", "breakfast", "the",
    "tokyo", "japan", "jp", "ja",
    "roppongi", "ginza", "shinjuku", "shibuya", "ikebukuro", "ueno", "akihabara",
    "asakusa", "asakusabashi", "nihonbashi", "kanda", "marunouchi", "shinagawa",
    "ebisu", "meguro", "setagaya", "chiyoda", "chuo", "minato", "taito",
    "osaka", "kyoto", "yokohama", "nagoya", "sapporo", "fukuoka", "hiroshima", "kobe",
    "ku", "cho", "machi", "dori", "eki", "station"
}


def clean_text(text: str) -> str:
    """Normalize and clean hotel name or slug text"""
    if not text:
        return ""
    text = unquote(text)
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\.html.*$", "", text)  # remove .html and suffixes
    text = re.sub(r"(hotel beschreibung|reviews|en gb|ja jp)$", "", text, flags=re.I)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # remove non-alphanumeric chars
    text = re.sub(r"\b([a-z]{2})-([a-z]{2})\b", "", text, flags=re.I)  # remove locale like de-de
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def extract_slug_name(url: str) -> str:
    """Extract the hotel slug name from different dealer URLs"""
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split("/") if p]
    slug = path_parts[-1] if path_parts else ""

    # Expedia: hotel name is usually in the second-to-last segment
    if "expedia" in parsed.netloc.lower():
        if len(path_parts) >= 2:
            slug = path_parts[-2]
        slug = slug.split(".")[0]

    # Booking: take the part before .html
    elif "booking" in parsed.netloc.lower():
        slug = slug.split(".")[0]

    # Agoda: hotel name appears in the path (e.g. /<locale>/<hotel-name>/hotel/<city-country>.html)
    elif "agoda" in parsed.netloc.lower():
        for i, part in enumerate(path_parts):
            if re.match(r"^[a-z]{2}-[a-z]{2}$", part, flags=re.I):  # skip locale
                continue
            if i + 1 < len(path_parts) and path_parts[i + 1].lower() == "hotel":
                slug = part
                break
        slug = slug.split(".")[0]

    # Trivago: slug appears after /oar/
    elif "trivago" in parsed.netloc.lower() and "oar" in path_parts:
        idx = path_parts.index("oar")
        if idx + 1 < len(path_parts):
            slug = path_parts[idx + 1]

    return slug


def hybrid_similarity_batch(names_clean, slugs_clean):
    """Compute hybrid similarity in batch using SBERT + Fuzzy"""
    emb_names = model.encode(names_clean, convert_to_tensor=True, device="cpu", batch_size=32)
    emb_slugs = model.encode(slugs_clean, convert_to_tensor=True, device="cpu", batch_size=32)

    sbert_scores = util.cos_sim(emb_names, emb_slugs).diagonal().cpu().numpy()

    fuzzy_scores = np.array([
        fuzz.token_set_ratio(n, s) / 100.0 if n and s else 0.0
        for n, s in zip(names_clean, slugs_clean)
    ])

    final_scores = 0.7 * sbert_scores + 0.3 * fuzzy_scores
    return final_scores


def run_mismatch(input_file="hotels_with_prices.xlsx", threshold=0.9):
    """Check URL slug vs Trivago hotel name and return DataFrame with mismatch flag"""
    df = pd.read_excel(input_file)

    # Pre-clean names and slugs
    df["trivago_name_clean"] = df["name"].astype(str).apply(clean_text)
    df["url_slug_name"] = df["dealUrl"].astype(str).apply(extract_slug_name)
    df["url_slug_clean"] = df["url_slug_name"].apply(clean_text)

    # Compute similarities
    df["similarity"] = hybrid_similarity_batch(
        df["trivago_name_clean"].tolist(),
        df["url_slug_clean"].tolist()
    )
    df["similarity"] = df["similarity"].round(3)

    # Mark mismatches
    df["is_mismatch"] = df["similarity"] < threshold

    return df
