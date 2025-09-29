import pandas as pd
from itertools import combinations
from sentence_transformers import SentenceTransformer, util
import torch

# device otomatik seçilsin
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)


def numeric_similarity(num1, num2):
    if not num1 or not num2:
        return None
    try:
        parts1 = num1.split('-')
        parts2 = num2.split('-')

        if parts1[0] != parts2[0]:
            return 0.0
        if len(parts1) > 1 and len(parts2) > 1 and parts1[1] != parts2[1]:
            return 0.5
        if len(parts1) > 2 and len(parts2) > 2:
            g1, g2 = int(parts1[2]), int(parts2[2])
            if g1 == g2:
                return 1.0
            diff = abs(g1 - g2)
            if diff == 1:
                return 0.9
            elif 2 <= diff <= 4:
                return 0.7
            elif 5 <= diff <= 9:
                return 0.5
            else:
                return 0.3
        return 0.8
    except:
        return 0.0


def weighted_average(similarities, weights):
    total, total_w = 0.0, 0.0
    for key, sim in similarities.items():
        if sim is not None:
            total += weights.get(key, 0) * sim
            total_w += weights.get(key, 0)
    return total / total_w if total_w > 0 else 0.0


def main(df_parsed: pd.DataFrame, threshold=0.75):
    """Parse edilmiş DataFrame üzerinden adres benzerliği hesaplar."""
    df_parsed = df_parsed.reset_index(drop=True)  # 🔑 index resetlendi

    text_fields = ["address_standardized", "area", "city_district"]
    embeddings, cos_matrices = {}, {}

    with torch.no_grad():
        for field in text_fields:
            values = df_parsed[field].astype(str).tolist()
            emb = model.encode(values, convert_to_tensor=True, device=device, batch_size=32)
            embeddings[field] = emb
            cos_matrices[field] = util.cos_sim(emb, emb).cpu().numpy()

    pairs = []
    for i, j in combinations(df_parsed.index, 2):
        if df_parsed.loc[i, "address_standardized"].strip().lower() == df_parsed.loc[j, "address_standardized"].strip().lower():
            final_score = 1.0
        else:
            sims = {
                "num": numeric_similarity(df_parsed.loc[i, "house_number"], df_parsed.loc[j, "house_number"]),
                "area": cos_matrices["area"][i, j],
                "district": cos_matrices["city_district"][i, j],
                "postcode": 1.0 if str(df_parsed.loc[i, "postcode"]) == str(df_parsed.loc[j, "postcode"]) and pd.notna(df_parsed.loc[i, "postcode"]) else None,
                "city": 1.0 if str(df_parsed.loc[i, "city"]).strip().lower() == str(df_parsed.loc[j, "city"]).strip().lower() and pd.notna(df_parsed.loc[i, "city"]) else 0.0,
                "country": 1.0 if str(df_parsed.loc[i, "country"]).strip().lower() == str(df_parsed.loc[j, "country"]).strip().lower() and pd.notna(df_parsed.loc[i, "country"]) else None,
            }

            weights = {"num": 0.5, "area": 0.07, "district": 0.05, "postcode": 0.03, "city": 0.03, "country": 0.02}
            parsed_score = weighted_average(sims, weights)
            raw_score = cos_matrices["address_standardized"][i, j]
            final_score = 0.7 * parsed_score + 0.3 * raw_score

        if final_score >= threshold:
            pairs.append({
                "id1": df_parsed.loc[i, "hotelId"], "id2": df_parsed.loc[j, "hotelId"],
                "addr_score": round(final_score, 3)
            })

    return pd.DataFrame(pairs)


if __name__ == "__main__":
    import address_parsing
    df_parsed = address_parsing.main()
    df = main(df_parsed)
    print(df.head())
