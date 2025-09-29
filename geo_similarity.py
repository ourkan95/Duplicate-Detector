import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from itertools import combinations


def haversine_m(lat1, lon1, lat2, lon2):
    """İki nokta arasındaki haversine mesafesini (metre) döndürür."""
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def main(input_file="hotels_with_prices.xlsx", threshold=0.75):
    df_hotels = pd.read_excel(input_file)

    records = []
    for i, j in combinations(range(len(df_hotels)), 2):
        h1, h2 = df_hotels.iloc[i], df_hotels.iloc[j]
        dist_m = haversine_m(h1["latitude"], h1["longitude"], h2["latitude"], h2["longitude"])

        # benzerlik hesaplama (distance → score)
        if dist_m <= 30:
            geo_sim = 1.0
        elif dist_m <= 100:
            geo_sim = 0.8
        elif dist_m <= 200:
            geo_sim = 0.6
        elif dist_m <= 300:
            geo_sim = 0.5
        else:
            geo_sim = 0.0

        if geo_sim >= threshold:  # threshold uygula
            records.append({
                "id1": h1["hotelId"], "id2": h2["hotelId"],
                "geo_distance_m": round(dist_m, 1),
                "geo_sim": round(geo_sim, 3)
            })

    df_geo = pd.DataFrame(records)
    return df_geo


if __name__ == "__main__":
    df = main()
    print(df.head())
