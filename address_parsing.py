import pandas as pd
from postal.parser import parse_address


def normalize_libpostal(parsed, city_from_excel=None):
    """Libpostal çıktısını normalize eder."""
    return {
        "house_number": parsed.get("house_number", ""),
        "area": parsed.get("suburb", "") or parsed.get("road", ""),
        "city_district": parsed.get("city_district", ""),
        "city": city_from_excel if city_from_excel else parsed.get("city", ""),
        "postcode": parsed.get("postcode", ""),
        "country": parsed.get("country", "")
    }


def main(input_file="hotels_with_prices.xlsx"):
    df = pd.read_excel(input_file)

    parsed_records = []
    for _, row in df.iterrows():
        addr = row["address_standardized"]
        if not isinstance(addr, str) or addr.strip() == "":
            parsed_norm = normalize_libpostal({}, city_from_excel=row.get("city", ""))
        else:
            parsed_list = parse_address(addr)
            parsed = {label: value for value, label in parsed_list}
            parsed_norm = normalize_libpostal(parsed, city_from_excel=row.get("city", ""))
        parsed_records.append(parsed_norm)

    df_parsed = pd.DataFrame(parsed_records)
    df_final = pd.concat([df, df_parsed], axis=1)

    if "city.1" in df_final.columns:
        df_final = df_final.drop(columns=["city.1"])

    return df_final


if __name__ == "__main__":
    df = main()
    print(df.head())
