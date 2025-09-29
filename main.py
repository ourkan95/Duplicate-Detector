import pandas as pd
from tabulate import tabulate

import address_parsing as address_parser
import address_similarity as address_sim
import geo_similarity as geo_sim
import bert_fuzzy_name_similarity as name_sim
import mismatch_checker as mismatch   # NEW


def print_table(df, step_name, n=5):
    """Pretty-print DataFrame as table (first n rows)."""
    if df is None or len(df) == 0:
        print(f"{step_name}: no results\n")
        return
    print(f"\n{step_name}: showing first {n} rows (of {len(df)})")
    print(tabulate(df.head(n), headers="keys", tablefmt="pretty", showindex=False))
    print()


def main():
    print("\n1) Address parsing...")
    df_parsed = address_parser.main()
    print(f"Parsed {len(df_parsed)} addresses")
    print_table(df_parsed, "Parsed addresses", n=3)

    # id -> name & address mapping
    id_to_name = dict(zip(df_parsed["hotelId"], df_parsed["name"]))
    id_to_address = dict(zip(df_parsed["hotelId"], df_parsed["address_standardized"]))

    print("\n2) Address similarity...")
    df_addr = address_sim.main(df_parsed, threshold=0.75)
    df_addr["name1"] = df_addr["id1"].map(id_to_name)
    df_addr["name2"] = df_addr["id2"].map(id_to_name)
    df_addr["address1"] = df_addr["id1"].map(id_to_address)
    df_addr["address2"] = df_addr["id2"].map(id_to_address)
    print_table(df_addr, "Address similarity", n=5)

    print("\n3) Geo similarity...")
    df_geo = geo_sim.main(threshold=0.75)
    df_geo["name1"] = df_geo["id1"].map(id_to_name)
    df_geo["name2"] = df_geo["id2"].map(id_to_name)
    df_geo["address1"] = df_geo["id1"].map(id_to_address)
    df_geo["address2"] = df_geo["id2"].map(id_to_address)
    print_table(df_geo, "Geo similarity", n=5)

    print("\n4) Name similarity...")
    df_name = name_sim.main(threshold=0.75)
    df_name["name1"] = df_name["id1"].map(id_to_name)
    df_name["name2"] = df_name["id2"].map(id_to_name)
    df_name["address1"] = df_name["id1"].map(id_to_address)
    df_name["address2"] = df_name["id2"].map(id_to_address)
    print_table(df_name, "Name similarity", n=5)

    print("\n5) Combining results...")

    # Merge on common keys
    df_final = df_addr.merge(df_geo, on=["id1", "id2", "name1", "name2", "address1", "address2"], how="outer")
    df_final = df_final.merge(df_name, on=["id1", "id2", "name1", "name2", "address1", "address2"], how="outer")

    df_final = df_final.fillna(0.0)

    weights = {"addr_score": 0.25, "geo_sim": 0.5, "name_score": 0.25}

    df_final["combined_score"] = (
        df_final.get("addr_score", 0) * weights["addr_score"] +
        df_final.get("geo_sim", 0) * weights["geo_sim"] +
        df_final.get("name_score", 0) * weights["name_score"]
    )

    threshold = 0.75
    df_candidates = df_final[df_final["combined_score"] >= threshold]

    desired_cols = [
        "id1", "name1", "address1",
        "id2", "name2", "address2",
        "addr_score", "geo_sim", "name_score", "combined_score"
    ]
    df_candidates = df_candidates[[c for c in desired_cols if c in df_candidates.columns]]

    print_table(df_candidates, "Final candidates", n=10)

    df_candidates.to_excel("final_similarity_candidates.xlsx", index=False)
    print("\nResults saved to final_similarity_candidates.xlsx")

    # NEW: Mismatch detection
    print("\n6) URL mismatch check...")
    df_mismatch = mismatch.run_mismatch("hotels_with_prices.xlsx", threshold=0.75)
    
    # Select only relevant columns for display
    cols_to_show = ["hotelId", "name", "trivago_name_clean", "url_slug_clean", "similarity", "is_mismatch"]
    df_mismatch_display = df_mismatch[cols_to_show]
    
    print_table(df_mismatch_display[df_mismatch_display["is_mismatch"]], "Potential mismatches", n=10)
    
    # Save full + mismatches
    df_mismatch.to_excel("url_hybrid_similarity.xlsx", index=False)
    df_mismatch[df_mismatch["is_mismatch"]].to_excel("hybrid_mismatched_candidates.xlsx", index=False)
    print("\nMismatch results saved to url_hybrid_similarity.xlsx and hybrid_mismatched_candidates.xlsx")



if __name__ == "__main__":
    main()
