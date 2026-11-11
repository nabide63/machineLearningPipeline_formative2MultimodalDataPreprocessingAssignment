"""Product Recommendation Model: inference helper for the command-line app.

The model and feature store are produced by `task1_data_merge.ipynb` (Section 11).
In the full system flow this runs AFTER the face model recognises the user and
BEFORE the voice model approves showing the prediction.

Usage:
    python scripts/product_model.py --customer-id 125
    python scripts/product_model.py --customer-id 125 --top-k 5

From another script:
    from scripts.product_model import recommend_products
    result = recommend_products(125)
"""
import argparse
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "product_recommender.joblib"
FEATURE_STORE_PATH = ROOT / "data" / "customer_feature_store.csv"


class UnknownCustomerError(KeyError):
    """Raised when the customer ID is not in the feature store, so the app should deny access."""


@lru_cache(maxsize=1)
def _load():
    # Load once and reuse on later calls
    bundle = joblib.load(MODEL_PATH)
    store = pd.read_csv(FEATURE_STORE_PATH).set_index("customer_id")
    return bundle, store


def recommend_products(customer_id, top_k=3):
    """Return the customer's top_k product categories with probabilities.

    Customers without a social profile get the overall purchase shares (popularity fallback),
    because the model was trained only on customers who have one.
    """
    bundle, store = _load()
    customer_id = int(customer_id)
    if customer_id not in store.index:
        raise UnknownCustomerError(f"Customer {customer_id} is not in the feature store")

    row = store.loc[[customer_id]]
    if bool(row["has_social_profile"].iloc[0]):
        proba = pd.Series(bundle["model"].predict_proba(row[bundle["features"]])[0], index=bundle["classes"])
        source = bundle["model_name"]
    else:
        proba = pd.Series(bundle["fallback_distribution"])
        source = "popularity fallback (no social profile)"

    top = proba.sort_values(ascending=False).head(top_k)
    return {
        "customer_id": customer_id,
        "top_product": top.index[0],
        "recommendations": [(cat, round(float(p), 3)) for cat, p in top.items()],
        "source": source,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict the product category a customer is likely to buy.")
    parser.add_argument("--customer-id", required=True, type=int)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        result = recommend_products(args.customer_id, args.top_k)
    except UnknownCustomerError as err:
        raise SystemExit(f"ACCESS DENIED: {err}")
    print(f"Customer {result['customer_id']}  [{result['source']}]")
    for rank, (category, prob) in enumerate(result["recommendations"], start=1):
        print(f"  {rank}. {category:<12} {prob:.1%}")


if __name__ == "__main__":
    main()
