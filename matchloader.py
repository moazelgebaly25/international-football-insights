import os, sys, pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

BASE = os.path.dirname(os.path.abspath(__file__))
RAW, PROCESSED = f"{BASE}/data/raw", f"{BASE}/data/processed"

os.makedirs(RAW, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)

api = KaggleApi()
api.authenticate()
sys.stdout = open(os.devnull, "w")
api.dataset_download_files(
    "patateriedata/all-international-football-results", path=RAW, unzip=True
)

df = (
    pd.read_csv(f"{RAW}/all_matches.csv")
    .replace(
        dict(
            zip(
                *pd.read_csv(f"{RAW}/countries_names.csv")[
                    ["original_name", "current_name"]
                ].apply(lambda c: c.str.strip())
            )
        )
    )
    .assign(date=lambda x: pd.to_datetime(x.date, errors="coerce"))
)

df.to_csv(f"{PROCESSED}/matches.csv", index=False)
