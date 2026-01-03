import os
import json
import pandas as pd
import pycountry
from kaggle.api.kaggle_api_extended import KaggleApi

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = f"{BASE}/data/raw"
PROCESSED = f"{BASE}/data/processed"

os.makedirs(RAW, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)

api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    "patateriedata/all-international-football-results", path=RAW, unzip=True
)

countries = (lambda d: (d.to_csv(f"{PROCESSED}/countries.csv", index=False), d)[1])(
    pd.read_csv(f"{RAW}/countries_names.csv")[["original_name", "current_name"]]
    .apply(lambda c: c.str.strip())
    .assign(
        iso_alpha=lambda df: df.current_name.map(
            lambda n: json.load(
                open(f"{BASE}/config/iso3_supp.json", encoding="utf-8")
            ).get(n)
            or (pycountry.countries.lookup(n).alpha_3 if n else None)
        )
    )
)

matches = (lambda d: (d.to_csv(f"{PROCESSED}/matches.csv", index=False), d)[1])(
    pd.read_csv(f"{RAW}/all_matches.csv")
    .replace(dict(zip(countries.original_name, countries.current_name)))
    .assign(date=lambda df: pd.to_datetime(df.date, errors="coerce"))
)
