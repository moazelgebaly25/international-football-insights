import os, pandas as pd, numpy as np, streamlit as st

st.set_page_config(layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))

team = st.sidebar.selectbox(
    "Select a team:",
    sorted(
        country := pd.read_csv(f"{BASE}/data/raw/countries_names.csv")["current_name"]
        .dropna()
        .unique()
    ),
    index=sorted(country).index("Egypt") if "Egypt" in country else 0,
)

team_matches = (
    pd.read_csv(f"{BASE}/data/processed/matches.csv", parse_dates=["date"])
    .loc[lambda df: df[["home_team", "away_team"]].eq(team).any(axis=1)]
    .assign(opponent=lambda df: df.away_team.where(df.home_team == team, df.home_team))
)

date_range = st.sidebar.date_input(
    "Select date range:",
    value=(
        max(pd.Timestamp("2001-03-19"), team_matches["date"].min()),
        min(pd.Timestamp("2013-11-19"), team_matches["date"].max()),
    ),
    min_value=team_matches["date"].min(),
    max_value=team_matches["date"].max(),
)

opponent = st.sidebar.selectbox(
    "Select an opponent:",
    ["All"]
    + sorted(
        team_matches.loc[lambda df: df["date"].between(*map(pd.Timestamp, date_range))][
            "opponent"
        ].unique()
    ),
)

tournament = st.sidebar.selectbox(
    "Select a tournament:",
    ["All"]
    + sorted(
        team_matches.loc[
            lambda df: (df["opponent"].eq(opponent) | (opponent == "All"))
            & df["date"].between(*map(pd.Timestamp, date_range))
        ]["tournament"].unique()
    ),
)

df = (
    team_matches.loc[
        lambda df: ((df["opponent"] == opponent) if opponent != "All" else True)
        & ((df["tournament"] == tournament) if tournament != "All" else True)
        & df["date"].between(*map(pd.Timestamp, date_range))
    ]
    .assign(
        venue=lambda df: [
            "Home" if c == team else "Away" if o == opponent else "Neutral"
            for c, o in zip(
                df.country, df.away_team.where(df.home_team == team, df.home_team)
            )
        ],
        result=lambda df: np.select(
            [
                ((df.home_team == team) & (df.home_score > df.away_score))
                | ((df.away_team == team) & (df.away_score > df.home_score)),
                ((df.home_team == team) & (df.home_score < df.away_score))
                | ((df.away_team == team) & (df.away_score < df.home_score)),
            ],
            ["W", "L"],
            default="D",
        ),
    )[["date", "tournament", "opponent", "venue", "home_score", "away_score", "result"]]
    .assign(date=lambda df: df.date.dt.date)
    .reset_index(drop=True)
)

st.dataframe(df)
