import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TERRITORY_PATH = f"{BASE_DIR}/config/territory_coords.json"
ALPHA = 3
DEFAULT_TEAM = "Egypt"
LEVELS = ["Easy", "Medium", "Hard", "Very Hard"]
ALL_LEVELS = LEVELS + ["No Data", "Scatter Data"]
COLORS = {
    "Easy": "#2ECC71",
    "Medium": "#F1C40F",
    "Hard": "#E67E22",
    "Very Hard": "#E74C3C",
    "No Data": "#FFFFFF",
    "Scatter Data": "#EBF2F5",
}
OCEAN_COLOR = "#CAF3F9"
LINE_COLOR = "black"

st.set_page_config(layout="wide")

countries_df = pd.read_csv(f"{BASE_DIR}/data/processed/countries.csv")
countries = sorted(countries_df["current_name"].dropna().unique())
matches = pd.read_csv(f"{BASE_DIR}/data/processed/matches.csv", parse_dates=["date"])

iso_map = (
    countries_df.drop_duplicates("current_name")
    .set_index("current_name")["iso_alpha"]
    .to_dict()
)
iso_counts = countries_df.groupby("iso_alpha")["current_name"].nunique()
complex_iso = iso_counts[iso_counts > 1].index.tolist()
complex_countries = {
    iso: countries_df[countries_df["iso_alpha"] == iso]["current_name"].tolist()
    for iso in complex_iso
}

with open(TERRITORY_PATH, "r", encoding="utf-8") as f:
    coords = json.load(f)


def calc_stats(df, opponent_col="opponent"):
    return (
        df.groupby(opponent_col)["result"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["W", "D", "L"], fill_value=0)
        .reset_index()
    )


def calc_rate(stats):
    total = stats[["W", "D", "L"]].sum(axis=1)
    return stats.assign(rate=stats["L"] / total)


def calc_score(filtered, overall, avg):
    if len(filtered) == 0:
        return pd.DataFrame(columns=["opponent", "score"])

    stats = calc_stats(filtered)
    total = stats[["W", "D", "L"]].sum(axis=1)
    weight = total / (total + ALPHA)

    opponent_rate = (stats["L"] + ALPHA * avg) / (total + ALPHA)
    overall_rate = (
        stats["opponent"]
        .map(
            overall.set_index("opponent")["rate"] if not overall.empty else pd.Series()
        )
        .fillna(avg)
    )

    return stats.assign(score=weight * opponent_rate + (1 - weight) * overall_rate)[
        ["opponent", "score"]
    ]


def get_bins(scores):
    if len(scores) == 0:
        return np.array([0, 0.25, 0.5, 0.75, 1])
    if len(scores.unique()) == 1:
        return np.array([0, scores.iloc[0], 1])

    bins = np.unique(np.percentile(scores, [0, 25, 50, 75, 100]))
    return bins if len(bins) >= 2 else np.array([scores.min(), scores.max()])


def assign_level(score, bins):
    if pd.isna(score):
        return "No Data"
    if len(bins) == 2:
        return LEVELS[0]
    if len(bins) == 3:
        return LEVELS[0] if score <= bins[1] else LEVELS[-1]

    labels = LEVELS[: len(bins) - 1]
    return pd.cut([score], bins, labels=labels, include_lowest=True, duplicates="drop")[
        0
    ]


def add_match_context(matches, team):
    return matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ].assign(
        opponent=lambda df: np.where(
            df["home_team"] == team, df["away_team"], df["home_team"]
        ),
        result=lambda df: np.select(
            [
                ((df["home_team"] == team) & (df["home_score"] > df["away_score"]))
                | ((df["away_team"] == team) & (df["away_score"] > df["home_score"]))
            ],
            ["W"],
            default=np.select(
                [
                    ((df["home_team"] == team) & (df["home_score"] < df["away_score"]))
                    | (
                        (df["away_team"] == team)
                        & (df["away_score"] < df["home_score"])
                    )
                ],
                ["L"],
                default="D",
            ),
        ),
    )


def filter_matches(matches, date_range, tournament, opponent):
    filtered = matches[
        matches["date"].between(
            pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        )
    ]
    if tournament != "All":
        filtered = filtered[filtered["tournament"] == tournament]
    if opponent != "All":
        filtered = filtered[filtered["opponent"] == opponent]
    return filtered


def get_scatter_isos(complex_iso, complex_countries, map_data):
    uk_isos = set(
        countries_df[
            countries_df["current_name"].isin(
                ["England", "Scotland", "Wales", "Northern Ireland"]
            )
        ]["iso_alpha"]
        .dropna()
        .unique()
    )

    scatter = uk_isos.copy()
    for iso in complex_iso:
        if iso not in scatter:
            sub_names = complex_countries.get(iso, [])
            if any(name in map_data["opponent"].values for name in sub_names):
                scatter.add(iso)
    return scatter


def build_iso_data(map_data, complex_iso, scatter_isos, bins):
    simple_isos = set(countries_df["iso_alpha"].dropna().unique()) - set(complex_iso)
    rows = []

    for iso in simple_isos:
        grp = map_data[map_data["iso_alpha"] == iso]
        level = "No Data"
        if not grp.empty and grp["score"].notna().any():
            level = assign_level(grp.loc[grp["score"].notna(), "score"].mean(), bins)
        hover = countries_df[countries_df["iso_alpha"] == iso]["current_name"].iloc[0]
        rows.append({"iso_alpha": iso, "level": level, "hover_name": hover})

    for iso in complex_iso:
        if iso in scatter_isos:
            rows.append({"iso_alpha": iso, "level": "Scatter Data", "hover_name": None})
        else:
            grp = map_data[map_data["iso_alpha"] == iso]
            level = "No Data"
            if not grp.empty and grp["score"].notna().any():
                level = assign_level(
                    grp.loc[grp["score"].notna(), "score"].mean(), bins
                )
            hover = countries_df[countries_df["iso_alpha"] == iso]["current_name"].iloc[
                0
            ]
            rows.append({"iso_alpha": iso, "level": level, "hover_name": hover})

    return pd.DataFrame(rows)


def build_scatter_data(scatter_isos, complex_countries, map_data, coords, bins):
    rows = []
    for iso in scatter_isos:
        for name in complex_countries.get(iso, []):
            lat, lon = coords.get(name, [None, None])
            if lat is not None and lon is not None:
                score_row = map_data[map_data["opponent"] == name]
                level = "No Data"
                if not score_row.empty and pd.notna(score_row["score"].iloc[0]):
                    level = assign_level(score_row["score"].iloc[0], bins)
                rows.append({"opponent": name, "lat": lat, "lon": lon, "level": level})
    return pd.DataFrame(rows)


def create_map(iso_data, scatter_data):
    regular = iso_data[iso_data["level"] != "Scatter Data"].copy()
    scatter_bg = iso_data[iso_data["level"] == "Scatter Data"].copy()

    fig = px.choropleth(
        regular,
        locations="iso_alpha",
        color="level",
        color_discrete_map=COLORS,
        category_orders={"difficulty_level": ALL_LEVELS},
        hover_name="hover_name",
    )

    for t in fig.data:
        t.hovertemplate = "<b>%{hovertext}</b><extra></extra>"

    if len(scatter_bg) > 0:
        bg_fig = px.choropleth(
            scatter_bg,
            locations="iso_alpha",
            color="level",
            color_discrete_map=COLORS,
            category_orders={"difficulty_level": ALL_LEVELS},
        )
        for t in bg_fig.data:
            t.hoverinfo = "skip"
            t.hovertemplate = None
            t.showlegend = False
            fig.add_trace(t)

    if len(scatter_data) > 0:
        scatter_fig = px.scatter_geo(
            scatter_data,
            lat="lat",
            lon="lon",
            hover_name="opponent",
            color="level",
            color_discrete_map=COLORS,
            category_orders={"difficulty_level": ALL_LEVELS},
        )
        for t in scatter_fig.data:
            t.hovertemplate = "<b>%{hovertext}</b><extra></extra>"
            t.showlegend = False
            fig.add_trace(t)

    fig.update_traces(marker_line_color=LINE_COLOR, marker_line_width=1).update_geos(
        showocean=True,
        oceancolor=OCEAN_COLOR,
        showlakes=True,
        lakecolor=OCEAN_COLOR,
        showcountries=True,
        countrycolor=LINE_COLOR,
    ).update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=700,
        legend=dict(
            title="Difficulty Level",
            y=0.5,
            yanchor="middle",
            x=1.02,
            xanchor="left",
            traceorder="normal",
        ),
    )

    fig.for_each_trace(
        lambda t: (
            t.update(legendrank=LEVELS.index(t.name))
            if hasattr(t, "name") and t.name in LEVELS
            else None
        )
    )

    for trace in fig.data:
        if hasattr(trace, "name") and trace.name in ["No Data", "Scatter Data"]:
            trace.showlegend = False

    return fig


with st.sidebar:
    team = st.selectbox(
        "Select a team:",
        countries,
        index=countries.index(DEFAULT_TEAM) if DEFAULT_TEAM in countries else 0,
    )

    team_matches = add_match_context(matches, team)

    date_range = st.date_input(
        "Select date range:",
        value=(team_matches["date"].min().date(), team_matches["date"].max().date()),
        min_value=team_matches["date"].min().date(),
        max_value=team_matches["date"].max().date(),
    )

    st.session_state.setdefault("opponent", "All")
    st.session_state.setdefault("tournament", "All")

    date_filtered = team_matches[
        team_matches["date"].between(
            pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        )
    ]

    current_opponent = st.session_state["opponent"]
    current_tournament = st.session_state["tournament"]

    tournament_opts = date_filtered.copy()
    if current_opponent != "All":
        tournament_opts = tournament_opts[
            tournament_opts["opponent"] == current_opponent
        ]

    tournament_list = ["All"] + sorted(tournament_opts["tournament"].unique())
    tournament_idx = (
        tournament_list.index(current_tournament)
        if current_tournament in tournament_list
        else 0
    )
    tournament = st.selectbox(
        "Select a tournament:", tournament_list, index=tournament_idx, key="tournament"
    )

    opponent_opts = date_filtered.copy()
    if tournament != "All":
        opponent_opts = opponent_opts[opponent_opts["tournament"] == tournament]

    opponent_list = ["All"] + sorted(opponent_opts["opponent"].unique())
    opponent_idx = (
        opponent_list.index(current_opponent)
        if current_opponent in opponent_list
        else 0
    )
    opponent = st.selectbox(
        "Select an opponent:", opponent_list, index=opponent_idx, key="opponent"
    )

filtered = filter_matches(date_filtered, date_range, tournament, opponent)
overall = calc_rate(calc_stats(date_filtered))
avg = overall["rate"].mean() if not overall.empty else 0

scores = calc_score(filtered, overall, avg)
all_iso = pd.DataFrame(list(iso_map.items()), columns=["opponent", "iso_alpha"])
map_data = (
    pd.DataFrame({"opponent": countries})
    .merge(scores, on="opponent", how="left")
    .merge(all_iso, on="opponent", how="left")
)

bins = get_bins(map_data["score"].dropna())
scatter_isos = get_scatter_isos(complex_iso, complex_countries, map_data)
iso_data = build_iso_data(map_data, complex_iso, scatter_isos, bins)
scatter_data = build_scatter_data(
    scatter_isos, complex_countries, map_data, coords, bins
)

st.dataframe(
    filtered.assign(
        venue=lambda df: np.where(
            df["home_team"] == team,
            "Home",
            np.where(df["away_team"] == team, "Away", "Neutral"),
        ),
        date=lambda df: df["date"].dt.date,
    )[
        [
            "date",
            "tournament",
            "opponent",
            "venue",
            "home_score",
            "away_score",
            "result",
        ]
    ].reset_index(
        drop=True
    )
)

st.plotly_chart(create_map(iso_data, scatter_data), use_container_width=True)
