"""Exploratory data analysis and visualizations for the MovieLens dataset."""

#Imports
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from data_preprocessing import PROCESSED_DIR, preprocess_dataset

#Constant
FIGURES_DIR = PROCESSED_DIR / "figures" #PNG files from py data_visualization.py go here
TOP_GENRE_COUNT = 12
TOP_TAG_COUNT = 20


#Load cleaned CSVs; run preprocessing first if they are missing
def load_processed_data(processed_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Load cleaned CSV files, running preprocessing if they are missing."""
    required_files = {
        "movies": processed_dir / "movies_clean.csv",
        "ratings": processed_dir / "ratings_clean.csv",
        "tags": processed_dir / "tags_clean.csv",
    }

    if not all(path.exists() for path in required_files.values()):
        print("Processed files not found. Running preprocessing...")
        return preprocess_dataset(output_dir=processed_dir, save_outputs=True)

    data = {name: pd.read_csv(path) for name, path in required_files.items()}
    data["ratings"]["rated_at"] = pd.to_datetime(data["ratings"]["rated_at"], utc=True)
    data["tags"]["tagged_at"] = pd.to_datetime(data["tags"]["tagged_at"], utc=True)
    return data


def _configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 120


#Save a figure as PNG, then close it so later plots do not pile up
def _save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


#Turn "Action|Comedy" into one row per genre so we can count or average by genre
def _expand_genres(movies: pd.DataFrame) -> pd.DataFrame:
    """Return one row per movie-genre pair."""
    if "genre_list" in movies.columns:
        genre_lists = movies["genre_list"].apply(
            lambda value: ast.literal_eval(value)
            if isinstance(value, str) and value.startswith("[")
            else str(value).split("|")
        )
    else:
        genre_lists = movies["genres"].astype(str).str.split("|")

    expanded = movies[["movieId"]].copy()
    expanded["genre"] = genre_lists
    expanded = expanded.explode("genre")
    expanded["genre"] = expanded["genre"].str.strip()
    return expanded.loc[expanded["genre"].ne("(no genres listed)")]


#White background so Streamlit charts match the app theme
def _white_figure(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def _label_bars(ax, fmt: str, horizontal: bool = False) -> None:
    for patch in ax.patches:
        value = patch.get_width() if horizontal else patch.get_height()
        if value is None or not np.isfinite(value):
            continue
        label = fmt.format(value)
        if horizontal:
            ax.annotate(
                label,
                (value, patch.get_y() + patch.get_height() / 2),
                ha="left",
                va="center",
                fontsize=7,
                color="#31333F",
                xytext=(4, 0),
                textcoords="offset points",
            )
        else:
            ax.annotate(
                label,
                (patch.get_x() + patch.get_width() / 2, value),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#31333F",
            )


def rating_distribution_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings["rating"]
        .value_counts()
        .sort_index()
        .rename_axis("rating")
        .reset_index(name="count")
    )


#1.1 Bar chart of how many 0.5, 1.0, ... 5.0 ratings there are
def make_rating_distribution(ratings: pd.DataFrame) -> plt.Figure:
    """1.1 How users rate movies and which scores are most common."""
    stats = rating_distribution_stats(ratings)
    fig, ax = _white_figure((6.2, 3.0))
    ax.bar(stats["rating"].astype(str), stats["count"], width=0.4, color="#E85D75")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of ratings")
    ax.set_title("Distribution of movie ratings")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    _label_bars(ax, "{:.0f}")
    ax.margins(y=0.12)
    fig.tight_layout()
    return fig


#1.2 How many movies each user has rated, grouped into simple activity bands
def make_user_activity(ratings: pd.DataFrame) -> plt.Figure:
    """1.2 Count users in easy-to-read activity groups."""
    counts = ratings.groupby("userId").size()
    bins = [0, 50, 100, 200, 500, 1000, np.inf]
    labels = ["1–50", "51–100", "101–200", "201–500", "501–1,000", "1,000+"]
    bands = pd.cut(counts, bins=bins, labels=labels, right=True, include_lowest=True)
    band_counts = bands.value_counts().reindex(labels).fillna(0).astype(int)

    fig, ax = _white_figure((6.6, 3.2))
    ax.bar(band_counts.index.astype(str), band_counts.values, color="#5B8FF9", width=0.7)
    ax.set_title("How many movies each user has rated")
    ax.set_xlabel("Ratings given by one user")
    ax.set_ylabel("Number of users")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    _label_bars(ax, "{:.0f}")
    ax.margins(y=0.12)
    fig.tight_layout()
    return fig


#Median / mean / min / max ratings per user (shown under the chart in the app)
def user_activity_stats(ratings: pd.DataFrame) -> pd.Series:
    return ratings.groupby("userId").size().rename("rating_count").describe().round(2)


#1.3 How many movies belong to each genre
def genre_distribution_stats(movies: pd.DataFrame) -> pd.DataFrame:
    genre_count = (
        movies["genres"]
        .fillna("")
        .astype(str)
        .str.get_dummies()
        .sum()
        .drop(labels=["(no genres listed)"], errors="ignore")
        .sort_values(ascending=False)
    )
    return genre_count.rename_axis("genre").reset_index(name="movie_count")


def make_genre_distribution(movies: pd.DataFrame) -> plt.Figure:
    """1.3 Genres with the largest number of movies."""
    stats = genre_distribution_stats(movies)
    fig, ax = _white_figure((6.6, 3.4))
    ax.bar(stats["genre"].astype(str), stats["movie_count"], color="#5AD8A6")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of movies")
    ax.set_title("Movie genre distribution")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    _label_bars(ax, "{:.0f}")
    ax.margins(y=0.14)
    fig.tight_layout()
    return fig


#1.4 Catalog size by release year
def movies_by_year_stats(movies: pd.DataFrame) -> pd.DataFrame:
    movie_year = movies.dropna(subset=["year"]).copy()
    movie_year["year"] = movie_year["year"].astype(int)
    return (
        movie_year["year"]
        .value_counts()
        .sort_index()
        .rename_axis("year")
        .reset_index(name="movie_count")
    )


def make_movies_by_year(movies: pd.DataFrame) -> plt.Figure:
    """1.4 How the catalog changes across release years."""
    stats = movies_by_year_stats(movies)
    fig, ax = _white_figure((6.6, 3.0))
    ax.plot(
        stats["year"],
        stats["movie_count"],
        marker="o",
        markersize=3,
        markeredgewidth=0,
        color="#E85D75",
        linewidth=1.5,
    )
    ax.set_xlabel("Release year")
    ax.set_ylabel("Number of movies")
    ax.set_title("Number of movies released by year")
    ax.grid(True, color="#D0D3DA", linewidth=0.8)
    fig.tight_layout()
    return fig


#1.5 Horizontal bar chart of the most common user tags
def make_top_tags(tags: pd.DataFrame) -> plt.Figure:
    """1.5 Most frequently applied user tags."""
    tag_column = "tag_standardization" if "tag_standardization" in tags.columns else "tag"
    tag_count = tags[tag_column].value_counts().head(TOP_TAG_COUNT)
    fig, ax = _white_figure((6.2, 4.0))
    ax.barh(tag_count.index.astype(str)[::-1], tag_count.values[::-1], color="#2b1153")
    ax.set_xlabel("Number of tags")
    ax.set_ylabel("Tag")
    ax.set_title(f"Top {TOP_TAG_COUNT} most common movie tags")
    _label_bars(ax, "{:.0f}", horizontal=True)
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig


#Every unique tag and how often it appears
def all_tag_counts(tags: pd.DataFrame) -> pd.DataFrame:
    tag_column = "tag_standardization" if "tag_standardization" in tags.columns else "tag"
    counts = tags[tag_column].value_counts().rename_axis("tag").reset_index(name="count")
    return counts


#Average rating and rating count for each genre
def genre_rating_stats(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    movie_genres = movies[["movieId", "genres"]].copy()
    movie_genres = movie_genres.assign(genre=movie_genres["genres"].str.split("|")).explode(
        "genre"
    )
    movie_genres["genre"] = movie_genres["genre"].astype(str).str.strip()
    genre_ratings = movie_genres.merge(ratings[["movieId", "rating"]], on="movieId", how="inner")
    stats = (
        genre_ratings.loc[genre_ratings["genre"].ne("(no genres listed)")]
        .groupby("genre")
        .agg(average_rating=("rating", "mean"), total_ratings=("rating", "count"))
        .reset_index()
        .sort_values("average_rating", ascending=False)
    )
    stats["average_rating"] = stats["average_rating"].round(2)
    return stats


#1.6 Which genres tend to get higher scores
def make_average_rating_by_genre(movies: pd.DataFrame, ratings: pd.DataFrame) -> plt.Figure:
    """1.6 Average user rating across genres."""
    stats = genre_rating_stats(movies, ratings)
    fig, ax = _white_figure((7.2, 5.6))
    sns.barplot(data=stats, x="average_rating", y="genre", color="#5B8FF9", ax=ax)
    ax.set_title("Average Movie Rating by Genre")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Genre")
    ax.set_xlim(0, 5.0)
    ax.set_xticks(np.arange(0, 5.1, 0.5))
    ax.xaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _label_bars(ax, "{:.2f}", horizontal=True)
    fig.tight_layout()
    return fig


#1.7 How many ratings were submitted in each year
def ratings_over_time_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    work = ratings.copy()
    work["rated_at"] = pd.to_datetime(work["rated_at"], utc=True)
    work["year"] = work["rated_at"].dt.year
    return (
        work.groupby("year")
        .size()
        .rename("rating_count")
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )


def make_ratings_over_time(ratings: pd.DataFrame) -> plt.Figure:
    """1.7 Rating volume by the year ratings were submitted."""
    stats = ratings_over_time_stats(ratings)
    fig, ax = _white_figure((6.6, 3.0))
    sns.lineplot(
        data=stats,
        x="year",
        y="rating_count",
        marker="o",
        markersize=4,
        color="#E85D75",
        ax=ax,
    )
    ax.set_title("User rating activity over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of ratings")
    ax.grid(True, color="#D0D3DA", linewidth=0.8)
    fig.tight_layout()
    return fig


#Colours for the sparsity heatmap and cold-start bars
_SPARSITY_FILLED = "#0984e3"
_SPARSITY_EMPTY = "#dfe6e9"
_MOVIE_HIST_COLOR = "#2c3e50"
_COLDSTART_BELOW = "#E85D75"
_COLDSTART_ABOVE = "#5AD8A6"


#How full the user–movie matrix is (low % = sparse, which is why CF is hard)
def sparsity_stats(ratings: pd.DataFrame) -> tuple[float, int, int]:
    n_users = int(ratings["userId"].nunique())
    n_movies = int(ratings["movieId"].nunique())
    filled_pct = 100.0 * len(ratings) / (n_users * n_movies) if n_users and n_movies else 0.0
    return filled_pct, n_users, n_movies


#1.8 Sample of the rating matrix: blue = rated, grey = missing
def make_matrix_sparsity_heatmap(
    ratings: pd.DataFrame,
    n_users: int = 80,
    n_movies: int = 80,
    seed: int = 42,
) -> plt.Figure:
    """1.8 Rated vs unrated cells in a random sample of the user-item matrix."""
    rng = np.random.default_rng(seed)
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()
    sampled_users = rng.choice(user_ids, size=min(n_users, len(user_ids)), replace=False)
    sampled_movies = rng.choice(movie_ids, size=min(n_movies, len(movie_ids)), replace=False)

    subset = ratings.loc[
        ratings["userId"].isin(sampled_users) & ratings["movieId"].isin(sampled_movies)
    ]
    matrix = subset.pivot_table(index="userId", columns="movieId", values="rating")
    matrix = matrix.reindex(index=sampled_users, columns=sampled_movies)
    binary = matrix.notna().astype(int) #1 = this user rated this movie, 0 = empty

    fig, ax = _white_figure((6.4, 5.6))
    cmap = ListedColormap([_SPARSITY_EMPTY, _SPARSITY_FILLED])
    ax.imshow(binary.values, cmap=cmap, aspect="auto", interpolation="none")
    ax.set_title("User-item matrix sparsity (sampled)")
    ax.set_xlabel(f"{len(sampled_movies)} sampled movies")
    ax.set_ylabel(f"{len(sampled_users)} sampled users")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=[
            Patch(facecolor=_SPARSITY_FILLED, label="Rated"),
            Patch(facecolor=_SPARSITY_EMPTY, edgecolor="#c7ccd4", label="Unrated"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return fig


#1.9 Long-tail: most movies have few ratings (log scale so the tail is visible)
def make_ratings_per_movie_distribution(ratings: pd.DataFrame) -> plt.Figure:
    """1.9 How many ratings each movie receives, on a log-scaled y-axis."""
    movie_counts = ratings.groupby("movieId").size()
    fig, ax = _white_figure((6.4, 3.2))
    sns.histplot(movie_counts, bins=40, color=_MOVIE_HIST_COLOR, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Distribution of ratings per movie")
    ax.set_xlabel("Number of ratings received")
    ax.set_ylabel("Number of movies (log scale)")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


#1.10 Share of users/movies with too few ratings (cold-start)
def make_coldstart_segment_bars(
    ratings: pd.DataFrame,
    threshold: int = 5,
) -> tuple[plt.Figure, float, float]:
    """1.10 Share of users and movies with too few ratings for reliable recommendations."""
    user_counts = ratings.groupby("userId").size()
    movie_counts = ratings.groupby("movieId").size()
    pct_users_below = float((user_counts <= threshold).mean() * 100)
    pct_movies_below = float((movie_counts <= threshold).mean() * 100)
    below_pct = [pct_users_below, pct_movies_below]
    above_pct = [100 - pct for pct in below_pct]

    fig, ax = _white_figure((6.0, 3.2))
    entities = ["Users", "Movies"]
    x = np.arange(len(entities))
    width = 0.35
    ax.bar(
        x - width / 2,
        below_pct,
        width,
        label=f"≤ {threshold} ratings (cold-start)",
        color=_COLDSTART_BELOW,
    )
    ax.bar(x + width / 2, above_pct, width, label=f"> {threshold} ratings", color=_COLDSTART_ABOVE)
    ax.set_xticks(x)
    ax.set_xticklabels(entities)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Share (%)")
    ax.set_title(f"Cold-start segment size (threshold = {threshold} ratings)")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig, pct_users_below, pct_movies_below


#Same cutoffs as hybrid "Why this mix?"
HYBRID_RARE_MAX = 15
HYBRID_POPULAR_MIN = 120


#1.11 How many movies fall into hybrid rare / mid / popular bands
def make_hybrid_support_histogram(ratings: pd.DataFrame) -> tuple[plt.Figure, float, float, float]:
    """Count movies in the same rare / mid / popular groups used by hybrid."""
    movie_counts = ratings.groupby("movieId").size()
    n_movies = int(len(movie_counts))
    n_rare = int((movie_counts <= HYBRID_RARE_MAX).sum())
    n_popular = int((movie_counts >= HYBRID_POPULAR_MIN).sum())
    n_mid = max(0, n_movies - n_rare - n_popular)
    pct_rare = 100.0 * n_rare / n_movies if n_movies else 0.0
    pct_mid = 100.0 * n_mid / n_movies if n_movies else 0.0
    pct_popular = 100.0 * n_popular / n_movies if n_movies else 0.0

    labels = [
        f"Rarely rated\n(≤{HYBRID_RARE_MAX} ratings)\nmore content",
        f"In between\n({HYBRID_RARE_MAX + 1}–{HYBRID_POPULAR_MIN - 1})\nmixed",
        f"Popular\n(≥{HYBRID_POPULAR_MIN} ratings)\nmore SVD",
    ]
    values = [n_rare, n_mid, n_popular]
    colors = ["#E85D75", "#5B8FF9", "#5AD8A6"]

    fig, ax = _white_figure((6.6, 3.4))
    ax.bar(labels, values, color=colors, width=0.65)
    ax.set_title("How hybrid treats movies by popularity")
    ax.set_ylabel("Number of movies")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    _label_bars(ax, "{:.0f}")
    ax.margins(y=0.14)
    fig.tight_layout()
    return fig, pct_rare, pct_mid, pct_popular


def _hybrid_popularity_band(count: int) -> str:
    if count <= HYBRID_RARE_MAX:
        return "Rarely rated"
    if count >= HYBRID_POPULAR_MIN:
        return "Popular"
    return "In between"


#EDA scatter: popularity vs average rating
def make_popularity_rating_scatter(ratings: pd.DataFrame) -> plt.Figure:
    """Each point is a movie: how often it was rated vs its average score."""
    stats = ratings.groupby("movieId").agg(
        n_ratings=("rating", "size"),
        avg_rating=("rating", "mean"),
    )
    stats["group"] = stats["n_ratings"].map(_hybrid_popularity_band)
    order = ["Rarely rated", "In between", "Popular"]
    colors = {"Rarely rated": "#E85D75", "In between": "#5B8FF9", "Popular": "#5AD8A6"}

    fig, ax = _white_figure((6.8, 3.6))
    sns.scatterplot(
        data=stats.reset_index(),
        x="n_ratings",
        y="avg_rating",
        hue="group",
        hue_order=order,
        palette=colors,
        alpha=0.45,
        s=22,
        ax=ax,
        linewidth=0,
    )
    ax.set_xscale("log")
    ax.set_ylim(0.4, 5.2)
    ax.set_title("Movie popularity vs average rating")
    ax.set_xlabel("Number of ratings (log scale)")
    ax.set_ylabel("Average rating")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(title="", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


#EDA heatmap: average rating by genre and decade
def make_genre_decade_heatmap(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    top_n_genres: int = 10,
) -> plt.Figure:
    """Average rating for the most-rated genres across release decades."""
    work = movies[["movieId", "genres"]].copy()
    if "year" in movies.columns:
        work["year"] = movies["year"]
    else:
        year_match = movies["title"].str.extract(r"\((\d{4})\)\s*$")
        work["year"] = pd.to_numeric(year_match[0], errors="coerce")
    work["decade"] = (pd.to_numeric(work["year"], errors="coerce") // 10 * 10).astype("Int64")
    exploded = work.assign(genre=work["genres"].astype(str).str.split("|")).explode("genre")
    exploded["genre"] = exploded["genre"].astype(str).str.strip()
    exploded = exploded.loc[exploded["genre"].ne("(no genres listed)")]
    merged = ratings.merge(exploded[["movieId", "genre", "decade"]], on="movieId", how="inner")
    merged = merged.dropna(subset=["decade"])
    top_genres = merged.groupby("genre").size().nlargest(top_n_genres).index
    merged = merged.loc[merged["genre"].isin(top_genres)]
    pivot = merged.pivot_table(
        index="genre",
        columns="decade",
        values="rating",
        aggfunc="mean",
    )
    pivot = pivot.reindex(top_genres)
    pivot.columns = [str(int(col)) + "s" for col in pivot.columns]

    fig, ax = _white_figure((7.4, 4.2))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Average rating"},
        vmin=3.0,
        vmax=4.2,
    )
    ax.set_title(f"Average rating by genre and decade (top {top_n_genres} genres)")
    ax.set_xlabel("Release decade")
    ax.set_ylabel("Genre")
    fig.tight_layout()
    return fig


#The plot_* functions below save PNGs when you run this file from the terminal
def plot_rating_distribution(ratings: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the overall distribution of star ratings."""
    order = sorted(ratings["rating"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=ratings, x="rating", order=order, color="#4C72B0", ax=ax)
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    return _save_figure(fig, output_dir, "rating_distribution.png")


def plot_user_activity(ratings: pd.DataFrame, output_dir: Path) -> Path:
    """Plot how many ratings each user contributed."""
    user_counts = ratings.groupby("userId").size().rename("rating_count")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(user_counts, bins=30, kde=True, color="#55A868", ax=ax)
    ax.set_title("User Activity")
    ax.set_xlabel("Ratings per user")
    ax.set_ylabel("Number of users")
    return _save_figure(fig, output_dir, "user_activity.png")


def plot_movie_popularity(ratings: pd.DataFrame, output_dir: Path) -> Path:
    """Plot how many ratings each movie received."""
    #Log y-axis: a few popular movies would otherwise hide the long tail
    movie_counts = ratings.groupby("movieId").size().rename("rating_count")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(movie_counts, bins=40, color="#C44E52", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Movie Popularity")
    ax.set_xlabel("Ratings per movie")
    ax.set_ylabel("Number of movies (log scale)")
    return _save_figure(fig, output_dir, "movie_popularity.png")


def plot_genre_frequency(movies: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the most common movie genres."""
    genre_rows = _expand_genres(movies)
    genre_counts = genre_rows["genre"].value_counts().head(TOP_GENRE_COUNT)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        x=genre_counts.values,
        y=genre_counts.index,
        hue=genre_counts.index,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title(f"Top {TOP_GENRE_COUNT} Genres")
    ax.set_xlabel("Number of movies")
    ax.set_ylabel("Genre")
    return _save_figure(fig, output_dir, "genre_frequency.png")


def plot_movies_by_year(movies: pd.DataFrame, output_dir: Path) -> Path:
    """Plot movie counts by release year."""
    yearly_counts = (
        movies.dropna(subset=["year"])
        .groupby("year")
        .size()
        .rename("movie_count")
        .reset_index()
        .sort_values("year")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=yearly_counts, x="year", y="movie_count", marker="o", markersize=3, ax=ax)
    ax.set_title("Movies by Release Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of movies")
    return _save_figure(fig, output_dir, "movies_by_year.png")


def plot_ratings_over_time(ratings: pd.DataFrame, output_dir: Path) -> Path:
    """Plot rating volume over time."""
    monthly_counts = (
        ratings.set_index("rated_at")
        .resample("ME")["rating"]
        .count()
        .rename("rating_count")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=monthly_counts, x="rated_at", y="rating_count", ax=ax)
    ax.set_title("Ratings Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of ratings")
    return _save_figure(fig, output_dir, "ratings_over_time.png")


def plot_average_rating_by_genre(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot average rating for the most common genres."""
    genre_rows = _expand_genres(movies)
    merged = ratings.merge(genre_rows, on="movieId", how="inner")
    genre_means = (
        merged.groupby("genre")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(TOP_GENRE_COUNT)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        x=genre_means.values,
        y=genre_means.index,
        hue=genre_means.index,
        palette="mako",
        legend=False,
        ax=ax,
    )
    ax.set_xlim(0.0, 5.0)
    ax.set_xticks(np.arange(0, 5.1, 0.5))
    ax.set_title("Average Movie Rating by Genre")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Genre")
    ax.xaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save_figure(fig, output_dir, "average_rating_by_genre.png")


def plot_top_tags(tags: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the most frequently applied user tags."""
    tag_column = "tag_standardization" if "tag_standardization" in tags.columns else "tag"
    tag_counts = tags[tag_column].value_counts().head(TOP_TAG_COUNT).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(
        x=tag_counts.values,
        y=tag_counts.index,
        hue=tag_counts.index,
        palette="rocket",
        legend=False,
        ax=ax,
    )
    ax.set_title(f"Top {TOP_TAG_COUNT} Tags")
    ax.set_xlabel("Count")
    ax.set_ylabel("Tag")
    return _save_figure(fig, output_dir, "top_tags.png")


def plot_rating_sparsity_overview(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot a compact dashboard of key dataset characteristics."""
    #Four small charts on one page: scores, user activity, movie popularity, years
    user_counts = ratings.groupby("userId").size()
    movie_counts = ratings.groupby("movieId").size()
    rating_counts = ratings["rating"].value_counts().sort_index()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sns.barplot(
        x=rating_counts.index,
        y=rating_counts.values,
        color="#4C72B0",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Rating Counts")
    axes[0, 0].set_xlabel("Rating")
    axes[0, 0].set_ylabel("Count")

    sns.histplot(user_counts, bins=25, color="#55A868", ax=axes[0, 1])
    axes[0, 1].set_title("Ratings per User")
    axes[0, 1].set_xlabel("Ratings")
    axes[0, 1].set_ylabel("Users")

    sns.histplot(movie_counts, bins=25, color="#C44E52", ax=axes[1, 0])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Ratings per Movie")
    axes[1, 0].set_xlabel("Ratings")
    axes[1, 0].set_ylabel("Movies (log scale)")

    yearly_counts = movies.dropna(subset=["year"]).groupby("year").size()
    axes[1, 1].plot(yearly_counts.index, yearly_counts.values, color="#8172B3")
    axes[1, 1].set_title("Movies by Year")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Movies")

    fig.suptitle("MovieLens EDA Overview", fontsize=14, y=1.02)
    return _save_figure(fig, output_dir, "eda_overview.png")


#Print counts used in the report (users, movies, density, etc.)
def print_eda_summary(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame) -> None:
    """Print key descriptive statistics for the dataset."""
    user_counts = ratings.groupby("userId").size()
    movie_counts = ratings.groupby("movieId").size()
    matrix_density = len(ratings) / (ratings["userId"].nunique() * ratings["movieId"].nunique())

    print("EDA summary")
    print("-" * 32)
    print(f"Users:                {ratings['userId'].nunique():,}")
    print(f"Movies:               {ratings['movieId'].nunique():,}")
    print(f"Ratings:              {len(ratings):,}")
    print(f"Tags:                 {len(tags):,}")
    print(f"Average rating:       {ratings['rating'].mean():.2f}")
    print(f"Median user activity: {user_counts.median():.0f} ratings")
    print(f"Median movie popularity: {movie_counts.median():.0f} ratings")
    print(f"Matrix density:       {matrix_density:.4%}")
    print(f"Most common rating:   {ratings['rating'].mode().iloc[0]:.1f}")


#CLI entry: load cleaned data, print a summary, save every figure as a PNG
def run_eda(
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Generate and save all EDA visualizations."""
    _configure_style()
    data = load_processed_data(processed_dir)
    movies = data["movies"]
    ratings = data["ratings"]
    tags = data["tags"]

    print_eda_summary(movies, ratings, tags)

    #Each plotter saves one PNG under processed/figures/
    plotters = [
        plot_rating_distribution(ratings, output_dir),
        plot_user_activity(ratings, output_dir),
        plot_movie_popularity(ratings, output_dir),
        plot_genre_frequency(movies, output_dir),
        plot_movies_by_year(movies, output_dir),
        plot_ratings_over_time(ratings, output_dir),
        plot_average_rating_by_genre(movies, ratings, output_dir),
        plot_top_tags(tags, output_dir),
        plot_rating_sparsity_overview(ratings, movies, output_dir),
    ]

    print("\nSaved figures:")
    for path in plotters:
        print(f"  - {path}")

    return plotters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MovieLens EDA visualizations.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing cleaned CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory where figure files will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(processed_dir=args.processed_dir, output_dir=args.output_dir)
