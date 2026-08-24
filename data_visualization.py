"""Exploratory data analysis and visualizations for the MovieLens dataset."""

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

FIGURES_DIR = PROCESSED_DIR / "figures"
TOP_GENRE_COUNT = 12
TOP_TAG_COUNT = 20


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


def _save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


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


def _white_figure(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def make_rating_distribution(ratings: pd.DataFrame) -> plt.Figure:
    """1.1 How users rate movies and which scores are most common."""
    rating_count = ratings["rating"].value_counts().sort_index()
    fig, ax = _white_figure((6.2, 3.0))
    ax.bar(rating_count.index.astype(str), rating_count.values, width=0.4, color="#E85D75")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of ratings")
    ax.set_title("Distribution of movie ratings")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def make_user_activity(ratings: pd.DataFrame) -> plt.Figure:
    """1.2 How many ratings each user submitted."""
    user_activity = ratings.groupby("userId").size().reset_index(name="rating_count")
    fig, ax = _white_figure((6.2, 3.0))
    sns.histplot(data=user_activity, x="rating_count", bins=40, color="#5B8FF9", ax=ax)
    ax.set_title("Distribution of ratings per user")
    ax.set_xlabel("Number of ratings")
    ax.set_ylabel("Number of users")
    fig.tight_layout()
    return fig


def user_activity_stats(ratings: pd.DataFrame) -> pd.Series:
    return ratings.groupby("userId").size().rename("rating_count").describe().round(2)


def make_genre_distribution(movies: pd.DataFrame) -> plt.Figure:
    """1.3 Genres with the largest number of movies."""
    genre_count = (
        movies["genres"]
        .fillna("")
        .astype(str)
        .str.get_dummies()
        .sum()
        .drop(labels=["(no genres listed)"], errors="ignore")
        .sort_values(ascending=False)
    )
    fig, ax = _white_figure((6.6, 3.2))
    ax.bar(genre_count.index.astype(str), genre_count.values, color="#5AD8A6")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of movies")
    ax.set_title("Movie genre distribution")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def make_movies_by_year(movies: pd.DataFrame) -> plt.Figure:
    """1.4 How the catalog changes across release years."""
    movie_year = movies.dropna(subset=["year"]).copy()
    movie_year["year"] = movie_year["year"].astype(int)
    year_count = movie_year["year"].value_counts().sort_index()
    fig, ax = _white_figure((6.6, 3.0))
    ax.plot(year_count.index, year_count.values, marker="o", color="#E85D75", linewidth=2)
    ax.set_xlabel("Release year")
    ax.set_ylabel("Number of movies")
    ax.set_title("Number of movies released by year")
    ax.grid(True, color="#D0D3DA", linewidth=0.8)
    fig.tight_layout()
    return fig


def make_top_tags(tags: pd.DataFrame) -> plt.Figure:
    """1.5 Most frequently applied user tags."""
    tag_column = "tag_standardization" if "tag_standardization" in tags.columns else "tag"
    tag_count = tags[tag_column].value_counts().head(TOP_TAG_COUNT)
    fig, ax = _white_figure((6.2, 4.0))
    ax.barh(tag_count.index.astype(str)[::-1], tag_count.values[::-1], color="#2b1153")
    ax.set_xlabel("Number of tags")
    ax.set_ylabel("Tag")
    ax.set_title("Top 20 most common movie tags")
    fig.tight_layout()
    return fig


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


def make_average_rating_by_genre(movies: pd.DataFrame, ratings: pd.DataFrame) -> plt.Figure:
    """1.6 Average user rating across genres."""
    stats = genre_rating_stats(movies, ratings)
    fig, ax = _white_figure((6.2, 4.0))
    sns.barplot(data=stats, x="average_rating", y="genre", color="#5B8FF9", ax=ax)
    ax.set_title("Average movie rating across different genres")
    ax.set_xlabel("Average rating")
    ax.set_ylabel("Genre")
    fig.tight_layout()
    return fig


def make_ratings_over_time(ratings: pd.DataFrame) -> plt.Figure:
    """1.7 Rating volume by the year ratings were submitted."""
    work = ratings.copy()
    work["rated_at"] = pd.to_datetime(work["rated_at"], utc=True)
    work["rating_year"] = work["rated_at"].dt.year
    ratings_over_time = work.groupby("rating_year").size().reset_index(name="rating_count")
    fig, ax = _white_figure((6.6, 3.0))
    sns.lineplot(
        data=ratings_over_time,
        x="rating_year",
        y="rating_count",
        marker="o",
        color="#E85D75",
        ax=ax,
    )
    ax.set_title("User rating activity over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of ratings")
    ax.grid(True, color="#D0D3DA", linewidth=0.8)
    fig.tight_layout()
    return fig


_SPARSITY_FILLED = "#0984e3"
_SPARSITY_EMPTY = "#dfe6e9"
_MOVIE_HIST_COLOR = "#2c3e50"
_COLDSTART_BELOW = "#E85D75"
_COLDSTART_ABOVE = "#5AD8A6"


def sparsity_stats(ratings: pd.DataFrame) -> tuple[float, int, int]:
    n_users = int(ratings["userId"].nunique())
    n_movies = int(ratings["movieId"].nunique())
    filled_pct = 100.0 * len(ratings) / (n_users * n_movies) if n_users and n_movies else 0.0
    return filled_pct, n_users, n_movies


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
    binary = matrix.notna().astype(int)

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
    sns.lineplot(data=yearly_counts, x="year", y="movie_count", marker="o", ax=ax)
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
    ax.set_xlim(0.5, 5.0)
    ax.set_title(f"Average Rating by Genre (Top {TOP_GENRE_COUNT})")
    ax.set_xlabel("Average rating")
    ax.set_ylabel("Genre")
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
