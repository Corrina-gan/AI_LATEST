"""Streamlit movie recommendation app (MovieLens assignment UI)."""

from __future__ import annotations

import ast
from html import escape
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import collaborative_filtering
import content_based
import data_visualization
import hybrid

# Streamlit can keep a stale collaborative_filtering module from an earlier
# script run. Reload it when this app expects APIs that are missing.
if not hasattr(collaborative_filtering, "render_controls"):
    collaborative_filtering = reload(collaborative_filtering)
if not hasattr(collaborative_filtering, "DEFAULT_ITEM_METHOD"):
    collaborative_filtering.DEFAULT_ITEM_METHOD = "cosine"
if not hasattr(collaborative_filtering, "DEFAULT_N_COMPONENTS"):
    collaborative_filtering.DEFAULT_N_COMPONENTS = 20
if not hasattr(collaborative_filtering, "ITEM_METHOD_OPTIONS"):
    collaborative_filtering.ITEM_METHOD_OPTIONS = {
        "Cosine": "cosine",
        "Pearson (adjusted cosine)": "pearson",
    }

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

HybridRecommender = hybrid.HybridRecommender
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_FILES = (
    "ratings_clean.csv",
    "movies_clean.csv",
    "movies_content.csv",
)

MAIN_TABS = (
    "🔀 Hybrid",
    "🎬 Content-based",
    "👥 Collaborative",
    "📈 Model Comparison",
    "📉 Data Visualization",
    "🔍 Data Explorer",
)
ALGORITHM_LABELS = {
    "hybrid": "Hybrid (Content + Collaborative)",
    content_based.ALGORITHM_KEY: "Content-based (TF-IDF)",
    collaborative_filtering.ALGORITHM_KEY: "Collaborative (SVD)",
}

DISPLAY_METRIC_KEYS = (
    "rmse",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "decision_threshold",
)

COMPARISON_ALGORITHMS = (
    "Content-based (TF-IDF)",
    "Collaborative (SVD)",
    "Hybrid",
)
CLASSIFICATION_METRICS = ("Precision", "Recall", "F1-score", "Accuracy")
MODEL_COLORS = {
    "Content-based (TF-IDF)": "#8B90A0",
    "Collaborative (SVD)": "#5B8FF9",
    "Hybrid": "#E85D75",
}


def _find_processed_file(filename: str) -> Path:
    for candidate in (
        BASE_DIR / "processed" / filename,
        BASE_DIR / "dataset" / "processed" / filename,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{filename} not found. Run `py data_preprocessing.py --output-dir processed` first."
    )


def processed_data_signature() -> str:
    """Cache key that changes when cleaned CSV files are rebuilt."""
    parts: list[str] = []
    for filename in PROCESSED_FILES:
        try:
            path = _find_processed_file(filename)
        except FileNotFoundError:
            parts.append(f"{filename}:missing")
            continue
        stat = path.stat()
        parts.append(f"{filename}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def _scalar_metrics(result: dict) -> dict[str, float]:
    """Keep RMSE / classification numbers; drop arrays from evaluate()."""
    return {key: float(result[key]) for key in DISPLAY_METRIC_KEYS if key in result}


def _pack_eval_arrays(
    metrics: dict,
    actual: np.ndarray,
    predicted: np.ndarray,
    relevance_threshold: float = 4.0,
) -> dict:
    """Attach predicted/actual arrays so the Evaluation tab can draw plots."""
    packed = {**metrics}
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    decision = float(metrics.get("decision_threshold", relevance_threshold))
    packed["actual"] = actual
    packed["predicted"] = predicted
    packed["y_true"] = (
        (actual >= relevance_threshold).astype(int) if actual.size else np.array([], dtype=int)
    )
    packed["y_pred"] = (
        (predicted >= decision).astype(int) if predicted.size else np.array([], dtype=int)
    )
    return packed


@st.cache_data(show_spinner=False)
def load_dataset(data_sig: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del data_sig  # used only as a Streamlit cache key
    ratings, movies, movie_content = content_based.load_data()
    links = pd.read_csv(_find_processed_file("links_clean.csv"))
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce").astype("Int64")
    links["imdbId"] = pd.to_numeric(links["imdbId"], errors="coerce").astype("Int64")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
    links = links.dropna(subset=["movieId"]).drop_duplicates("movieId")

    try:
        posters = pd.read_csv(_find_processed_file("posters.csv"))
        posters["movieId"] = pd.to_numeric(posters["movieId"], errors="coerce").astype("Int64")
        posters = posters.dropna(subset=["movieId"]).drop_duplicates("movieId")
        posters = posters[["movieId", "poster_url"]]
    except FileNotFoundError:
        posters = pd.DataFrame(columns=["movieId", "poster_url"])

    return ratings, movies, movie_content, links, posters


@st.cache_data(show_spinner=False)
def load_tags(data_sig: str) -> pd.DataFrame:
    del data_sig
    tags = pd.read_csv(_find_processed_file("tags_clean.csv"))
    if "tagged_at" in tags.columns:
        tags["tagged_at"] = pd.to_datetime(tags["tagged_at"], utc=True)
    return tags


@st.cache_resource(show_spinner=False)
def train_models(
    data_sig: str,
    n_factors: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
):
    ratings, movies, movie_content, links, posters = load_dataset(data_sig)
    # Shared 80/20 split so the three standalone evaluations are comparable.
    # cf-v2: assignment CF engine (user kNN, item kNN, TruncatedSVD).
    train_ratings, test_ratings = content_based.split_train_test(
        ratings, test_size=test_size, random_state=random_state
    )

    content_model = content_based.ContentBasedRecommender().fit(
        train_ratings, movie_content, movies=movies
    )
    collaborative_model = collaborative_filtering.CollaborativeFiltering(
        n_factors=n_factors
    ).fit(train_ratings, movies=movies)
    best_alpha, tuning_results, hybrid_model, hybrid_metrics = hybrid.tune_alpha(
        train_ratings,
        test_ratings,
        movie_content,
        movies,
        n_factors=n_factors,
        relevance_threshold=4.0,
        metric="f1_score",
    )

    content_eval = content_model.evaluate(test_ratings, relevance_threshold=4.0)
    collaborative_eval = collaborative_model.evaluate(test_ratings, relevance_threshold=4.0)

    test_user_ids = test_ratings["userId"].astype(int).to_numpy()
    test_movie_ids = test_ratings["movieId"].astype(int).to_numpy()
    hybrid_actual = test_ratings["rating"].to_numpy(dtype=float)
    hybrid_predicted = hybrid.blend_hybrid_scores(
        hybrid_model.content_model.predict_many(test_user_ids, test_movie_ids),
        hybrid_model.collaborative_model.predict_many(test_user_ids, test_movie_ids),
        test_movie_ids,
        hybrid_model.item_counts,
        hybrid_model.alpha,
        shrink=hybrid_model.item_shrink,
        min_mix=hybrid_model.min_content_mix,
    )
    all_eval = {
        content_based.ALGORITHM_KEY: content_eval,
        collaborative_filtering.ALGORITHM_KEY: collaborative_eval,
        "hybrid": _pack_eval_arrays(hybrid_metrics["hybrid"], hybrid_actual, hybrid_predicted),
    }
    all_metrics = {key: _scalar_metrics(value) for key, value in all_eval.items()}

    user_ids = sorted(ratings["userId"].astype(int).unique().tolist())
    return (
        hybrid_model,
        content_model,
        collaborative_model,
        all_metrics,
        all_eval,
        tuning_results,
        ratings,
        movies,
        movie_content,
        links,
        posters,
        user_ids,
        len(train_ratings),
        len(test_ratings),
        best_alpha,
    )


def _imdb_url(imdb_id: object) -> str | None:
    if pd.isna(imdb_id):
        return None
    return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/"


def _tmdb_url(tmdb_id: object) -> str | None:
    if pd.isna(tmdb_id):
        return None
    return f"https://www.themoviedb.org/movie/{int(tmdb_id)}"


def attach_meta(
    frame: pd.DataFrame,
    links: pd.DataFrame,
    posters: pd.DataFrame,
    rating_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    enriched = frame.merge(links[["movieId", "imdbId", "tmdbId"]], on="movieId", how="left")
    enriched = enriched.merge(posters, on="movieId", how="left")
    if rating_stats is not None and not rating_stats.empty:
        enriched = enriched.merge(rating_stats, on="movieId", how="left")
    enriched["IMDb"] = enriched["imdbId"].map(_imdb_url)
    enriched["TMDB"] = enriched["tmdbId"].map(_tmdb_url)
    return enriched


def movie_rating_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.groupby("movieId", as_index=False).agg(
        avg_rating=("rating", "mean"),
        n_ratings=("rating", "size"),
    )


def user_summary(ratings: pd.DataFrame, user_id: int) -> dict[str, float]:
    user = ratings.loc[ratings["userId"] == user_id]
    return {
        "movies_rated": int(len(user)),
        "avg_rating": float(user["rating"].mean()) if len(user) else 0.0,
        "liked": int((user["rating"] >= 4.0).sum()),
        "disliked": int((user["rating"] <= 2.0).sum()),
    }


def user_history(
    ratings: pd.DataFrame, movies: pd.DataFrame, user_id: int
) -> pd.DataFrame:
    history = ratings.loc[ratings["userId"] == user_id].copy()
    history = history.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    history = history.sort_values("rating", ascending=False)
    return history[["movieId", "title", "genres", "rating"]].reset_index(drop=True)


def get_recommendations(
    algorithm: str,
    user_id: int,
    top_n: int,
    alpha: float,
    links: pd.DataFrame,
    posters: pd.DataFrame,
    content_model: content_based.ContentBasedRecommender | None = None,
    collaborative_model: collaborative_filtering.CollaborativeFiltering | None = None,
    hybrid_model: HybridRecommender | None = None,
    rating_stats: pd.DataFrame | None = None,
    diversify: bool = False,
    diversity: float = 0.3,
    cf_variant: str = collaborative_filtering.VARIANT_SVD,
    cf_k: int = collaborative_filtering.DEFAULT_NEIGHBORHOOD_K,
    cf_genres: list[str] | tuple[str, ...] | None = None,
    cf_item_method: str = collaborative_filtering.DEFAULT_ITEM_METHOD,
    cf_n_components: int = collaborative_filtering.DEFAULT_N_COMPONENTS,
) -> tuple[pd.DataFrame, list[str]]:
    if algorithm == content_based.ALGORITHM_KEY:
        if content_model is None:
            raise RuntimeError("Content-based model is not trained.")
        recs = content_model.recommend(
            user_id,
            n_recommendations=top_n,
            diversify=diversify,
            diversity=diversity,
        )
        recs = attach_meta(recs, links, posters, rating_stats=rating_stats)
        display = recs.rename(columns={"predicted_rating": "Score"})
        return display, ["Score"]

    if algorithm == collaborative_filtering.ALGORITHM_KEY:
        if collaborative_model is None:
            raise RuntimeError("Collaborative model is not trained.")
        recs = collaborative_model.recommend(
            user_id,
            n_recommendations=top_n,
            variant=cf_variant,
            neighborhood_k=cf_k,
            genres=cf_genres,
            item_method=cf_item_method,
            n_components=cf_n_components,
        )
        recs = attach_meta(recs, links, posters, rating_stats=rating_stats)
        display = recs.rename(columns={"predicted_rating": "Score"})
        return display, ["Score"]

    if hybrid_model is None:
        raise RuntimeError("Hybrid model is not trained.")
    previous_alpha = hybrid_model.alpha
    hybrid_model.alpha = float(alpha)
    try:
        recs = hybrid_model.recommend(user_id, n_recommendations=top_n)
        recs = attach_meta(recs, links, posters, rating_stats=rating_stats)
        display = recs.rename(
            columns={
                "content_rating": "Content score",
                "cf_rating": "Collaborative score",
                "hybrid_rating": "Hybrid score",
            }
        )
        return display, ["Content score", "Collaborative score", "Hybrid score"]
    finally:
        hybrid_model.alpha = previous_alpha


def _genre_labels(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                value = "|".join(str(item) for item in parsed)
        except (SyntaxError, ValueError):
            pass
    return [
        part.strip()
        for part in str(value).split("|")
        if part.strip() and part.strip() != "(no genres listed)"
    ]


def _score_pill_class(column: str) -> str:
    name = str(column).lower()
    if "content" in name:
        return "content"
    if "collab" in name or "svd" in name:
        return "collab"
    if "hybrid" in name:
        return "hybrid"
    return "score"


def _stat_tiles_html(tiles: list[tuple[str, str, str]]) -> str:
    cards = []
    for label, value, tone in tiles:
        cards.append(
            '<div class="stat-card '
            + tone
            + '">'
            f'<div class="stat-label">{escape(label)}</div>'
            f'<div class="stat-value">{escape(value)}</div>'
            "</div>"
        )
    return (
        f'<div class="stat-grid cols-{len(tiles)}">{"".join(cards)}</div>'
    )


def _movie_card_html(row: pd.Series, score_columns: list[str]) -> str:
    title = escape(str(row.get("title") or "Unknown title"))
    poster_url = row.get("poster_url")
    tmdb = row.get("TMDB") if pd.notna(row.get("TMDB")) else None
    imdb = row.get("IMDb") if pd.notna(row.get("IMDb")) else None
    href = tmdb or imdb

    if pd.notna(poster_url) and str(poster_url).strip():
        poster = (
            f'<img class="poster" src="{escape(str(poster_url), quote=True)}" alt="{title}">'
        )
        if href:
            poster = (
                f'<a href="{escape(str(href), quote=True)}" target="_blank" rel="noopener">'
                f"{poster}</a>"
            )
    else:
        poster = '<div class="poster-fallback">No poster</div>'

    genres = _genre_labels(row.get("genres"))
    chips = "".join(
        '<span class="chip" style="background:'
        + content_based.GENRE_COLORS.get(genre, content_based.DEFAULT_GENRE_COLOR)
        + f';color:#fff">{escape(genre)}</span>'
        for genre in genres
    )

    meta: list[str] = []
    avg = row.get("avg_rating")
    count = row.get("n_ratings")
    if pd.notna(avg):
        meta.append(f'<span class="star">★ {float(avg):.1f}</span>')
    if pd.notna(count):
        n_ratings = int(count)
        label = "rating" if n_ratings == 1 else "ratings"
        meta.append(f'<span class="count">{n_ratings:,} {label}</span>')
    for column in score_columns:
        if column in row.index and pd.notna(row[column]):
            short = str(column).replace(" score", "")
            meta.append(
                f'<span class="pred {_score_pill_class(column)}">'
                f"{escape(short)} {float(row[column]):.1f}</span>"
            )

    links: list[str] = []
    if imdb:
        links.append(
            f'<a href="{escape(str(imdb), quote=True)}" target="_blank" rel="noopener">IMDb</a>'
        )
    if tmdb:
        links.append(
            f'<a href="{escape(str(tmdb), quote=True)}" target="_blank" rel="noopener">TMDB</a>'
        )

    chip_row = f'<div class="chips">{chips}</div>'
    meta_row = f'<div class="meta">{"".join(meta)}</div>'
    link_row = f'<div class="links">{" · ".join(links)}</div>'
    return (
        '<div class="movie-grid-card">'
        f"{poster}"
        f'<div class="title">{title}</div>'
        f"{chip_row}{meta_row}{link_row}"
        "</div>"
    )


def render_recommendation_cards(frame: pd.DataFrame, score_columns: list[str]) -> None:
    if frame.empty:
        st.info("No recommendations available.")
        return
    cards = "".join(
        _movie_card_html(row, score_columns) for _, row in frame.iterrows()
    )
    st.html(f'<div class="movie-grid">{cards}</div>')


def _metric_cards(metrics: dict[str, float]) -> None:
    st.html(
        _stat_tiles_html(
            [
                ("RMSE", f"{metrics['rmse']:.4f}", "sky"),
                ("Precision", f"{metrics['precision']:.4f}", "mint"),
                ("Recall", f"{metrics['recall']:.4f}", "violet"),
                ("F1-score", f"{metrics['f1_score']:.4f}", "rose"),
                ("Accuracy", f"{metrics['accuracy']:.4f}", "amber"),
                (
                    "Pred. liked cutoff",
                    f"{metrics.get('decision_threshold', 4.0):.2f}",
                    "coral",
                ),
            ]
        )
    )


def _evaluation_classification_chart(metrics: dict) -> plt.Figure:
    labels = ["Precision", "Recall", "F1-score", "Accuracy"]
    values = [
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["f1_score"]),
        float(metrics["accuracy"]),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bars = ax.bar(labels, values, color=["#5B8FF9", "#5AD8A6", "#E85D75", "#FFC857"], width=0.55)
    _add_bar_labels(ax, bars, values, ".3f", 0.02)
    _style_chart_axes(ax, "Score (higher is better)")
    ax.set_ylim(0, 1.08)
    fig.tight_layout()
    return fig


def _evaluation_pred_vs_actual_chart(actual: np.ndarray, predicted: np.ndarray) -> plt.Figure:
    sample_actual = np.asarray(actual, dtype=float)
    sample_predicted = np.asarray(predicted, dtype=float)
    if sample_actual.size > 2500:
        idx = np.random.default_rng(42).choice(sample_actual.size, size=2500, replace=False)
        sample_actual = sample_actual[idx]
        sample_predicted = sample_predicted[idx]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.scatter(sample_actual, sample_predicted, alpha=0.22, s=10, color="#E85D75", linewidths=0)
    ax.plot([0.5, 5.0], [0.5, 5.0], color="#444444", linestyle="--", linewidth=1)
    _style_chart_axes(ax, "Predicted rating")
    ax.set_xlabel("Actual rating", color=_CHART_TEXT, fontsize=10)
    ax.set_xlim(0.4, 5.2)
    ax.set_ylim(0.4, 5.2)
    fig.tight_layout()
    return fig


def _evaluation_residual_chart(actual: np.ndarray, predicted: np.ndarray) -> plt.Figure:
    residuals = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.hist(residuals, bins=30, color="#5B8FF9")
    ax.axvline(0, color="#E85D75", linewidth=1.2)
    _style_chart_axes(ax, "Number of ratings")
    ax.set_xlabel("Predicted − actual", color=_CHART_TEXT, fontsize=10)
    fig.tight_layout()
    return fig


def _evaluation_confusion_chart(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = np.array(
        [
            [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
            [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.imshow(matrix, cmap="Blues")
    peak = float(matrix.max()) if matrix.max() else 1.0
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="#FFFFFF" if matrix[row, col] > peak * 0.55 else _CHART_TEXT,
            )
    ax.set_xticks([0, 1], ["Pred. not liked", "Pred. liked"], color=_CHART_TEXT)
    ax.set_yticks([0, 1], ["Actual not liked", "Actual liked"], color=_CHART_TEXT)
    ax.set_title("Liked vs not liked", color=_CHART_TEXT, fontsize=11)
    fig.patch.set_facecolor(_CHART_BG)
    ax.set_facecolor(_CHART_BG)
    for spine in ax.spines.values():
        spine.set_color(_CHART_GRID)
    fig.tight_layout()
    return fig


_CHART_BG = "#FFFFFF"
_CHART_TEXT = "#222222"
_CHART_GRID = "#D0D3DA"
_CHART_PANEL = "#FFFFFF"

SHORT_ALGORITHM_LABELS = {
    "Content-based (TF-IDF)": "Content-based",
    "Collaborative (SVD)": "Collaborative",
    "Hybrid": "Hybrid",
}


def _style_chart_axes(ax, ylabel: str) -> None:
    ax.set_facecolor(_CHART_BG)
    ax.figure.set_facecolor(_CHART_BG)
    ax.set_ylabel(ylabel, color=_CHART_TEXT, fontsize=10)
    ax.tick_params(colors=_CHART_TEXT, labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_CHART_GRID)
    ax.spines["bottom"].set_color(_CHART_GRID)
    ax.yaxis.grid(True, color=_CHART_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _add_bar_labels(ax, bars, values: list[float], fmt: str, offset: float) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            format(value, fmt),
            ha="center",
            va="bottom",
            color=_CHART_TEXT,
            fontsize=9,
            fontweight="bold",
        )


def _classification_comparison_chart(comparison: pd.DataFrame) -> plt.Figure:
    """Side-by-side bars per metric so models can be compared without stacking."""
    metrics = list(CLASSIFICATION_METRICS)
    algorithms = list(COMPARISON_ALGORITHMS)
    by_algo = comparison.set_index("Algorithm")
    x = list(range(len(metrics)))
    width = 0.24
    n = len(algorithms)

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for index, algorithm in enumerate(algorithms):
        values = [float(by_algo.loc[algorithm, metric]) for metric in metrics]
        offsets = [pos + (index - (n - 1) / 2) * width for pos in x]
        bars = ax.bar(offsets, values, width, label=algorithm, color=MODEL_COLORS[algorithm])
        _add_bar_labels(ax, bars, values, ".3f", 0.012)

    _style_chart_axes(ax, "Score (higher is better)")
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.08)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        fontsize=9,
        facecolor=_CHART_PANEL,
        edgecolor=_CHART_GRID,
        labelcolor=_CHART_TEXT,
    )
    legend.get_frame().set_alpha(1)
    fig.tight_layout()
    return fig


def _alpha_sweep_chart(tune_display: pd.DataFrame, best_alpha: float) -> plt.Figure:
    """Alpha sweep with labeled axes, zoomed y-scale, and the F1-tuned alpha marked."""
    metrics = ["Precision", "Recall", "F1-score", "Accuracy"]
    colors = {
        "Precision": "#5B8FF9",
        "Recall": "#5AD8A6",
        "F1-score": "#E85D75",
        "Accuracy": "#FFC857",
    }
    alphas = tune_display["alpha"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for metric in metrics:
        values = tune_display[metric].to_numpy(dtype=float)
        ax.plot(
            alphas,
            values,
            color=colors[metric],
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            label=metric,
        )
    ax.axvline(
        float(best_alpha),
        color="#31333F",
        linestyle="--",
        linewidth=1.3,
        label=f"F1-tuned alpha ({best_alpha:.2f})",
    )
    stacked = np.concatenate([tune_display[metric].to_numpy(dtype=float) for metric in metrics])
    pad = max(0.04, float(stacked.max() - stacked.min()) * 0.35)
    ax.set_ylim(max(0.0, float(stacked.min()) - pad), min(1.02, float(stacked.max()) + pad))
    ax.set_xlim(float(alphas.min()) - 0.03, float(alphas.max()) + 0.03)
    if len(alphas) > 10:
        ax.set_xticks(alphas[::2])
    else:
        ax.set_xticks(alphas)
    _style_chart_axes(ax, "Score (higher is better)")
    ax.set_xlabel("Alpha — maximum content weight (0 = more SVD, 1 = more content)", color=_CHART_TEXT, fontsize=10)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=True,
        fontsize=9,
        facecolor=_CHART_PANEL,
        edgecolor=_CHART_GRID,
        labelcolor=_CHART_TEXT,
    )
    legend.get_frame().set_alpha(1)
    fig.tight_layout()
    return fig


def _f1_comparison_chart(comparison: pd.DataFrame) -> plt.Figure:
    """Zoom the F1 axis so small gaps between models are visible."""
    algorithms = list(COMPARISON_ALGORITHMS)
    by_algo = comparison.set_index("Algorithm")
    values = [float(by_algo.loc[algorithm, "F1-score"]) for algorithm in algorithms]
    colors = [MODEL_COLORS[algorithm] for algorithm in algorithms]
    labels = [SHORT_ALGORITHM_LABELS[algorithm] for algorithm in algorithms]

    f1_min = min(values)
    f1_max = max(values)
    pad = max(0.02, (f1_max - f1_min) * 0.7)
    y_min = max(0.0, f1_min - pad)
    y_max = min(1.05, f1_max + pad)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    _add_bar_labels(ax, bars, values, ".4f", (y_max - y_min) * 0.02)
    _style_chart_axes(ax, "F1-score (higher is better)")
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    return fig


def _comparison_ranking(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank models on RMSE (lower better) and classification metrics (higher better)."""
    ranked = comparison[["Algorithm"]].copy()
    directions = {"RMSE": True, **{name: False for name in CLASSIFICATION_METRICS}}
    rank_columns = []
    for metric, lower_better in directions.items():
        column = f"{metric} rank"
        ranked[column] = comparison[metric].rank(method="min", ascending=lower_better).astype(int)
        rank_columns.append(column)
    ranked["wins"] = (ranked[rank_columns] == 1).sum(axis=1)
    ranked["avg_rank"] = ranked[rank_columns].mean(axis=1)
    ranked = ranked.sort_values(["avg_rank", "wins"], ascending=[True, False]).reset_index(drop=True)
    ranked["place"] = range(1, len(ranked) + 1)
    winners = []
    for _, row in ranked.iterrows():
        won = [metric for metric, col in zip(directions, rank_columns, strict=True) if row[col] == 1]
        winners.append(won)
    ranked["won_metrics"] = winners
    return ranked.merge(comparison, on="Algorithm")


def _rank_reason(row: pd.Series) -> str:
    won = row["won_metrics"]
    if len(won) == 5:
        return "Wins every metric"
    if won:
        return "Best on " + ", ".join(won)
    if row["place"] == 2:
        return "Second on every metric"
    return "Third on every metric"


def _ranking_board_html(ranked: pd.DataFrame) -> str:
    n = max(len(ranked), 1)
    best = {
        "RMSE": float(ranked["RMSE"].min()),
        "F1-score": float(ranked["F1-score"].max()),
        "Precision": float(ranked["Precision"].max()),
        "Accuracy": float(ranked["Accuracy"].max()),
    }
    headlines = {
        1: "🥇 1st",
        2: "🥈 2nd",
        3: "🥉 3rd",
    }
    cards: list[str] = []
    for _, row in ranked.iterrows():
        place = int(row["place"])
        title = f"{headlines.get(place, f'{place}th')} — {row['Algorithm']}"
        won = row["won_metrics"]
        chips = (
            "".join(
                f'<span class="rank-chip chip-{escape(str(metric)).lower().replace(" ", "-")}">'
                f"{escape(str(metric))}</span>"
                for metric in won
            )
            if won
            else '<span class="rank-chip empty">No first-place metrics</span>'
        )
        stats = []
        for label, key, fmt in (
            ("RMSE", "RMSE", ".4f"),
            ("F1-score", "F1-score", ".4f"),
            ("Precision", "Precision", ".4f"),
            ("Accuracy", "Accuracy", ".4f"),
        ):
            value = float(row[key])
            extra = " best" if abs(value - best[key]) < 1e-9 else ""
            stats.append(
                f'<div class="rank-stat{extra}">'
                f'<span class="lbl">{escape(label)}</span>'
                f'<span class="val">{value:{fmt}}</span>'
                "</div>"
            )
        cards.append(
            f'<article class="rank-card place-{place}">'
            f'<div class="rank-name">{escape(title)}</div>'
            f'<div class="rank-why">{escape(_rank_reason(row))}</div>'
            f'<div class="rank-chips">{chips}</div>'
            f'<div class="rank-stats">{"".join(stats)}</div>'
            "</article>"
        )
    return (
        f'<div class="rank-board" style="grid-template-columns: repeat({n}, minmax(0, 1fr));">'
        f"{''.join(cards)}</div>"
    )


def _render_comparison_ranking(ranked: pd.DataFrame) -> None:
    st.markdown("#### Ranking")
    st.caption("Overall place uses average rank across RMSE, precision, recall, F1-score, and accuracy. Green scores are the best in that metric.")
    st.html(_ranking_board_html(ranked))


def _render_comparison_conclusion(ranked: pd.DataFrame) -> None:
    winner = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else None
    third = ranked.iloc[2] if len(ranked) > 2 else None
    won = winner["won_metrics"]
    st.markdown("#### Conclusion")
    with st.container(border=True):
        st.markdown(f":material/emoji_events: **{winner['Algorithm']}** is the best model")
        if len(won) == 5:
            st.caption(
                "On the same 80/20 test split it wins every metric: lowest RMSE and the highest "
                "precision, recall, F1-score, and accuracy."
            )
        else:
            st.caption(
                f"Average rank {winner['avg_rank']:.1f}. Best on {', '.join(won) or 'none'}."
            )
        st.markdown(
            f":green-badge[RMSE {winner['RMSE']:.4f}] "
            f":green-badge[F1 {winner['F1-score']:.4f}] "
            f":green-badge[Precision {winner['Precision']:.4f}] "
            f":green-badge[Recall {winner['Recall']:.4f}] "
            f":green-badge[Accuracy {winner['Accuracy']:.4f}]"
        )
        notes: list[str] = []
        if second is not None:
            notes.append(
                f"**{second['Algorithm']}** is second (RMSE {second['RMSE']:.4f}, "
                f"F1 {second['F1-score']:.4f})."
            )
        if third is not None:
            notes.append(
                f"**{third['Algorithm']}** is third — useful for similar-movie / cold-start cases, "
                "but weakest at predicting held-out ratings."
            )
        if notes:
            st.markdown(" ".join(notes))


def _poster_notice(posters: pd.DataFrame) -> None:
    if posters.empty or posters["poster_url"].isna().all():
        st.info(
            "Posters not found. Run `py fetch_posters.py` once to create "
            "`processed/posters.csv`. Links still work without posters."
        )


def _load_recs(
    *,
    algorithm: str,
    user_id: int,
    top_n: int,
    alpha: float,
    links: pd.DataFrame,
    posters: pd.DataFrame,
    content_model,
    collaborative_model,
    hybrid_model,
    rating_stats,
    diversify: bool = False,
    diversity: float = 0.3,
    cf_variant: str = collaborative_filtering.VARIANT_USER,
    cf_k: int = collaborative_filtering.DEFAULT_NEIGHBORHOOD_K,
    cf_genres: list[str] | tuple[str, ...] | None = None,
    cf_item_method: str = collaborative_filtering.DEFAULT_ITEM_METHOD,
    cf_n_components: int = collaborative_filtering.DEFAULT_N_COMPONENTS,
) -> tuple[pd.DataFrame, list[str]] | None:
    cache_key = (
        algorithm,
        int(user_id),
        int(top_n),
        round(float(alpha), 4),
        bool(diversify),
        round(float(diversity), 4),
        str(cf_variant),
        int(cf_k),
        tuple(cf_genres or ()),
        str(cf_item_method),
        int(cf_n_components),
    )
    cached = st.session_state.get("_recs_cache")
    if cached and cached[0] == cache_key:
        return cached[1], cached[2]
    try:
        with st.spinner("Generating recommendations..."):
            display, score_cols = get_recommendations(
                algorithm=algorithm,
                user_id=int(user_id),
                top_n=int(top_n),
                alpha=float(alpha),
                links=links,
                posters=posters,
                content_model=content_model,
                collaborative_model=collaborative_model,
                hybrid_model=hybrid_model,
                rating_stats=rating_stats,
                diversify=bool(diversify),
                diversity=float(diversity),
                cf_variant=str(cf_variant),
                cf_k=int(cf_k),
                cf_genres=cf_genres,
                cf_item_method=str(cf_item_method),
                cf_n_components=int(cf_n_components),
            )
    except ValueError as error:
        st.error(str(error))
        return None
    st.session_state["_recs_cache"] = (cache_key, display, score_cols)
    return display, score_cols


def render_user_panel(ratings: pd.DataFrame, movies: pd.DataFrame, user_id: int) -> None:
    summary = user_summary(ratings, int(user_id))
    st.html(
        _stat_tiles_html(
            [
                ("Movies Rated", f"{summary['movies_rated']:,}", "rose"),
                ("Avg Rating", f"{summary['avg_rating']:.2f}", "amber"),
                ("Liked (≥4★)", f"{summary['liked']:,}", "mint"),
                ("Disliked (≤2★)", f"{summary['disliked']:,}", "violet"),
            ]
        )
    )
    with st.expander(f"👤 User {user_id}'s Rating History"):
        history = user_history(ratings, movies, int(user_id))
        st.dataframe(history, use_container_width=True, hide_index=True)


def _recs_table(display: pd.DataFrame) -> None:
    with st.expander("Table view"):
        table = display.drop(columns=["imdbId", "tmdbId", "poster_url"], errors="ignore")
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "IMDb": st.column_config.LinkColumn("IMDb"),
                "TMDB": st.column_config.LinkColumn("TMDB"),
            },
        )


def render_hybrid_recs(
    *,
    user_id: int,
    top_n: int,
    alpha: float,
    links,
    posters,
    hybrid_model,
    rating_stats,
) -> None:
    st.subheader(f"🏆 Top {top_n} hybrid recommendations")
    st.caption(
        "Support-aware hybrid: more SVD when a movie has many ratings, more content "
        "(title/year/genres/tags) when it is rarely rated."
    )
    loaded = _load_recs(
        algorithm="hybrid",
        user_id=user_id,
        top_n=top_n,
        alpha=alpha,
        links=links,
        posters=posters,
        content_model=None,
        collaborative_model=None,
        hybrid_model=hybrid_model,
        rating_stats=rating_stats,
    )
    if loaded is None:
        return
    display, _score_cols = loaded
    render_recommendation_cards(display, [])
    reasons = {
        int(row["movieId"]): hybrid_model.recommendation_reasons(row, user_id=int(user_id))
        for _, row in display.iterrows()
    }
    st.html(hybrid.explain_blend_table_html(display, reasons))
    _recs_table(display)


def render_hybrid_weight(
    *,
    user_id: int,
    top_n: int,
    alpha: float,
    best_alpha: float,
    tuning_results: pd.DataFrame,
    links,
    posters,
    hybrid_model,
    rating_stats,
) -> None:
    st.subheader("⚖️ Content vs Collaborative weight")
    st.write(
        "The hybrid rating is `content_weight × content score + collaborative_weight × SVD score`. "
        "Alpha is the **maximum** content weight. Rarely rated movies lean on content; "
        "popular movies stay closer to SVD."
    )
    st.html(
        _stat_tiles_html(
            [
                ("Current alpha", f"{alpha:.2f}", "rose"),
                ("F1-tuned alpha", f"{best_alpha:.2f}", "sky"),
            ]
        )
    )
    st.pyplot(
        hybrid.plot_blend_weights(
            float(alpha),
            shrink=hybrid_model.item_shrink,
            min_mix=hybrid_model.min_content_mix,
        ),
        clear_figure=True,
        width="content",
    )
    st.caption("As a movie collects more ratings, the blend shifts toward collaborative filtering.")

    loaded = _load_recs(
        algorithm="hybrid",
        user_id=user_id,
        top_n=top_n,
        alpha=alpha,
        links=links,
        posters=posters,
        content_model=None,
        collaborative_model=None,
        hybrid_model=hybrid_model,
        rating_stats=rating_stats,
    )
    if loaded is not None:
        display, _score_cols = loaded
        source_counts = display["blend_source"].value_counts()
        st.html(
            _stat_tiles_html(
                [
                    (
                        "Content-leaning",
                        str(int(source_counts.get("Content-leaning", 0))),
                        "sky",
                    ),
                    ("Balanced", str(int(source_counts.get("Balanced", 0))), "violet"),
                    (
                        "Collaborative-leaning",
                        str(int(source_counts.get("Collaborative-leaning", 0))),
                        "mint",
                    ),
                ]
            )
        )

    st.markdown("#### How alpha changes liked / not-liked metrics")
    st.caption(
        "Each point is Hybrid on the 80/20 test set at that alpha. "
        "**Alpha** is the maximum content weight: **0 = more SVD**, **1 = more content**. "
        "The dashed line is the F1-tuned alpha (the default). "
        "The y-axis is zoomed in so small gaps are visible — nearly flat lines mean alpha barely changes these scores."
    )
    tune_display = tuning_results.rename(
        columns={
            "rmse": "RMSE",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-score",
            "accuracy": "Accuracy",
            "decision_threshold": "Pred. liked cutoff",
        }
    ).round(4)
    if not tune_display.empty and "alpha" in tune_display.columns:
        st.pyplot(
            _alpha_sweep_chart(tune_display, float(best_alpha)),
            clear_figure=True,
        )
    st.dataframe(tune_display, use_container_width=True, hide_index=True)


def render_hybrid_search(
    *,
    user_id: int,
    top_n: int,
    alpha: float,
    movies: pd.DataFrame,
    links,
    posters,
    hybrid_model,
    rating_stats,
) -> None:
    st.subheader("🔍 Search a movie")
    st.caption(
        "Find a title, then see how much the selected user is predicted to like it "
        "(content score + SVD score mixed into a hybrid rating) and similar movies they have not rated."
    )
    genre_options = content_based.catalog_genres(movies)
    with st.form("hybrid_search_form", border=False):
        search_query = st.text_input("Movie title", placeholder="e.g. Toy Story")
        search_genres = st.multiselect(
            "Genres",
            options=genre_options,
            placeholder="Optional. Pick one or more genres",
        )
        search_submitted = st.form_submit_button("Search")
    if search_submitted:
        if not search_query.strip() and not search_genres:
            st.warning("Enter a title and/or pick at least one genre.")
        else:
            st.session_state["hybrid_search_query"] = search_query.strip()
            st.session_state["hybrid_search_genres"] = tuple(search_genres)
            st.session_state.pop("hybrid_search_run", None)
    stored_query = st.session_state.get("hybrid_search_query", "")
    stored_genres = tuple(st.session_state.get("hybrid_search_genres") or ())
    if stored_query or stored_genres:
        matches = hybrid_model.search_movies(stored_query, genres=stored_genres)
        if matches.empty:
            bits = []
            if stored_query:
                bits.append(f'title "{stored_query}"')
            if stored_genres:
                bits.append("genre " + " + ".join(stored_genres))
            st.info("No movies match " + " and ".join(bits) + ".")
        else:
            matches = matches.copy()
            matches["label"] = matches["title"].astype(str)
            if "genres" in matches.columns:
                matches["label"] = (
                    matches["label"] + "  ·  " + matches["genres"].fillna("").astype(str)
                )
            if matches["label"].duplicated().any():
                matches["label"] = (
                    matches["label"] + " [" + matches["movieId"].astype(str) + "]"
                )
            picked_label = st.selectbox(
                "Pick a movie from the results",
                matches["label"].tolist(),
                key="hybrid_search_title",
            )
            picked_id = int(
                matches.loc[matches["label"] == picked_label, "movieId"].iloc[0]
            )
            st.caption(
                "This does not add a rating. It estimates a 0.5–5★ score for the User ID in the sidebar."
            )
            if st.button("Show predicted rating for this user", type="primary"):
                st.session_state["hybrid_search_run"] = (
                    int(user_id),
                    picked_id,
                    stored_query,
                    int(top_n),
                    float(alpha),
                )
    run = st.session_state.get("hybrid_search_run")
    if not (run and run[0] == int(user_id)):
        return
    _, movie_id, _, similar_n, search_alpha = run
    previous_alpha = hybrid_model.alpha
    hybrid_model.alpha = float(search_alpha)
    try:
        scored = hybrid_model.score_movie(int(user_id), int(movie_id))
        scored = attach_meta(scored, links, posters, rating_stats)
        scored = scored.rename(
            columns={
                "content_rating": "Content score",
                "cf_rating": "Collaborative score",
                "hybrid_rating": "Hybrid score",
            }
        )
        st.markdown(f"**Predicted rating for user {user_id}**")
        st.caption("Content = genres/tags · SVD = people with similar ratings · Hybrid = the mix.")
        render_recommendation_cards(
            scored, ["Content score", "Collaborative score", "Hybrid score"]
        )
        row = scored.iloc[0]
        your_rating = row.get("your_rating")
        if pd.notna(your_rating):
            st.caption(f"You already rated this movie **{float(your_rating):.1f}★**.")
        else:
            st.caption("You have not rated this movie yet.")
        reasons = {int(row["movieId"]): hybrid_model.recommendation_reasons(row)}
        st.html(hybrid.explain_blend_table_html(scored, reasons))

        similar = hybrid_model.similar_to_movie(
            int(user_id),
            int(movie_id),
            n_recommendations=int(similar_n),
        )
        similar = attach_meta(similar, links, posters, rating_stats)
        similar = similar.rename(
            columns={
                "content_rating": "Content score",
                "cf_rating": "Collaborative score",
                "hybrid_rating": "Hybrid score",
            }
        )
        seed_title = str(row.get("title") or "this movie")
        st.markdown(f"**Similar unseen movies · {seed_title}**")
        st.caption("Closest in genres/tags/title, then ranked by this user's hybrid score.")
        render_recommendation_cards(
            similar, ["Content score", "Collaborative score", "Hybrid score"]
        )
    except ValueError as error:
        st.info(str(error))
    finally:
        hybrid_model.alpha = previous_alpha


def render_content_recs(
    *,
    user_id: int,
    top_n: int,
    links,
    posters,
    content_model,
    rating_stats,
    diversify: bool,
    diversity: float,
) -> None:
    st.subheader(f"🏆 Top {top_n} content-based recommendations")
    st.caption("TF-IDF on genres + tags, ranked by cosine similarity to this user's profile.")
    loaded = _load_recs(
        algorithm=content_based.ALGORITHM_KEY,
        user_id=user_id,
        top_n=top_n,
        alpha=0.0,
        links=links,
        posters=posters,
        content_model=content_model,
        collaborative_model=None,
        hybrid_model=None,
        rating_stats=rating_stats,
        diversify=diversify,
        diversity=diversity,
    )
    if loaded is None:
        return
    display, _score_cols = loaded
    render_recommendation_cards(display, [])
    reasons = {
        int(row.movieId): content_model.recommendation_reasons(int(user_id), int(row.movieId))
        for row in display.itertuples()
    }
    st.html(
        content_based.why_recommended_table_html(
            display, reasons, empty_why="Matches your content profile"
        )
    )
    if diversify:
        baseline = content_model.recommend(user_id, n_recommendations=top_n, diversify=False)
        st.markdown("**🔀 Before vs after diversity**")
        st.caption(
            "Same rank, two rankings: relevance-only on the left, MMR-diversified "
            "(current diversity strength) on the right. Highlighted titles on the "
            "right replaced a different movie at that rank."
        )
        st.html(content_based.diversity_comparison_table_html(baseline, display))
    _recs_table(display)


def render_content_like_genre(
    *,
    user_id: int,
    top_n: int,
    movies: pd.DataFrame,
    links,
    posters,
    content_model,
    rating_stats,
) -> None:
    st.subheader("🎬 More like this genre")
    genre_options = content_based.catalog_genres(movies)
    with st.form("like_genre_form", border=False):
        picked_genres = st.multiselect(
            "Pick genres",
            options=genre_options,
            placeholder="Search and select one or more genres",
            help="Type to search. Nothing runs until you click Find similar movies.",
        )
        submitted = st.form_submit_button("Find similar movies")
    if submitted:
        if not picked_genres:
            st.warning("Pick at least one genre.")
        else:
            st.session_state["like_genre_run"] = (
                tuple(picked_genres),
                int(user_id),
                int(top_n),
            )
    run = st.session_state.get("like_genre_run")
    if not (run and run[1] == int(user_id)):
        return
    genre_names, _, similar_n = run
    if isinstance(genre_names, str):
        genre_names = (genre_names,)
    genre_names = [str(name).strip() for name in genre_names if str(name).strip()]
    if not genre_names:
        st.warning("Pick at least one genre.")
        return
    try:
        similar = content_model.similar_to_genre(
            genre_names,
            n_recommendations=similar_n,
            user_id=int(user_id),
        )
    except ValueError as error:
        st.info(str(error))
        return
    similar = attach_meta(similar, links, posters, rating_stats)
    similar = similar.rename(columns={"predicted_rating": "Score"})
    label = " + ".join(genre_names)
    st.caption(
        f"Unseen movies closest to **{label}** (TF-IDF centroid) for user {user_id}."
    )
    render_recommendation_cards(similar, [])


def render_content_profile(ratings, movies, user_id: int) -> None:
    st.subheader(f"👤 Genre profile · user {user_id}")
    st.caption(
        "Built from the user's ratings ≥ 3.0, weighted by rating and split "
        "evenly across each movie's genres, then shown as a share of 100%."
    )
    profile = content_based.user_genre_profile(ratings, movies, int(user_id))
    if profile.empty:
        st.info("This user has no ratings ≥ 3.0 to build a genre profile.")
        return
    st.pyplot(
        content_based.plot_genre_profile(profile, int(user_id)),
        clear_figure=True,
    )


def render_collab_recs(
    *,
    user_id: int,
    top_n: int,
    links,
    posters,
    collaborative_model,
    rating_stats,
    cf_variant: str,
    cf_k: int,
    cf_genres: list[str] | tuple[str, ...] | None,
    cf_item_method: str,
    cf_n_components: int,
) -> None:
    variant_names = {value: key for key, value in collaborative_filtering.VARIANT_OPTIONS.items()}
    variant_name = variant_names.get(cf_variant, "Matrix Factorization (SVD)")
    st.subheader(f"🏆 Top {top_n} collaborative recommendations")
    if cf_variant == collaborative_filtering.VARIANT_USER:
        st.caption(
            f"{variant_name}: similar users (k={cf_k}) vote on movies you have not rated."
        )
    elif cf_variant == collaborative_filtering.VARIANT_ITEM:
        method_names = {
            value: key for key, value in collaborative_filtering.ITEM_METHOD_OPTIONS.items()
        }
        method_name = method_names.get(cf_item_method, cf_item_method)
        st.caption(
            f"{variant_name}: similar movies (k={cf_k}, {method_name}) to ones you already liked."
        )
    else:
        st.caption(
            f"Matrix factorization (SVD) with {cf_n_components} latent factors "
            "on the user–movie rating matrix."
        )
    if cf_genres:
        st.caption("Genre filter: " + ", ".join(cf_genres))
    loaded = _load_recs(
        algorithm=collaborative_filtering.ALGORITHM_KEY,
        user_id=user_id,
        top_n=top_n,
        alpha=0.0,
        links=links,
        posters=posters,
        content_model=None,
        collaborative_model=collaborative_model,
        hybrid_model=None,
        rating_stats=rating_stats,
        cf_variant=cf_variant,
        cf_k=cf_k,
        cf_genres=cf_genres,
        cf_item_method=cf_item_method,
        cf_n_components=cf_n_components,
    )
    if loaded is None:
        return
    if getattr(collaborative_model, "last_is_cold_start", False):
        st.info(str(getattr(collaborative_model, "last_rationale_label", "Popularity fallback")))
    display, _score_cols = loaded
    render_recommendation_cards(display, [])
    reasons = {
        int(row.movieId): collaborative_model.recommendation_reasons(
            int(user_id),
            int(row.movieId),
            variant=cf_variant,
            neighborhood_k=cf_k,
            item_method=cf_item_method,
        )
        for row in display.itertuples()
    }
    st.html(
        content_based.why_recommended_table_html(
            display, reasons, empty_why="Matches users like you"
        )
    )
    _recs_table(display)


def render_evaluation(algorithm_label: str, algorithm: str, all_eval, train_size, test_size) -> None:
    st.subheader("Evaluation (80/20 split)")
    st.caption(
        f"Showing **{algorithm_label}** only. Open another algorithm tab to see a different model. "
        f"Train ratings: {train_size:,} · Test ratings: {test_size:,} · "
        "Liked = actual rating ≥ 4.0."
    )
    selected_eval = all_eval[algorithm]
    _metric_cards(selected_eval)
    actual = selected_eval.get("actual")
    predicted = selected_eval.get("predicted")
    y_true = selected_eval.get("y_true")
    y_pred = selected_eval.get("y_pred")
    has_arrays = (
        actual is not None
        and predicted is not None
        and np.asarray(actual).size > 0
        and np.asarray(predicted).size > 0
    )
    st.markdown("**Classification metrics** — higher is better")
    st.pyplot(_evaluation_classification_chart(selected_eval), clear_figure=True, width="content")
    if not has_arrays:
        return
    st.markdown("**Predicted vs actual ratings**")
    st.caption("Dashed line is a perfect prediction. Points are a sample of the test set.")
    st.pyplot(_evaluation_pred_vs_actual_chart(actual, predicted), clear_figure=True, width="content")
    st.markdown("**Error distribution**")
    st.caption("How far predictions miss the true rating. Closer to 0 is better.")
    st.pyplot(_evaluation_residual_chart(actual, predicted), clear_figure=True, width="content")
    if y_true is not None and y_pred is not None and np.asarray(y_true).size > 0:
        st.markdown("**Liked vs not liked**")
        st.caption("Actual liked = rating ≥ 4.0. Predicted liked uses this model's cutoff.")
        st.pyplot(_evaluation_confusion_chart(y_true, y_pred), clear_figure=True, width="content")


def render_comparison(all_metrics, best_alpha: float, tuning_results: pd.DataFrame) -> None:
    st.subheader("Model Comparison")
    st.caption(
        f"Each row is that module's own evaluate() on the same 80/20 split. "
        f"Hybrid uses the F1-tuned alpha ({best_alpha:.2f}) from `hybrid.py` only."
    )
    comparison = pd.DataFrame(
        [
            {"Algorithm": "Content-based (TF-IDF)", **all_metrics[content_based.ALGORITHM_KEY]},
            {
                "Algorithm": "Collaborative (SVD)",
                **all_metrics[collaborative_filtering.ALGORITHM_KEY],
            },
            {"Algorithm": "Hybrid", **all_metrics["hybrid"]},
        ]
    ).rename(
        columns={
            "rmse": "RMSE",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-score",
            "accuracy": "Accuracy",
            "decision_threshold": "Pred. liked cutoff",
        }
    )
    comparison = comparison.round(4)
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.caption(
        "Bars are grouped side by side (not stacked) so each 0–1 metric can be compared across models. "
        "Values are labeled on the bars."
    )
    chart_cols = st.columns((3, 2), gap="large")
    with chart_cols[0]:
        st.markdown("**Classification metrics** — higher is better")
        st.pyplot(_classification_comparison_chart(comparison), clear_figure=True)
    with chart_cols[1]:
        st.markdown("**F1-score** — higher is better")
        st.caption("Axis is zoomed in so the small gaps between models are easier to see.")
        st.pyplot(_f1_comparison_chart(comparison), clear_figure=True)
    ranked = _comparison_ranking(comparison)
    _render_comparison_ranking(ranked)
    _render_comparison_conclusion(ranked)


def render_visualization(data_sig: str, ratings, movies) -> None:
    st.subheader("Data Visualization")
    st.caption("Exploratory analysis of the cleaned MovieLens dataset.")
    tags = load_tags(data_sig)

    st.markdown("#### 1.1 Distribution of user ratings")
    st.caption("How users rate movies and which scores are most common.")
    st.pyplot(data_visualization.make_rating_distribution(ratings), clear_figure=True, width="content")

    st.markdown("#### 1.2 User activity")
    st.caption("How many movies each user has rated.")
    st.pyplot(data_visualization.make_user_activity(ratings), clear_figure=True, width="content")
    st.dataframe(
        data_visualization.user_activity_stats(ratings).to_frame("value"),
        width="content",
    )

    st.markdown("#### 1.3 Movie genre distribution")
    st.caption("Genres with the largest number of movies in the catalog.")
    st.pyplot(data_visualization.make_genre_distribution(movies), clear_figure=True, width="content")

    st.markdown("#### 1.4 Movie release trends")
    st.caption("How the number of movies in the dataset changes across release years.")
    st.pyplot(data_visualization.make_movies_by_year(movies), clear_figure=True, width="content")

    st.markdown("#### 1.5 Top 20 most common tags")
    st.caption("The tags users apply most often.")
    st.pyplot(data_visualization.make_top_tags(tags), clear_figure=True, width="content")

    st.markdown("#### 1.6 Average rating by genre")
    st.caption("Genres that tend to receive higher audience ratings.")
    st.pyplot(
        data_visualization.make_average_rating_by_genre(movies, ratings),
        clear_figure=True,
        width="content",
    )
    st.dataframe(
        data_visualization.genre_rating_stats(movies, ratings),
        width="content",
        hide_index=True,
    )

    st.markdown("#### 1.7 Rating activity over time")
    st.caption("How many ratings were submitted in each year.")
    st.pyplot(data_visualization.make_ratings_over_time(ratings), clear_figure=True, width="content")

    st.markdown("#### 1.8 User-Item Matrix Sparsity")
    st.caption(
        "How much of the ratings matrix is empty, showing why collaborative filtering "
        "must handle sparse data."
    )
    filled_pct, _n_users, _n_movies = data_visualization.sparsity_stats(ratings)
    st.caption(f"**{100 - filled_pct:.1f}%** of the user-item matrix is empty.")
    st.pyplot(
        data_visualization.make_matrix_sparsity_heatmap(ratings),
        clear_figure=True,
        width="content",
    )

    st.markdown("#### 1.9 Distribution of Ratings per Movie")
    st.caption(
        "How many ratings each movie receives, showing a small number of popular movies "
        "versus a long tail of rarely-rated ones."
    )
    st.pyplot(
        data_visualization.make_ratings_per_movie_distribution(ratings),
        clear_figure=True,
        width="content",
    )

    st.markdown("#### 1.10 Cold-Start Segment Size")
    st.caption(
        "The proportion of users and movies with too few ratings to support reliable "
        "recommendations."
    )
    coldstart_fig, pct_users_below, pct_movies_below = data_visualization.make_coldstart_segment_bars(
        ratings
    )
    st.pyplot(coldstart_fig, clear_figure=True, width="content")
    st.caption(
        f"{pct_users_below:.1f}% of users and {pct_movies_below:.1f}% of movies fall "
        "in the cold-start segment."
    )


def render_explorer(movies, ratings, links, posters) -> None:
    st.subheader("Data Explorer")
    filled_pct, n_users, _n_rated_movies = data_visualization.sparsity_stats(ratings)
    st.html(
        _stat_tiles_html(
            [
                ("Users", f"{n_users:,}", "sky"),
                ("Movies", f"{len(movies):,}", "violet"),
                ("Ratings", f"{len(ratings):,}", "rose"),
                ("Matrix filled", f"{filled_pct:.2f}%", "amber"),
            ]
        )
    )
    explore_choice = st.selectbox("Dataset", ["Movies", "Ratings", "Links", "Posters"])
    if explore_choice == "Movies":
        query = st.text_input("Search title / genres")
        view = movies.copy()
        if query.strip():
            mask = view["title"].str.contains(query, case=False, na=False) | view[
                "genres"
            ].str.contains(query, case=False, na=False)
            view = view.loc[mask]
        st.dataframe(view, use_container_width=True, hide_index=True)
    elif explore_choice == "Ratings":
        st.dataframe(ratings.head(1000), use_container_width=True, hide_index=True)
        st.caption("Showing first 1,000 rating rows.")
    elif explore_choice == "Links":
        st.dataframe(links, use_container_width=True, hide_index=True)
    else:
        if posters.empty:
            st.info("No posters.csv found. Run `py fetch_posters.py`.")
        else:
            st.dataframe(posters, use_container_width=True, hide_index=True)


def render_home(
    *,
    ratings,
    movies,
    user_id: int,
    data_sig: str,
    all_metrics,
    best_alpha: float,
    tuning_results,
    links,
    posters,
    top_n: int,
    alpha: float,
    diversify: bool,
    diversity: float,
    hybrid_model,
    content_model,
    collaborative_model,
    rating_stats,
    all_eval,
    train_size,
    test_size,
    cf_variant: str,
    cf_k: int,
    cf_genres: list[str] | tuple[str, ...] | None,
    cf_item_method: str,
    cf_n_components: int,
) -> None:
    render_user_panel(ratings, movies, int(user_id))
    section = st.pills(
        "Page",
        MAIN_TABS,
        default=MAIN_TABS[0],
        key="main_tab",
        label_visibility="collapsed",
    )
    if section is None:
        section = MAIN_TABS[0]

    if section == "🔀 Hybrid":
        render_hybrid_page(
            user_id=int(user_id),
            top_n=int(top_n),
            alpha=float(alpha),
            best_alpha=float(best_alpha),
            movies=movies,
            links=links,
            posters=posters,
            hybrid_model=hybrid_model,
            rating_stats=rating_stats,
            tuning_results=tuning_results,
            all_eval=all_eval,
            train_size=train_size,
            test_size=test_size,
        )
    elif section == "🎬 Content-based":
        render_content_page(
            user_id=int(user_id),
            top_n=int(top_n),
            movies=movies,
            ratings=ratings,
            links=links,
            posters=posters,
            content_model=content_model,
            rating_stats=rating_stats,
            diversify=bool(diversify),
            diversity=float(diversity),
            all_eval=all_eval,
            train_size=train_size,
            test_size=test_size,
        )
    elif section == "👥 Collaborative":
        render_collab_page(
            user_id=int(user_id),
            top_n=int(top_n),
            links=links,
            posters=posters,
            collaborative_model=collaborative_model,
            rating_stats=rating_stats,
            all_eval=all_eval,
            train_size=train_size,
            test_size=test_size,
            cf_variant=cf_variant,
            cf_k=cf_k,
            cf_genres=cf_genres,
            cf_item_method=cf_item_method,
            cf_n_components=cf_n_components,
        )
    elif section == "📈 Model Comparison":
        render_comparison(all_metrics, float(best_alpha), tuning_results)
    elif section == "📉 Data Visualization":
        render_visualization(data_sig, ratings, movies)
    else:
        render_explorer(movies, ratings, links, posters)


def render_hybrid_page(
    *,
    user_id: int,
    top_n: int,
    alpha: float,
    best_alpha: float,
    movies,
    links,
    posters,
    hybrid_model,
    rating_stats,
    tuning_results,
    all_eval,
    train_size,
    test_size,
) -> None:
    st.caption(
        "Combines movie content (genres/tags) with user ratings (SVD). "
        "Rarely rated movies lean on content; popular movies stay closer to SVD."
    )
    _poster_notice(posters)
    features = (
        "🏆 Top 10 recommendations",
        "🔍 Search a movie",
        "⚖️ Content vs Collaborative weight",
        "📊 Evaluation",
    )
    feature = st.pills(
        "Hybrid features",
        features,
        default=features[0],
        key="hybrid_feature",
        label_visibility="collapsed",
    )
    if feature is None:
        feature = features[0]
    if feature == "🏆 Top 10 recommendations":
        render_hybrid_recs(
            user_id=user_id,
            top_n=top_n,
            alpha=alpha,
            links=links,
            posters=posters,
            hybrid_model=hybrid_model,
            rating_stats=rating_stats,
        )
    elif feature == "🔍 Search a movie":
        render_hybrid_search(
            user_id=user_id,
            top_n=top_n,
            alpha=alpha,
            movies=movies,
            links=links,
            posters=posters,
            hybrid_model=hybrid_model,
            rating_stats=rating_stats,
        )
    elif feature == "⚖️ Content vs Collaborative weight":
        render_hybrid_weight(
            user_id=user_id,
            top_n=top_n,
            alpha=alpha,
            best_alpha=best_alpha,
            tuning_results=tuning_results,
            links=links,
            posters=posters,
            hybrid_model=hybrid_model,
            rating_stats=rating_stats,
        )
    else:
        render_evaluation(
            ALGORITHM_LABELS["hybrid"],
            "hybrid",
            all_eval,
            train_size,
            test_size,
        )


def render_content_page(
    *,
    user_id: int,
    top_n: int,
    movies,
    ratings,
    links,
    posters,
    content_model,
    rating_stats,
    diversify: bool,
    diversity: float,
    all_eval,
    train_size,
    test_size,
) -> None:
    st.caption("Uses movie content such as title, year, genres, and tags (TF-IDF).")
    _poster_notice(posters)
    features = (
        "🏆 Top recommendations",
        "🎬 More like this genre",
        "👤 Genre profile",
        "📊 Evaluation",
    )
    feature = st.pills(
        "Content-based features",
        features,
        default=features[0],
        key="content_feature",
        label_visibility="collapsed",
    )
    if feature is None:
        feature = features[0]
    if feature == "🏆 Top recommendations":
        render_content_recs(
            user_id=user_id,
            top_n=top_n,
            links=links,
            posters=posters,
            content_model=content_model,
            rating_stats=rating_stats,
            diversify=diversify,
            diversity=diversity,
        )
    elif feature == "🎬 More like this genre":
        render_content_like_genre(
            user_id=user_id,
            top_n=top_n,
            movies=movies,
            links=links,
            posters=posters,
            content_model=content_model,
            rating_stats=rating_stats,
        )
    elif feature == "👤 Genre profile":
        render_content_profile(ratings, movies, user_id)
    else:
        render_evaluation(
            ALGORITHM_LABELS[content_based.ALGORITHM_KEY],
            content_based.ALGORITHM_KEY,
            all_eval,
            train_size,
            test_size,
        )


def render_collab_page(
    *,
    user_id: int,
    top_n: int,
    links,
    posters,
    collaborative_model,
    rating_stats,
    all_eval,
    train_size,
    test_size,
    cf_variant: str,
    cf_k: int,
    cf_genres: list[str] | tuple[str, ...] | None,
    cf_item_method: str,
    cf_n_components: int,
) -> None:
    st.caption("Uses user ratings and similar-user / similar-movie patterns.")
    _poster_notice(posters)
    features = (
        "🏆 Top recommendations",
        "📊 Evaluation",
    )
    feature = st.pills(
        "Collaborative features",
        features,
        default=features[0],
        key="collab_feature",
        label_visibility="collapsed",
    )
    if feature is None:
        feature = features[0]
    if feature == "🏆 Top recommendations":
        render_collab_recs(
            user_id=user_id,
            top_n=top_n,
            links=links,
            posters=posters,
            collaborative_model=collaborative_model,
            rating_stats=rating_stats,
            cf_variant=cf_variant,
            cf_k=cf_k,
            cf_genres=cf_genres,
            cf_item_method=cf_item_method,
            cf_n_components=cf_n_components,
        )
    else:
        st.caption("Evaluation numbers below are from Matrix Factorization (SVD) on the 80/20 split.")
        render_evaluation(
            ALGORITHM_LABELS[collaborative_filtering.ALGORITHM_KEY],
            collaborative_filtering.ALGORITHM_KEY,
            all_eval,
            train_size,
            test_size,
        )


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #FFFFFF;
        }
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
            border: 1px solid #C5C9D3 !important;
            border-radius: 8px !important;
            background-color: #FFFFFF !important;
            min-height: 2.4rem;
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
            border-color: #E85D75 !important;
        }
        .hero-bar {
            height: 7px;
            border-radius: 999px;
            background: linear-gradient(90deg, #E85D75, #FF8A5B, #FFC857, #5AD8A6, #5B8FF9, #7B5EA7);
            margin: 0 0 1.1rem;
        }
        .stat-grid {
            display: grid;
            gap: 12px;
            margin: 0 0 1rem;
        }
        .stat-grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        .stat-grid.cols-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); }
        @media (max-width: 1100px) {
            .stat-grid.cols-6 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .stat-grid.cols-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        .stat-card {
            border-radius: 16px;
            padding: 14px 16px 12px;
            border: 1px solid transparent;
        }
        .stat-card .stat-label {
            font-size: 0.78rem;
            font-weight: 650;
            margin-bottom: 4px;
        }
        .stat-card .stat-value {
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .stat-card.rose { background: #FFF1F4; border-color: #F8C9D2; color: #C2415C; }
        .stat-card.amber { background: #FFF8E8; border-color: #F3DFA3; color: #B58105; }
        .stat-card.mint { background: #ECFDF5; border-color: #A7F3D0; color: #047857; }
        .stat-card.sky { background: #EFF6FF; border-color: #BFDBFE; color: #1D4ED8; }
        .stat-card.violet { background: #F5F3FF; border-color: #DDD6FE; color: #6D28D9; }
        .stat-card.coral { background: #FFF4ED; border-color: #FED7AA; color: #C2410C; }
        div[data-testid="stBaseButton-pillsActive"] {
            background-color: #E85D75 !important;
            color: #FFFFFF !important;
            border-color: #E85D75 !important;
        }
        div.stButton > button[kind="primary"] {
            background-color: #E85D75;
            border-color: #E85D75;
        }
        .movie-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            column-gap: 18px;
            row-gap: 0;
        }
        .movie-grid-card {
            display: grid;
            grid-template-rows: subgrid;
            grid-row: span 5;
            margin-bottom: 18px;
            overflow: hidden;
            background: #FFFFFF;
            border: 1px solid #F0F2F6;
            border-radius: 16px;
            box-shadow: 0 6px 16px rgba(49, 51, 63, 0.06);
        }
        .movie-grid-card .poster {
            width: 100%;
            aspect-ratio: 2 / 3;
            object-fit: cover;
            border-radius: 16px 16px 0 0;
            display: block;
            background: #F0F2F6;
        }
        .movie-grid-card .poster-fallback {
            width: 100%;
            aspect-ratio: 2 / 3;
            border-radius: 16px 16px 0 0;
            background: linear-gradient(180deg, #FDE68A, #FCA5A5);
            color: #7C2D12;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .movie-grid-card .title {
            color: #31333F;
            font-weight: 700;
            font-size: 0.98rem;
            line-height: 1.3;
            min-height: 2.6em;
            margin: 0.55rem 12px 0.4rem;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        .movie-grid-card .chips {
            display: flex;
            flex-wrap: wrap;
            align-content: flex-start;
            gap: 5px;
            padding: 0 12px 2px;
        }
        .movie-grid-card .chip {
            background: #EEF0F4;
            color: #4B5563;
            border-radius: 999px;
            padding: 3px 9px;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .movie-grid-card .meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            align-content: flex-start;
            gap: 8px;
            margin-top: 0.4rem;
            padding: 0 12px;
            font-size: 0.8rem;
        }
        .movie-grid-card .star { color: #C9A227; font-weight: 600; }
        .movie-grid-card .count { color: #6B7280; }
        .movie-grid-card .pred {
            font-weight: 700;
            border-radius: 999px;
            padding: 1px 8px;
            font-size: 0.72rem;
            background: rgba(232, 93, 117, 0.14);
            color: #E85D75;
        }
        .movie-grid-card .pred.content { background: rgba(91, 143, 249, 0.16); color: #1D4ED8; }
        .movie-grid-card .pred.collab { background: rgba(16, 185, 129, 0.16); color: #047857; }
        .movie-grid-card .pred.hybrid { background: rgba(232, 93, 117, 0.16); color: #C2415C; }
        .movie-grid-card .pred.score { background: rgba(123, 94, 167, 0.14); color: #6D28D9; }
        .movie-grid-card .links {
            margin-top: 0.35rem;
            padding: 0 12px 12px;
            font-size: 0.75rem;
        }
        .movie-grid-card .links a { color: #2563EB; text-decoration: none; }
        .genre-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 2px 9px;
            margin: 2px 4px 2px 0;
            font-size: 0.72rem;
            color: #fff;
            font-weight: 600;
        }
        table.why-table {
            width: 100%;
            border-collapse: collapse;
            color: #31333F;
            font-size: 0.9rem;
        }
        table.why-table th {
            text-align: left;
            padding: 8px 10px;
            border-bottom: 1px solid #E5E7EB;
            font-weight: 600;
            color: #4B5563;
        }
        table.why-table th.you { color: #2563EB; }
        table.why-table th.avg { color: #E76F51; }
        table.why-table td {
            padding: 10px;
            border-bottom: 1px solid #F0F2F6;
            vertical-align: top;
        }
        table.why-table td.num { color: #6B7280; width: 2.2rem; }
        table.why-table td.title-cell { font-weight: 700; }
        table.why-table td.genre-cell { max-width: 16rem; }
        table.why-table .pct-cell span { display: block; margin-bottom: 4px; }
        table.why-table .bar {
            height: 7px;
            background: #EEF0F4;
            border-radius: 999px;
            overflow: hidden;
            min-width: 88px;
        }
        table.why-table .fill { height: 100%; border-radius: 999px; }
        table.why-table .fill-you { background: #4C9AFF; }
        table.why-table .fill-avg { background: #E76F51; }
        table.why-table .why-line { margin: 0 0 4px; color: #047857; }
        table.why-table .why-empty { color: #6B7280; }
        table.why-table tbody tr:hover { background: #FFF7F9; }
        table.diversity-table th.grp-orig, table.diversity-table th.grp-div {
            text-align: center;
            font-weight: 700;
        }
        table.diversity-table th.grp-orig { color: #6B7280; }
        table.diversity-table th.grp-div { color: #6D28D9; }
        table.diversity-table th:nth-child(5),
        table.diversity-table td:nth-child(5) {
            border-left: 2px solid #E5E7EB;
        }
        table.diversity-table td.title-cell.changed {
            color: #6D28D9;
            position: relative;
        }
        table.diversity-table td.title-cell.changed::before {
            content: "";
            position: absolute;
            left: 0;
            top: 8px;
            bottom: 8px;
            width: 3px;
            border-radius: 999px;
            background: #7B5EA7;
        }
        .blend-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 2px 9px;
            font-size: 0.75rem;
            font-weight: 700;
        }
        .blend-badge.blend-content { background: #EFF6FF; color: #1D4ED8; }
        .blend-badge.blend-cf { background: #ECFDF5; color: #047857; }
        .blend-badge.blend-balanced { background: #F5F3FF; color: #6D28D9; }
        table.why-table td.score-cell { font-weight: 800; }
        table.why-table td.score-cell.content { color: #1D4ED8; }
        table.why-table td.score-cell.collab { color: #047857; }
        table.why-table td.score-cell.hybrid { color: #C2415C; }
        .rank-board {
            display: grid;
            gap: 14px;
            margin: 0.35rem 0 0.85rem;
        }
        .rank-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 16px 16px 14px;
            position: relative;
            overflow: hidden;
            min-height: 100%;
        }
        .rank-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
        }
        .rank-card.place-1 {
            border-color: rgba(232, 93, 117, 0.5);
            box-shadow: 0 8px 24px rgba(232, 93, 117, 0.12);
        }
        .rank-card.place-1::before { background: linear-gradient(90deg, #E85D75, #FFC857); }
        .rank-card.place-2::before { background: #8B9BB4; }
        .rank-card.place-3::before { background: #C47B5A; }
        .rank-name {
            color: #31333F;
            font-weight: 800;
            font-size: 1.18rem;
            line-height: 1.35;
            margin: 6px 0 6px;
        }
        .rank-card.place-1 .rank-name { color: #C2415C; }
        .rank-card.place-2 .rank-name { color: #4B5563; }
        .rank-card.place-3 .rank-name { color: #9A5B3C; }
        .rank-why { color: #4B5563; font-size: 0.86rem; margin-bottom: 10px; }
        .rank-chips { display: flex; flex-wrap: wrap; gap: 5px; min-height: 1.5rem; }
        .rank-chip {
            background: rgba(16, 185, 129, 0.12);
            color: #059669;
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 0.68rem;
            font-weight: 700;
        }
        .rank-chip.chip-rmse { background: #EFF6FF; color: #1D4ED8; }
        .rank-chip.chip-precision { background: #ECFDF5; color: #047857; }
        .rank-chip.chip-recall { background: #F5F3FF; color: #6D28D9; }
        .rank-chip.chip-f1-score { background: #FFF1F4; color: #C2415C; }
        .rank-chip.chip-accuracy { background: #FFF8E8; color: #B58105; }
        .rank-chip.empty { background: #EEF0F4; color: #6B7280; font-weight: 600; }
        .rank-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 12px;
        }
        .rank-stat {
            background: #F0F2F6;
            border-radius: 10px;
            padding: 8px 10px;
        }
        .rank-stat .lbl {
            display: block;
            font-size: 0.68rem;
            color: #6B7280;
            margin-bottom: 2px;
        }
        .rank-stat .val {
            font-size: 1.02rem;
            font-weight: 700;
            color: #31333F;
        }
        .rank-stat.best .val { color: #059669; }
        .rank-stat.best .lbl { color: #059669; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🎬 Movie Recommendation System")
    st.html('<div class="hero-bar"></div>')

    data_sig = processed_data_signature()

    try:
        with st.spinner("Training models (content + collaborative + hybrid)^_^"):
            (
                hybrid_model,
                content_model,
                collaborative_model,
                all_metrics,
                all_eval,
                tuning_results,
                ratings,
                movies,
                _movie_content,
                links,
                posters,
                user_ids,
                train_size,
                test_size,
                best_alpha,
            ) = train_models(data_sig)
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Run `py data_preprocessing.py --output-dir processed` first.")
        return

    # Cached instances can keep an older class after a module reload.
    hybrid_model.__class__ = hybrid.HybridRecommender
    hybrid_model.content_model.__class__ = content_based.ContentBasedRecommender
    content_model.__class__ = content_based.ContentBasedRecommender
    collaborative_model.__class__ = collaborative_filtering.CollaborativeFiltering
    if getattr(hybrid_model.content_model, "movies", None) is None:
        hybrid_model.content_model.movies = getattr(hybrid_model, "movies", None)
    if getattr(content_model, "movies", None) is None:
        content_model.movies = movies

    rating_stats = movie_rating_stats(ratings)

    with st.sidebar:
        st.header("Settings")
        user_id = st.selectbox(
            "User ID",
            user_ids,
            index=user_ids.index(1) if 1 in user_ids else 0,
        )
        top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)
        section = st.session_state.get("main_tab", MAIN_TABS[0])
        diversify = False
        diversity_strength = 0.3
        alpha = float(st.session_state.get("hybrid_alpha", best_alpha))
        cf_variant = collaborative_filtering.VARIANT_USER
        cf_k = collaborative_filtering.DEFAULT_NEIGHBORHOOD_K
        cf_genres: list[str] = []
        cf_item_method = collaborative_filtering.DEFAULT_ITEM_METHOD
        cf_n_components = collaborative_filtering.DEFAULT_N_COMPONENTS
        if section == "🔀 Hybrid":
            alpha_token = (data_sig, round(float(best_alpha), 4))
            if st.session_state.get("_alpha_token") != alpha_token:
                st.session_state.hybrid_alpha = float(best_alpha)
                st.session_state._alpha_token = alpha_token
            alpha = st.slider(
                "Content vs collaborative weight (alpha)",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="hybrid_alpha",
                help="Maximum content weight. Cold movies use more content; popular movies "
                "stay closer to SVD.",
            )
            st.caption(f"Tuned alpha = {best_alpha:.2f}.")
        if section == "🎬 Content-based":
            diversify, diversity_strength = content_based.render_diversity_controls(enabled=True)
        if section == "👥 Collaborative":
            cf_variant, cf_genres, cf_k, cf_item_method, cf_n_components = (
                collaborative_filtering.render_controls(movies)
            )
        if st.button("Clear model cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    render_home(
        ratings=ratings,
        movies=movies,
        user_id=int(user_id),
        data_sig=data_sig,
        all_metrics=all_metrics,
        best_alpha=float(best_alpha),
        tuning_results=tuning_results,
        links=links,
        posters=posters,
        top_n=int(top_n),
        alpha=float(alpha),
        diversify=bool(diversify),
        diversity=float(diversity_strength),
        hybrid_model=hybrid_model,
        content_model=content_model,
        collaborative_model=collaborative_model,
        rating_stats=rating_stats,
        all_eval=all_eval,
        train_size=train_size,
        test_size=test_size,
        cf_variant=cf_variant,
        cf_k=int(cf_k),
        cf_genres=cf_genres,
        cf_item_method=str(cf_item_method),
        cf_n_components=int(cf_n_components),
    )



if __name__ == "__main__":
    main()
