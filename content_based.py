"""Content-based movie recommender using TF-IDF vectors and cosine similarity.

Standalone module used by the Streamlit app for the Content-based algorithm path.
Hybrid keeps its own content implementation and does not import this file.

Evaluation matches collaborative_filtering.py: RMSE/MSE, liked/not-liked
classification (precision/recall/F1/accuracy), plus raw arrays for plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from data_preprocessing import CONTENT_VECTORIZER_PARAMS, build_movie_content

BASE_DIR = Path(__file__).resolve().parent
MIN_RATING = 0.5
MAX_RATING = 5.0
CONTENT_NEIGHBOR_K = 40
CONTENT_SHRINKAGE = 12.0
# Key used by app.py ALGORITHM_OPTIONS. Sidebar diversity is only active
# when this algorithm is selected — hybrid does not use these controls.
ALGORITHM_KEY = "content"

# Kept consistent everywhere: genre chips, why-table, and the profile chart.
GENRE_COLORS = {
    "Action": "#2EC4B6",
    "Adventure": "#7BD389",
    "Animation": "#F4D35E",
    "Children": "#90BE6D",
    "Comedy": "#FFB703",
    "Crime": "#2A9D8F",
    "Documentary": "#4CC9F0",
    "Drama": "#4C9AFF",
    "Fantasy": "#7B5EA7",
    "Film-Noir": "#4A4E69",
    "Horror": "#9B2226",
    "IMAX": "#48CAE4",
    "Musical": "#E5989B",
    "Mystery": "#577590",
    "Romance": "#F4A261",
    "Sci-Fi": "#E76F51",
    "Thriller": "#E63946",
    "War": "#52B788",
    "Western": "#BC6C25",
}
DEFAULT_GENRE_COLOR = "#6B7280"


def _clip_rating(value: float) -> float:
    return float(np.clip(value, MIN_RATING, MAX_RATING))


def _find_processed_dir() -> Path:
    for candidate in (BASE_DIR / "processed", BASE_DIR / "dataset" / "processed"):
        if (candidate / "ratings_clean.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Processed data not found. Run `py data_preprocessing.py` first."
    )


def load_data(
    processed_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = processed_dir or _find_processed_dir()
    ratings = pd.read_csv(processed_dir / "ratings_clean.csv")
    movies = pd.read_csv(processed_dir / "movies_clean.csv")

    movie_content_path = processed_dir / "movies_content.csv"
    if movie_content_path.exists():
        movie_content = pd.read_csv(movie_content_path)
    else:
        tags = pd.read_csv(processed_dir / "tags_clean.csv")
        movie_content = build_movie_content(movies, tags)

    return ratings, movies, movie_content


def split_train_test(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user 80/20 split so every user with 2+ ratings appears in both sets."""
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, user_ratings in ratings.groupby("userId"):
        if len(user_ratings) < 2:
            train_parts.append(user_ratings)
            continue
        train_split, test_split = train_test_split(
            user_ratings,
            test_size=test_size,
            random_state=random_state,
        )
        train_parts.append(train_split)
        test_parts.append(test_split)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return train_df, test_df


def split_movie_genres(value: object) -> list[str]:
    """Split a MovieLens genres cell into clean labels."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        parts = [str(item).strip() for item in value]
    else:
        text = str(value)
        parts = text.split("|")
    return [part.strip() for part in parts if part.strip() and part.strip() != "(no genres listed)"]


def catalog_genres(movies: pd.DataFrame) -> list[str]:
    """Sorted unique genres in the catalog, used by the More Like This picker."""
    labels: set[str] = set()
    for value in movies["genres"].dropna():
        labels.update(split_movie_genres(value))
    return sorted(labels)


def user_genre_profile(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    min_rating: float = 3.0,
) -> pd.DataFrame:
    """Share of a user's taste per genre.

    Ratings below ``min_rating`` are ignored. Each kept rating is split evenly
    across that movie's genres, then shares are normalised to 100%.
    """
    user = ratings.loc[
        (ratings["userId"] == int(user_id)) & (ratings["rating"] >= min_rating)
    ].copy()
    if user.empty:
        return pd.DataFrame(columns=["genre", "share", "percent", "color"])

    merged = user.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    weights: dict[str, float] = {}
    for row in merged.itertuples():
        genres = split_movie_genres(getattr(row, "genres", None))
        if not genres:
            continue
        piece = float(row.rating) / len(genres)
        for genre in genres:
            weights[genre] = weights.get(genre, 0.0) + piece

    total = sum(weights.values())
    if total <= 0:
        return pd.DataFrame(columns=["genre", "share", "percent", "color"])

    rows = [
        {
            "genre": genre,
            "share": weight / total,
            "percent": 100.0 * weight / total,
            "color": GENRE_COLORS.get(genre, DEFAULT_GENRE_COLOR),
        }
        for genre, weight in weights.items()
    ]
    return (
        pd.DataFrame(rows).sort_values("percent", ascending=False).reset_index(drop=True)
    )


def plot_genre_profile(
    profile: pd.DataFrame, user_id: int, top_n: int = 10
):
    """Horizontal bar chart of a user's genre share (same colours as the pills)."""
    import matplotlib.pyplot as plt

    data = profile.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    if data.empty:
        ax.set_title(f"User {user_id}'s Genre Preferences", color="#31333F")
        ax.text(0.5, 0.5, "Not enough liked ratings", ha="center", color="#6B7280")
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    bars = ax.barh(
        data["genre"],
        data["percent"],
        color=data["color"],
        height=0.66,
    )
    ax.set_title(f"User {user_id}'s Genre Preferences", color="#31333F", pad=12)
    xmax = max(25.0, float(data["percent"].max()) * 1.18)
    ax.set_xlim(0, xmax)
    ax.tick_params(colors="#31333F", labelsize=10)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("")
    for bar, percent in zip(bars, data["percent"], strict=True):
        ax.text(
            bar.get_width() + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{percent:.0f}%",
            va="center",
            color="#31333F",
            fontsize=9,
        )
    fig.tight_layout()
    return fig


def genre_pill_html(genre: str) -> str:
    from html import escape

    color = GENRE_COLORS.get(genre, DEFAULT_GENRE_COLOR)
    return (
        f'<span class="genre-pill" style="background:{color}">{escape(genre)}</span>'
    )


def why_recommended_table_html(
    frame: pd.DataFrame,
    reasons: dict[int, list[str]],
    empty_why: str = "Matches your profile",
) -> str:
    """Pretty content-based table: genre pills, % bars, and why-recommended notes."""
    from html import escape

    rows_html: list[str] = []
    for number, row in enumerate(frame.itertuples(), start=1):
        movie_id = int(row.movieId)
        title = escape(str(getattr(row, "title", "Unknown title")))
        pills = "".join(genre_pill_html(genre) for genre in split_movie_genres(getattr(row, "genres", "")))
        similarity = getattr(row, "similarity", None)
        if similarity is not None and not pd.isna(similarity):
            for_you = int(round(float(np.clip(similarity, 0.0, 1.0)) * 100))
        else:
            score = None
            for name in ("Score", "predicted_rating", "Hybrid_score", "hybrid_rating"):
                if hasattr(row, name):
                    value = getattr(row, name)
                    if value is not None and not pd.isna(value):
                        score = value
                        break
            for_you = (
                int(round(float(score) / MAX_RATING * 100)) if score is not None else 0
            )
        avg = getattr(row, "avg_rating", None)
        avg_pct = int(round(float(avg) / MAX_RATING * 100)) if avg is not None and pd.notna(avg) else 0
        why_bits = reasons.get(movie_id, [])
        why = "".join(
            f'<div class="why-line">✓ {escape(item)}</div>' for item in why_bits
        ) or f'<span class="why-empty">{escape(empty_why)}</span>'
        rows_html.append(
            "<tr>"
            f"<td class='num'>{number}</td>"
            f"<td class='title-cell'>{title}</td>"
            f"<td class='genre-cell'>{pills}</td>"
            f"<td class='pct-cell'><span>{for_you}%</span>"
            f"<div class='bar'><div class='fill fill-you' style='width:{for_you}%'></div></div></td>"
            f"<td class='pct-cell'><span>{avg_pct}%</span>"
            f"<div class='bar'><div class='fill fill-avg' style='width:{avg_pct}%'></div></div></td>"
            f"<td class='why-cell'>{why}</td>"
            "</tr>"
        )

    return (
        "<table class='why-table'>"
        "<thead><tr>"
        "<th>No.</th><th>Movie</th><th>Genres</th>"
        "<th class='you'>Rating (for you)</th>"
        "<th class='avg'>Average Rating</th>"
        "<th>Why recommended?</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


class ContentBasedRecommender:
    """
    Content-based recommender: TF-IDF over genres/tags + cosine similarity.

    Optional MMR diversification for Top-N lists. Evaluation API matches
    collaborative_filtering.CollaborativeFiltering.evaluate().
    """

    RATING_MIN = MIN_RATING
    RATING_MAX = MAX_RATING

    def __init__(self) -> None:
        self.movie_ids: np.ndarray | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.movie_id_to_index: dict[int, int] = {}
        self.user_ratings: dict[int, dict[int, float]] = {}
        self.user_profiles: dict[int, np.ndarray] = {}
        self.user_rated_indices: dict[int, np.ndarray] = {}
        self.user_rated_values: dict[int, np.ndarray] = {}
        self.user_means: dict[int, float] = {}
        self.global_mean: float = 3.5
        self.movies: pd.DataFrame | None = None
        self.movie_content: pd.DataFrame | None = None
        self.neighbor_k = CONTENT_NEIGHBOR_K
        self.shrinkage = CONTENT_SHRINKAGE
        self.vectorizer = TfidfVectorizer(**CONTENT_VECTORIZER_PARAMS)

    def fit(
        self,
        ratings: pd.DataFrame,
        movie_content: pd.DataFrame,
        movies: pd.DataFrame | None = None,
    ) -> "ContentBasedRecommender":
        content = movie_content.copy()
        content["content_features"] = content["content_features"].fillna("").astype(str)

        self.movie_content = content
        self.movie_ids = content["movieId"].to_numpy()
        tfidf_sparse = self.vectorizer.fit_transform(content["content_features"])
        # L2-normalize so cosine similarity == dot product.
        self.tfidf_matrix = normalize(tfidf_sparse, norm="l2").toarray()

        self.movie_id_to_index = {
            int(movie_id): index for index, movie_id in enumerate(self.movie_ids)
        }
        self.global_mean = float(ratings["rating"].mean())
        self.user_ratings = {
            int(user_id): {
                int(row.movieId): float(row.rating) for row in user_frame.itertuples()
            }
            for user_id, user_frame in ratings.groupby("userId")
        }
        self.movies = movies
        self._build_user_indexes()
        return self

    def _build_user_indexes(self) -> None:
        if self.tfidf_matrix is None:
            return

        self.user_profiles = {}
        self.user_rated_indices = {}
        self.user_rated_values = {}
        self.user_means = {}

        for user_id, rated_movies in self.user_ratings.items():
            indices = []
            values = []
            for movie_id, rating in rated_movies.items():
                index = self.movie_id_to_index.get(movie_id)
                if index is not None:
                    indices.append(index)
                    values.append(rating)
            if not indices:
                continue

            index_array = np.asarray(indices, dtype=int)
            value_array = np.asarray(values, dtype=float)
            profile = np.average(self.tfidf_matrix[index_array], axis=0, weights=value_array)
            self.user_profiles[user_id] = normalize(profile.reshape(1, -1), norm="l2").ravel()
            self.user_rated_indices[user_id] = index_array
            self.user_rated_values[user_id] = value_array
            self.user_means[user_id] = float(value_array.mean())

    def _user_profile(self, user_id: int) -> np.ndarray | None:
        return self.user_profiles.get(user_id)

    def similarity_to_profile(self, user_id: int, movie_id: int) -> float:
        """Cosine similarity between the user TF-IDF profile and one movie."""
        profile = self._user_profile(user_id)
        movie_index = self.movie_id_to_index.get(int(movie_id))
        if profile is None or movie_index is None or self.tfidf_matrix is None:
            return 0.0
        similarity = float(
            cosine_similarity(
                profile.reshape(1, -1),
                self.tfidf_matrix[movie_index].reshape(1, -1),
            )[0, 0]
        )
        return float(np.clip(similarity, 0.0, 1.0))

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating via cosine-weighted kNN of the user's rated movies."""
        if self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")

        movie_index = self.movie_id_to_index.get(movie_id)
        rated_indices = self.user_rated_indices.get(user_id)
        rated_values = self.user_rated_values.get(user_id)
        user_mean = self.user_means.get(user_id, self.global_mean)
        if movie_index is None or rated_indices is None or rated_values is None:
            return user_mean

        similarities = np.clip(
            self.tfidf_matrix[rated_indices] @ self.tfidf_matrix[movie_index],
            0.0,
            None,
        )
        k = min(self.neighbor_k, similarities.size)
        if k <= 0:
            return user_mean
        if similarities.size > k:
            top = np.argpartition(similarities, -k)[-k:]
            similarities = similarities[top]
            rated_values = rated_values[top]
        mass = float(similarities.sum())
        if mass <= 1e-9:
            return user_mean
        neighbor = float(np.dot(similarities, rated_values) / mass)
        predicted = (mass * neighbor + self.shrinkage * user_mean) / (
            mass + self.shrinkage
        )
        return _clip_rating(predicted)

    def top_candidates(
        self,
        user_id: int,
        n_candidates: int,
        diversify: bool = False,
        diversity: float = 0.3,
    ) -> list[int]:
        if self.movie_ids is None or self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")

        profile = self._user_profile(user_id)
        if profile is None:
            return []

        rated_movies = set(self.user_ratings.get(user_id, {}))
        candidate_indices = [
            index
            for index, movie_id in enumerate(self.movie_ids)
            if int(movie_id) not in rated_movies
        ]
        if not candidate_indices:
            return []

        scores = cosine_similarity(
            profile.reshape(1, -1),
            self.tfidf_matrix[candidate_indices],
        ).flatten()

        pool_size = (
            int(n_candidates * (4 + 6 * float(diversity))) if diversify else n_candidates
        )
        pool_size = min(max(pool_size, n_candidates), len(candidate_indices))
        top_local = np.argsort(scores)[-pool_size:][::-1]
        pool_indices = [candidate_indices[i] for i in top_local]
        pool_scores = scores[top_local]
        pool_ids = [int(self.movie_ids[i]) for i in pool_indices]

        if not diversify or len(pool_ids) <= n_candidates:
            return pool_ids[:n_candidates]

        return self._mmr_diversify(
            pool_ids, pool_indices, pool_scores, n_candidates, float(diversity)
        )

    def _mmr_diversify(
        self,
        candidate_ids: list[int],
        candidate_indices: list[int],
        candidate_scores: np.ndarray,
        top_n: int,
        diversity: float,
    ) -> list[int]:
        """Greedy Maximal Marginal Relevance re-ranking over TF-IDF vectors."""
        if self.tfidf_matrix is None:
            return candidate_ids[:top_n]

        scores = np.asarray(candidate_scores, dtype=float)
        if float(scores.max()) > float(scores.min()):
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            norm_scores = np.ones_like(scores)

        remaining = list(range(len(candidate_ids)))
        selected: list[int] = []
        while remaining and len(selected) < top_n:
            if not selected:
                pick = max(remaining, key=lambda i: float(norm_scores[i]))
            else:

                def mmr(i: int) -> float:
                    redundancy = max(
                        float(
                            cosine_similarity(
                                self.tfidf_matrix[candidate_indices[i]].reshape(1, -1),
                                self.tfidf_matrix[candidate_indices[j]].reshape(1, -1),
                            )[0, 0]
                        )
                        for j in selected
                    )
                    return (1.0 - diversity) * float(norm_scores[i]) - diversity * redundancy

                pick = max(remaining, key=mmr)
            selected.append(pick)
            remaining.remove(pick)

        return [candidate_ids[i] for i in selected]

    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        diversify: bool = False,
        diversity: float = 0.3,
    ) -> pd.DataFrame:
        """Return top-N unseen movie recommendations for a user."""
        movie_ids = self.top_candidates(
            user_id,
            n_recommendations,
            diversify=diversify,
            diversity=diversity,
        )
        if not movie_ids:
            raise ValueError(f"No usable ratings found for user {user_id}.")

        recommendations = pd.DataFrame(
            {
                "movieId": movie_ids,
                "predicted_rating": [
                    self.predict(user_id, movie_id) for movie_id in movie_ids
                ],
                "similarity": [
                    self.similarity_to_profile(user_id, movie_id) for movie_id in movie_ids
                ],
            }
        )
        if self.movies is not None:
            recommendations = recommendations.merge(
                self.movies[["movieId", "title", "genres"]],
                on="movieId",
                how="left",
            )
        return recommendations.reset_index(drop=True)

    def similar_to_genre(
        self,
        genre: str | list[str] | tuple[str, ...],
        n_recommendations: int = 10,
        user_id: int | None = None,
    ) -> pd.DataFrame:
        """Rank unseen movies by cosine similarity to the chosen genre's TF-IDF centroid."""
        if getattr(self, "movies", None) is None or self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")

        if isinstance(genre, str):
            wanted = [genre.strip()]
        else:
            wanted = [str(item).strip() for item in genre if str(item).strip()]
        wanted = [item for item in wanted if item]
        if not wanted:
            raise ValueError("Pick at least one genre.")
        label = " + ".join(wanted)

        def matches(genres_value: object, require_all: bool) -> bool:
            movie_genres = set(split_movie_genres(genres_value))
            if require_all:
                return all(name in movie_genres for name in wanted)
            return any(name in movie_genres for name in wanted)

        seed_ids = [
            int(movie_id)
            for movie_id, genres in zip(
                self.movies["movieId"], self.movies["genres"], strict=False
            )
            if matches(genres, require_all=True)
        ]
        if not seed_ids and len(wanted) > 1:
            seed_ids = [
                int(movie_id)
                for movie_id, genres in zip(
                    self.movies["movieId"], self.movies["genres"], strict=False
                )
                if matches(genres, require_all=False)
            ]
        seed_indices = [
            self.movie_id_to_index[movie_id]
            for movie_id in seed_ids
            if movie_id in self.movie_id_to_index
        ]
        if not seed_indices:
            raise ValueError(f"No movies found for genre '{label}'.")

        centroid = self.tfidf_matrix[seed_indices].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-12:
            centroid = centroid / norm

        scores = self.tfidf_matrix @ centroid
        rated = set(self.user_ratings.get(int(user_id), {})) if user_id is not None else set()
        ranked = np.argsort(scores)[::-1]
        picked: list[int] = []
        picked_scores: list[float] = []
        for index in ranked:
            movie_id = int(self.movie_ids[index])
            if movie_id in rated:
                continue
            picked.append(movie_id)
            picked_scores.append(float(np.clip(scores[index], 0.0, 1.0)))
            if len(picked) >= n_recommendations:
                break
        if not picked:
            raise ValueError(f"No unseen movies found for genre '{label}'.")

        frame = pd.DataFrame(
            {
                "movieId": picked,
                "similarity": picked_scores,
                "predicted_rating": [
                    self.predict(int(user_id), movie_id) if user_id is not None else np.nan
                    for movie_id in picked
                ],
            }
        )
        return frame.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left",
        ).reset_index(drop=True)

    def recommendation_reasons(self, user_id: int, movie_id: int) -> list[str]:
        """Short why-recommended notes: shared liked genres + nearest 4★+ neighbour."""
        if getattr(self, "movies", None) is None or self.tfidf_matrix is None:
            return []

        movie_row = self.movies.loc[self.movies["movieId"] == int(movie_id)]
        if movie_row.empty:
            return []
        rec_genres = split_movie_genres(movie_row.iloc[0]["genres"])

        liked = [
            (mid, rating)
            for mid, rating in self.user_ratings.get(int(user_id), {}).items()
            if rating >= 4.0
        ]
        liked_genres: set[str] = set()
        for mid, _rating in liked:
            liked_row = self.movies.loc[self.movies["movieId"] == int(mid)]
            if liked_row.empty:
                continue
            liked_genres.update(split_movie_genres(liked_row.iloc[0]["genres"]))

        reasons = [genre for genre in rec_genres if genre in liked_genres][:3]

        rec_index = self.movie_id_to_index.get(int(movie_id))
        best: tuple[float, int, float] | None = None
        if rec_index is not None:
            for mid, rating in liked:
                other_index = self.movie_id_to_index.get(int(mid))
                if other_index is None:
                    continue
                similarity = float(self.tfidf_matrix[rec_index] @ self.tfidf_matrix[other_index])
                if best is None or similarity > best[0]:
                    best = (similarity, int(mid), float(rating))
        if best is not None and best[0] >= 0.08:
            neighbour = self.movies.loc[self.movies["movieId"] == best[1]]
            if not neighbour.empty:
                title = str(neighbour.iloc[0]["title"])
                reasons.append(
                    f"Similar to '{title}' (you rated it {best[2]:.0f}★)"
                )
        return reasons

    # ------------------------------------------------------------------
    # Evaluation (same protocol as collaborative_filtering.py)
    # ------------------------------------------------------------------
    def score_test_ratings(
        self, test_ratings: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        if test_ratings.empty:
            return np.array([]), np.array([])
        actual = np.empty(len(test_ratings), dtype=float)
        predicted = np.empty(len(test_ratings), dtype=float)
        for index, row in enumerate(test_ratings.itertuples()):
            actual[index] = float(row.rating)
            predicted[index] = self.predict(int(row.userId), int(row.movieId))
        return actual, predicted

    @staticmethod
    def _classification_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        relevance_threshold: float,
        decision_threshold: float,
    ) -> dict[str, float]:
        actual_liked = (actual >= relevance_threshold).astype(int)
        predicted_liked = (predicted >= decision_threshold).astype(int)
        return {
            "precision": float(precision_score(actual_liked, predicted_liked, zero_division=0)),
            "recall": float(recall_score(actual_liked, predicted_liked, zero_division=0)),
            "f1_score": float(f1_score(actual_liked, predicted_liked, zero_division=0)),
            "accuracy": float(accuracy_score(actual_liked, predicted_liked)),
            "decision_threshold": float(decision_threshold),
        }

    def evaluate_predictions(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        relevance_threshold: float = 4.0,
        decision_threshold: float | None = None,
    ) -> dict[str, float]:
        if actual.size == 0:
            return {
                "rmse": float("nan"),
                "mse": float("nan"),
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "decision_threshold": float(relevance_threshold),
                "n_ratings_evaluated": 0,
            }

        mse = float(mean_squared_error(actual, predicted))
        metrics: dict[str, float] = {
            "rmse": float(np.sqrt(mse)),
            "mse": mse,
            "n_ratings_evaluated": float(actual.size),
        }
        thresholds = np.round(np.arange(2.5, 4.25, 0.05), 2)
        if decision_threshold is None:
            best: dict[str, float] | None = None
            for threshold in thresholds:
                candidate = self._classification_metrics(
                    actual, predicted, relevance_threshold, float(threshold)
                )
                if best is None or candidate["f1_score"] > best["f1_score"]:
                    best = candidate
            if best is None:
                raise RuntimeError("Unable to compute classification metrics.")
            metrics.update(best)
        else:
            metrics.update(
                self._classification_metrics(
                    actual, predicted, relevance_threshold, decision_threshold
                )
            )
        return metrics

    def evaluate(
        self,
        test_ratings: pd.DataFrame,
        relevance_threshold: float = 4.0,
    ) -> dict[str, float | np.ndarray]:
        """RMSE + liked/not-liked metrics plus raw arrays for plots."""
        actual, predicted = self.score_test_ratings(test_ratings)
        metrics = self.evaluate_predictions(
            actual, predicted, relevance_threshold=relevance_threshold
        )
        decision = float(metrics.get("decision_threshold", relevance_threshold))
        return {
            **metrics,
            "actual": actual,
            "predicted": predicted,
            "y_true": (actual >= relevance_threshold).astype(int) if actual.size else np.array([]),
            "y_pred": (predicted >= decision).astype(int) if predicted.size else np.array([]),
        }

    def evaluate_precision_recall_at_k(
        self,
        train_ratings: pd.DataFrame,
        test_ratings: pd.DataFrame,
        k: int = 10,
        threshold: float = 4.0,
        max_users: int = 150,
    ) -> dict[str, float]:
        """
        Optional Top-K ranking metrics (not part of evaluate(), same as CF).

        For each test user with relevant held-out movies (rating >= threshold),
        recommend Top-K among movies not seen in train and measure overlap.
        """
        train_users = set(train_ratings["userId"].astype(int).unique())
        eval_users = [
            int(user_id)
            for user_id in test_ratings["userId"].unique()
            if int(user_id) in train_users and int(user_id) in self.user_profiles
        ]
        if len(eval_users) > max_users:
            rng = np.random.default_rng(42)
            eval_users = list(rng.choice(eval_users, size=max_users, replace=False))

        precisions: list[float] = []
        recalls: list[float] = []
        for user_id in eval_users:
            relevant = set(
                test_ratings.loc[
                    (test_ratings["userId"] == user_id)
                    & (test_ratings["rating"] >= threshold),
                    "movieId",
                ]
                .astype(int)
                .tolist()
            )
            if not relevant:
                continue

            recommended = set(self.top_candidates(user_id, k))
            hits = len(recommended & relevant)
            precisions.append(hits / k)
            recalls.append(hits / len(relevant))

        precision_at_k = float(np.mean(precisions)) if precisions else 0.0
        recall_at_k = float(np.mean(recalls)) if recalls else 0.0
        f1_at_k = (
            2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
            if (precision_at_k + recall_at_k) > 0
            else 0.0
        )
        return {
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "f1_at_k": f1_at_k,
            "n_users_evaluated": float(len(precisions)),
        }


def render_diversity_controls(*, enabled: bool = True) -> tuple[bool, float]:
    """Draw MMR diversity widgets in the current Streamlit sidebar.

    Call this only when Content-based is selected so other algorithms do not
    show these controls.
    """
    import streamlit as st

    diversify = st.checkbox(
        "Diversify recommendations",
        value=False,
        disabled=not enabled,
        help=(
            "Re-ranks the top results with MMR so they aren't near-duplicates "
            "of each other (e.g. 10 Pixar sequels back to back)."
        ),
    )
    diversity_strength = st.slider(
        "Diversity strength",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        disabled=not enabled or not diversify,
        help="0 = pure relevance ranking, 1 = maximise spread over relevance.",
    )
    if not enabled:
        return False, 0.3
    return bool(diversify), float(diversity_strength)


def demo(
    user_id: int = 1,
    top_n: int = 10,
    diversify: bool = False,
    diversity: float = 0.3,
) -> pd.DataFrame:
    ratings, movies, movie_content = load_data()
    model = ContentBasedRecommender().fit(ratings, movie_content, movies=movies)
    return model.recommend(
        user_id,
        n_recommendations=top_n,
        diversify=diversify,
        diversity=diversity,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the content-based recommender.")
    parser.add_argument("--user-id", type=int, default=1, help="User id to recommend for.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations.")
    parser.add_argument(
        "--diversify",
        action="store_true",
        help="Re-rank with MMR so near-duplicate titles are less dominant.",
    )
    parser.add_argument(
        "--diversity",
        type=float,
        default=0.3,
        help="MMR diversity strength in [0, 1] (only used with --diversify).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run train/test evaluation (RMSE + liked/not-liked metrics) instead of demo recs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ratings, movies, movie_content = load_data()

    if args.evaluate:
        print("Splitting ratings 80/20 per user and evaluating content-based model...\n")
        train_ratings, test_ratings = split_train_test(ratings, test_size=0.2, random_state=42)
        model = ContentBasedRecommender().fit(train_ratings, movie_content, movies=movies)
        metrics = model.evaluate(test_ratings, relevance_threshold=4.0)
        print(f"RMSE:          {metrics['rmse']:.4f}")
        print(f"MSE:           {metrics['mse']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"F1:            {metrics['f1_score']:.4f}")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Pred. cutoff:  {metrics['decision_threshold']:.2f}")
        print(f"N ratings:     {int(metrics['n_ratings_evaluated'])}")
    else:
        print(
            demo(
                user_id=args.user_id,
                top_n=args.top_n,
                diversify=args.diversify,
                diversity=args.diversity,
            ).to_string(index=False)
        )
