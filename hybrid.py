from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
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
DEFAULT_ALPHA_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]
CLASSIFICATION_THRESHOLDS = np.round(np.arange(2.5, 4.25, 0.05), 2)
ITEM_SUPPORT_SHRINK = 25.0
MIN_CONTENT_MIX = 0.25
CONTENT_NEIGHBOR_K = 40
CONTENT_SHRINKAGE = 12.0


def _clip_rating(value: float) -> float:
    return float(np.clip(value, MIN_RATING, MAX_RATING))


def _find_processed_dir() -> Path:
    for candidate in (BASE_DIR / "processed", BASE_DIR / "dataset" / "processed"):
        if (candidate / "ratings_clean.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Processed data not found. Run `py data_preprocessing.py` first."
    )


def blend_hybrid_details(
    content_scores: np.ndarray,
    cf_scores: np.ndarray,
    movie_ids: np.ndarray,
    item_counts: dict[int, int],
    alpha: float,
    shrink: float = ITEM_SUPPORT_SHRINK,
    min_mix: float = MIN_CONTENT_MIX,
) -> dict[str, np.ndarray]:
    """Return hybrid scores plus the per-movie mix weights used to make them."""
    if len(content_scores) == 0:
        empty = np.array([], dtype=float)
        return {
            "rating_count": empty,
            "content_weight": empty,
            "cf_weight": empty,
            "hybrid_rating": empty,
        }
    counts = np.fromiter(
        (item_counts.get(int(movie_id), 0) for movie_id in movie_ids),
        dtype=float,
        count=len(movie_ids),
    )
    item_confidence = counts / (counts + shrink)
    content_weight = np.clip(
        alpha * (min_mix + (1.0 - min_mix) * (1.0 - item_confidence)),
        0.0,
        1.0,
    )
    cf_weight = 1.0 - content_weight
    hybrid_rating = np.clip(
        content_weight * content_scores + cf_weight * cf_scores,
        MIN_RATING,
        MAX_RATING,
    )
    return {
        "rating_count": counts,
        "content_weight": content_weight,
        "cf_weight": cf_weight,
        "hybrid_rating": hybrid_rating,
    }


def blend_hybrid_scores(
    content_scores: np.ndarray,
    cf_scores: np.ndarray,
    movie_ids: np.ndarray,
    item_counts: dict[int, int],
    alpha: float,
    shrink: float = ITEM_SUPPORT_SHRINK,
    min_mix: float = MIN_CONTENT_MIX,
) -> np.ndarray:
    """Support-aware weighted hybrid: more content when a movie has few ratings."""
    return blend_hybrid_details(
        content_scores,
        cf_scores,
        movie_ids,
        item_counts,
        alpha,
        shrink=shrink,
        min_mix=min_mix,
    )["hybrid_rating"]


def blend_source_label(content_weight: float) -> str:
    """Label whether content or SVD carried more of the hybrid score."""
    if content_weight >= 0.45:
        return "Content-leaning"
    if content_weight <= 0.25:
        return "Collaborative-leaning"
    return "Balanced"


def _hybrid_score_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    content = "content_rating" if "content_rating" in frame.columns else "Content score"
    collaborative = "cf_rating" if "cf_rating" in frame.columns else "Collaborative score"
    hybrid = "hybrid_rating" if "hybrid_rating" in frame.columns else "Hybrid score"
    return content, collaborative, hybrid


def explain_blend_table_html(frame: pd.DataFrame, reasons: dict[int, list[str]]) -> str:
    """Table explaining the content/SVD mix for each hybrid recommendation."""
    from html import escape

    from content_based import genre_pill_html, split_movie_genres

    content_col, cf_col, hybrid_col = _hybrid_score_columns(frame)
    rows_html: list[str] = []
    for number, (_, row) in enumerate(frame.iterrows(), start=1):
        movie_id = int(row["movieId"])
        title = escape(str(row.get("title") or "Unknown title"))
        genres = split_movie_genres(row.get("genres", ""))
        pills = "".join(genre_pill_html(genre) for genre in genres) or (
            '<span class="why-empty">No genres listed</span>'
        )
        content = float(row.get(content_col, 0.0) or 0.0)
        cf_score = float(row.get(cf_col, 0.0) or 0.0)
        hybrid_score = float(row.get(hybrid_col, 0.0) or 0.0)
        genre_set = set(genres)
        why_bits = [item for item in reasons.get(movie_id, []) if item not in genre_set]
        why = "".join(
            f'<div class="why-line">✓ {escape(item)}</div>' for item in why_bits
        ) or '<span class="why-empty">Support-aware hybrid blend</span>'
        rows_html.append(
            "<tr>"
            f"<td class='num'>{number}</td>"
            f"<td class='title-cell'>{title}</td>"
            f"<td class='genre-cell'>{pills}</td>"
            f"<td class='score-cell content'>{content:.2f}</td>"
            f"<td class='score-cell collab'>{cf_score:.2f}</td>"
            f"<td class='score-cell hybrid'>{hybrid_score:.2f}</td>"
            f"<td class='why-cell'>{why}</td>"
            "</tr>"
        )
    return (
        "<table class='why-table'>"
        "<thead><tr>"
        "<th>No.</th><th>Movie</th><th>Genres</th>"
        "<th>Content</th><th>SVD</th><th>Hybrid</th>"
        "<th>Why this mix?</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def plot_score_breakdown(frame: pd.DataFrame):
    """Grouped bars of content, SVD, and hybrid scores for the current list."""
    import matplotlib.pyplot as plt

    content_col, cf_col, hybrid_col = _hybrid_score_columns(frame)
    labels = [
        str(title)[:26] + ("…" if len(str(title)) > 26 else "")
        for title in frame["title"].tolist()
    ]
    content = frame[content_col].astype(float).to_numpy()
    collaborative = frame[cf_col].astype(float).to_numpy()
    hybrid_scores = frame[hybrid_col].astype(float).to_numpy()
    stacked = np.concatenate([content, collaborative, hybrid_scores])
    y_min = max(0.0, float(np.nanmin(stacked)) - 0.2)
    y_max = min(5.35, float(np.nanmax(stacked)) + 0.15)
    if y_max - y_min < 0.7:
        mid = (y_min + y_max) / 2
        y_min = max(0.0, mid - 0.45)
        y_max = min(5.35, mid + 0.45)

    x = list(range(len(frame)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.bar([pos - width for pos in x], content, width, label="Content", color="#5B8FF9")
    ax.bar(x, collaborative, width, label="Collaborative (SVD)", color="#5AD8A6")
    ax.bar([pos + width for pos in x], hybrid_scores, width, label="Hybrid", color="#E85D75")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel("Predicted rating (zoomed)")
    ax.set_ylim(y_min, y_max)
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#C5C9D3")
    legend = ax.legend(
        frameon=True,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        facecolor="#FFFFFF",
        edgecolor="#D0D3DA",
    )
    legend.get_frame().set_alpha(1)
    fig.tight_layout()
    return fig


def plot_blend_weights(
    alpha: float,
    shrink: float = ITEM_SUPPORT_SHRINK,
    min_mix: float = MIN_CONTENT_MIX,
    max_count: int = 250,
):
    """How content vs SVD weight changes as a movie gets more ratings."""
    import matplotlib.pyplot as plt

    counts = np.arange(0, max_count + 1, dtype=int)
    item_counts = {int(index): int(count) for index, count in enumerate(counts)}
    details = blend_hybrid_details(
        np.full(len(counts), 4.0),
        np.full(len(counts), 4.0),
        np.arange(len(counts)),
        item_counts,
        float(alpha),
        shrink=shrink,
        min_mix=min_mix,
    )
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.plot(counts, details["content_weight"], color="#5B8FF9", label="Content weight")
    ax.plot(counts, details["cf_weight"], color="#5AD8A6", label="Collaborative weight")
    ax.set_xlabel("Ratings the movie already has")
    ax.set_ylabel("Blend weight")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


class ContentBasedRecommender:
    """TF-IDF + cosine-similarity content model used only by HybridRecommender.

    Pipeline:
      1. Fit TF-IDF on each movie's title/year/genres/tags text
      2. L2-normalize vectors so dot product == cosine similarity
      3. Predict ratings with a cosine-weighted kNN of the user's rated movies
    """

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
        self.neighbor_k = CONTENT_NEIGHBOR_K
        self.shrinkage = CONTENT_SHRINKAGE
        self.vectorizer = TfidfVectorizer(**CONTENT_VECTORIZER_PARAMS)

    def fit(
        self, ratings: pd.DataFrame, movie_content: pd.DataFrame
    ) -> "ContentBasedRecommender":
        content = movie_content.copy()
        content["content_features"] = content["content_features"].fillna("").astype(str)
        self.movie_ids = content["movieId"].to_numpy()
        # TF-IDF vectors; L2-normalize so cosine similarity = dot product.
        tfidf_sparse = self.vectorizer.fit_transform(content["content_features"])
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
            indices, values = [], []
            for movie_id, rating in rated_movies.items():
                index = self.movie_id_to_index.get(movie_id)
                if index is not None:
                    indices.append(index)
                    values.append(rating)
            if not indices:
                continue
            index_array = np.asarray(indices, dtype=int)
            value_array = np.asarray(values, dtype=float)
            # Rating-weighted average of TF-IDF movie vectors → user profile.
            profile = np.average(self.tfidf_matrix[index_array], axis=0, weights=value_array)
            self.user_profiles[user_id] = normalize(profile.reshape(1, -1), norm="l2").ravel()
            self.user_rated_indices[user_id] = index_array
            self.user_rated_values[user_id] = value_array
            self.user_means[user_id] = float(value_array.mean())

    def _user_profile(self, user_id: int) -> np.ndarray | None:
        return self.user_profiles.get(user_id)

    def _cosine_to_profile(self, user_id: int, movie_id: int) -> float | None:
        """Cosine similarity between the user TF-IDF profile and one movie."""
        if self.tfidf_matrix is None:
            return None
        profile = self._user_profile(user_id)
        movie_index = self.movie_id_to_index.get(movie_id)
        if profile is None or movie_index is None:
            return None
        similarity = float(
            cosine_similarity(
                profile.reshape(1, -1),
                self.tfidf_matrix[movie_index].reshape(1, -1),
            )[0, 0]
        )
        return float(np.clip(similarity, 0.0, 1.0))

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict a rating from similar movies the user has already rated."""
        if self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")

        movie_index = self.movie_id_to_index.get(movie_id)
        rated_indices = self.user_rated_indices.get(user_id)
        rated_values = self.user_rated_values.get(user_id)
        user_mean = self.user_means.get(user_id, self.global_mean)
        if movie_index is None or rated_indices is None or rated_values is None:
            return user_mean
        return self._predict_from_neighbors(
            rated_indices, rated_values, movie_index, user_mean
        )

    def _predict_from_neighbors(
        self,
        rated_indices: np.ndarray,
        rated_values: np.ndarray,
        movie_index: int,
        user_mean: float,
    ) -> float:
        if self.tfidf_matrix is None:
            return user_mean
        similarities = np.clip(
            self.tfidf_matrix[rated_indices] @ self.tfidf_matrix[movie_index],
            0.0,
            None,
        )
        return self._aggregate_neighbors(similarities, rated_values, user_mean)

    def _aggregate_neighbors(
        self,
        similarities: np.ndarray,
        rated_values: np.ndarray,
        user_mean: float,
    ) -> float:
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

    def predict_many(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Vectorized content predictions for an evaluation split."""
        if self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")

        predictions = np.full(len(user_ids), self.global_mean, dtype=float)
        if len(user_ids) == 0:
            return predictions

        grouped = pd.DataFrame(
            {
                "row": np.arange(len(user_ids)),
                "userId": np.asarray(user_ids, dtype=int),
                "movieId": np.asarray(movie_ids, dtype=int),
            }
        )
        for user_id, group in grouped.groupby("userId"):
            rated_indices = self.user_rated_indices.get(int(user_id))
            rated_values = self.user_rated_values.get(int(user_id))
            user_mean = self.user_means.get(int(user_id), self.global_mean)
            row_indices = group["row"].to_numpy()
            predictions[row_indices] = user_mean
            if rated_indices is None or rated_values is None:
                continue

            movie_indices = np.array(
                [self.movie_id_to_index.get(int(movie_id), -1) for movie_id in group["movieId"]],
                dtype=int,
            )
            known = movie_indices >= 0
            if not np.any(known):
                continue

            similarities = np.clip(
                self.tfidf_matrix[rated_indices] @ self.tfidf_matrix[movie_indices[known]].T,
                0.0,
                None,
            )
            known_rows = row_indices[known]
            for local_index, row_index in enumerate(known_rows):
                predictions[row_index] = self._aggregate_neighbors(
                    similarities[:, local_index],
                    rated_values,
                    user_mean,
                )
        return predictions

    def top_candidates(self, user_id: int, n_candidates: int) -> list[int]:
        """Rank unseen movies by cosine similarity to the user TF-IDF profile."""
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
            profile.reshape(1, -1), self.tfidf_matrix[candidate_indices]
        ).flatten()
        top_indices = np.argsort(scores)[-n_candidates:][::-1]
        return [int(self.movie_ids[candidate_indices[index]]) for index in top_indices]


class CollaborativeFilteringRecommender:
    """Collaborative filtering with SVD matrix factorization."""

    def __init__(self, n_factors: int = 20) -> None:
        self.n_factors = n_factors
        self.user_ids: np.ndarray | None = None
        self.movie_ids: np.ndarray | None = None
        self.user_id_to_index: dict[int, int] = {}
        self.movie_id_to_index: dict[int, int] = {}
        self.user_factors: np.ndarray | None = None
        self.movie_factors: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.user_means: np.ndarray | None = None
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_train_movies: dict[int, set[int]] = {}
        self.item_counts: dict[int, int] = {}
        self.global_mean: float = 3.5

    def fit(self, ratings: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        self.user_ids = np.sort(ratings["userId"].astype(int).unique())
        self.movie_ids = np.sort(ratings["movieId"].astype(int).unique())
        self.user_id_to_index = {
            int(user_id): index for index, user_id in enumerate(self.user_ids)
        }
        self.movie_id_to_index = {
            int(movie_id): index for index, movie_id in enumerate(self.movie_ids)
        }

        row_indices = ratings["userId"].map(self.user_id_to_index).to_numpy()
        col_indices = ratings["movieId"].map(self.movie_id_to_index).to_numpy()
        values = ratings["rating"].to_numpy(dtype=float)
        rating_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )

        rating_count = np.diff(rating_matrix.indptr)
        rating_sum = np.asarray(rating_matrix.sum(axis=1)).ravel()
        self.user_means = np.divide(
            rating_sum,
            rating_count,
            out=np.full(len(self.user_ids), ratings["rating"].mean()),
            where=rating_count > 0,
        )
        self.global_mean = float(ratings["rating"].mean())
        item_count = np.asarray(rating_matrix.getnnz(axis=0), dtype=float)
        item_sum = np.asarray(rating_matrix.sum(axis=0)).ravel()
        item_means = np.divide(
            item_sum,
            item_count,
            out=np.full(len(self.movie_ids), self.global_mean),
            where=item_count > 0,
        )
        self.user_bias = self.user_means - self.global_mean
        self.item_bias = item_means - self.global_mean
        self.item_counts = {
            int(movie_id): int(count)
            for movie_id, count in zip(self.movie_ids, item_count)
        }

        centered = rating_matrix.copy().astype(float)
        for user_index in range(len(self.user_ids)):
            start, end = rating_matrix.indptr[user_index], rating_matrix.indptr[user_index + 1]
            if start < end:
                movie_indices = rating_matrix.indices[start:end]
                centered.data[start:end] -= (
                    self.user_means[user_index] + self.item_bias[movie_indices]
                )

        n_users, n_movies = centered.shape
        k = min(self.n_factors, min(n_users, n_movies) - 1)
        if k < 1:
            raise ValueError("Not enough data to run SVD.")

        user_factors, singular_values, movie_factors_t = svds(centered, k=k)
        order = np.argsort(singular_values)[::-1]
        self.user_factors = user_factors[:, order]
        self.singular_values = singular_values[order]
        self.movie_factors = movie_factors_t[order].T

        self.user_train_movies = {}
        for user_id, user_frame in ratings.groupby("userId"):
            self.user_train_movies[int(user_id)] = set(user_frame["movieId"].astype(int))
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if (
            self.user_factors is None
            or self.movie_factors is None
            or self.singular_values is None
            or self.user_means is None
            or self.item_bias is None
        ):
            raise RuntimeError("Collaborative model is not fitted.")

        user_index = self.user_id_to_index.get(user_id)
        movie_index = self.movie_id_to_index.get(movie_id)
        if user_index is None or movie_index is None:
            return self.global_mean

        latent_score = np.dot(
            self.user_factors[user_index] * self.singular_values,
            self.movie_factors[movie_index],
        )
        return _clip_rating(
            float(self.user_means[user_index] + self.item_bias[movie_index] + latent_score)
        )

    def predict_many(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Vectorized collaborative predictions for an evaluation split."""
        if (
            self.user_factors is None
            or self.movie_factors is None
            or self.singular_values is None
            or self.user_means is None
            or self.item_bias is None
        ):
            raise RuntimeError("Collaborative model is not fitted.")

        n_rows = len(user_ids)
        predictions = np.full(n_rows, self.global_mean, dtype=float)
        user_index = np.fromiter(
            (self.user_id_to_index.get(int(user_id), -1) for user_id in user_ids),
            dtype=int,
            count=n_rows,
        )
        movie_index = np.fromiter(
            (self.movie_id_to_index.get(int(movie_id), -1) for movie_id in movie_ids),
            dtype=int,
            count=n_rows,
        )
        valid = (user_index >= 0) & (movie_index >= 0)
        if not np.any(valid):
            return predictions

        valid_users = user_index[valid]
        valid_movies = movie_index[valid]
        latent = np.einsum(
            "ij,ij->i",
            self.user_factors[valid_users] * self.singular_values,
            self.movie_factors[valid_movies],
        )
        predictions[valid] = np.clip(
            self.user_means[valid_users] + self.item_bias[valid_movies] + latent,
            MIN_RATING,
            MAX_RATING,
        )
        return predictions

    def top_candidates(self, user_id: int, n_candidates: int) -> list[int]:
        if (
            self.user_factors is None
            or self.movie_factors is None
            or self.singular_values is None
            or self.user_means is None
            or self.item_bias is None
            or self.movie_ids is None
        ):
            raise RuntimeError("Collaborative model is not fitted.")

        user_index = self.user_id_to_index.get(user_id)
        if user_index is None:
            return []

        user_vector = self.user_factors[user_index] * self.singular_values
        predicted_scores = (
            self.user_means[user_index] + self.item_bias + self.movie_factors @ user_vector
        )

        rated_movies = self.user_train_movies.get(user_id, set())
        candidate_indices = [
            index
            for index, movie_id in enumerate(self.movie_ids)
            if int(movie_id) not in rated_movies
        ]
        candidate_scores = predicted_scores[candidate_indices]
        top_indices = np.argsort(candidate_scores)[-n_candidates:][::-1]
        return [int(self.movie_ids[candidate_indices[index]]) for index in top_indices]


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
    """Split ratings 80/20 per user so every user appears in both sets when possible."""
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


class HybridRecommender:
    """Hybrid of TF-IDF/cosine content-based filtering and SVD collaborative filtering."""

    def __init__(self, alpha: float = 0.3, n_factors: int = 20) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.content_model = ContentBasedRecommender()
        self.collaborative_model = CollaborativeFilteringRecommender(n_factors=n_factors)
        self.movies: pd.DataFrame | None = None
        self.item_counts: dict[int, int] = {}
        self.item_shrink = ITEM_SUPPORT_SHRINK
        self.min_content_mix = MIN_CONTENT_MIX

    def fit(
        self,
        ratings: pd.DataFrame,
        movie_content: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> "HybridRecommender":
        self.content_model.fit(ratings, movie_content, movies=movies)
        self.collaborative_model.fit(ratings)
        self.movies = movies
        self.item_counts = dict(self.collaborative_model.item_counts)
        return self

    def attach_fitted_models(
        self,
        content_model: ContentBasedRecommender,
        collaborative_model: CollaborativeFilteringRecommender,
        movies: pd.DataFrame,
        item_counts: dict[int, int] | None = None,
    ) -> "HybridRecommender":
        """Reuse already-fitted base models when tuning alpha."""
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.movies = movies
        self.item_counts = item_counts or dict(collaborative_model.item_counts)
        return self

    def _hybrid_rating(
        self,
        content_rating: float,
        cf_rating: float,
        movie_id: int | None = None,
    ) -> float:
        """Support-aware blend: more content when the movie has few ratings."""
        if movie_id is None:
            return _clip_rating(self.alpha * content_rating + (1 - self.alpha) * cf_rating)
        blended = blend_hybrid_scores(
            np.array([content_rating], dtype=float),
            np.array([cf_rating], dtype=float),
            np.array([movie_id]),
            self.item_counts,
            self.alpha,
            shrink=self.item_shrink,
            min_mix=self.min_content_mix,
        )
        return float(blended[0])

    def _candidate_scores(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """Score the union of content and collaborative candidate lists."""
        candidate_count = max(n_recommendations * 5, 50)
        content_candidates = self.content_model.top_candidates(user_id, candidate_count)
        cf_candidates = self.collaborative_model.top_candidates(user_id, candidate_count)
        candidate_movie_ids = list(dict.fromkeys(content_candidates + cf_candidates))
        if not candidate_movie_ids:
            raise ValueError(f"Unable to generate candidates for user {user_id}.")

        content_scores = np.array(
            [self.content_model.predict(user_id, movie_id) for movie_id in candidate_movie_ids],
            dtype=float,
        )
        cf_scores = np.array(
            [self.collaborative_model.predict(user_id, movie_id) for movie_id in candidate_movie_ids],
            dtype=float,
        )
        details = blend_hybrid_details(
            content_scores,
            cf_scores,
            np.array(candidate_movie_ids),
            self.item_counts,
            self.alpha,
            shrink=self.item_shrink,
            min_mix=self.min_content_mix,
        )
        scored = pd.DataFrame(
            {
                "movieId": candidate_movie_ids,
                "content_rating": content_scores,
                "cf_rating": cf_scores,
                "hybrid_rating": details["hybrid_rating"],
                "rating_count": details["rating_count"].astype(int),
                "content_weight": details["content_weight"],
                "cf_weight": details["cf_weight"],
                "from_content": np.isin(candidate_movie_ids, content_candidates),
                "from_collaborative": np.isin(candidate_movie_ids, cf_candidates),
            }
        )
        scored["blend_source"] = [
            blend_source_label(float(weight)) for weight in scored["content_weight"]
        ]
        if self.movies is not None:
            scored = scored.merge(
                self.movies[["movieId", "title", "genres"]],
                on="movieId",
                how="left",
            )
        return scored

    def recommend(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        scored = self._candidate_scores(user_id, n_recommendations=n_recommendations)
        return (
            scored.sort_values("hybrid_rating", ascending=False)
            .head(n_recommendations)
            .reset_index(drop=True)
        )

    def candidate_overlap(self, user_id: int, n_recommendations: int = 10) -> dict[str, int]:
        """How the content and SVD candidate pools overlap for this user."""
        scored = self._candidate_scores(user_id, n_recommendations=n_recommendations)
        recs = scored.sort_values("hybrid_rating", ascending=False).head(n_recommendations)
        return {
            "content_candidates": int(scored["from_content"].sum()),
            "collaborative_candidates": int(scored["from_collaborative"].sum()),
            "in_both_pools": int((scored["from_content"] & scored["from_collaborative"]).sum()),
            "recs_from_both": int((recs["from_content"] & recs["from_collaborative"]).sum()),
            "recs_content_only": int((recs["from_content"] & ~recs["from_collaborative"]).sum()),
            "recs_collaborative_only": int((recs["from_collaborative"] & ~recs["from_content"]).sum()),
        }

    def score_disagreements(self, user_id: int, n_movies: int = 8) -> pd.DataFrame:
        """Movies where content and SVD scores differ the most."""
        scored = self._candidate_scores(user_id, n_recommendations=max(n_movies, 10))
        scored = scored.copy()
        scored["score_gap"] = (scored["content_rating"] - scored["cf_rating"]).abs()
        scored["favored_by"] = np.where(
            scored["content_rating"] >= scored["cf_rating"],
            "Content",
            "Collaborative",
        )
        return scored.sort_values("score_gap", ascending=False).head(n_movies).reset_index(drop=True)

    def recommendation_reasons(self, row: pd.Series, user_id: int | None = None) -> list[str]:
        """Short notes explaining why this hybrid recommendation was mixed this way."""
        count = int(row.get("rating_count", 0) or 0)
        content_weight = float(row.get("content_weight", 0.5) or 0.5)
        content_score = float(row.get("content_rating", row.get("Content score", 0.0)) or 0.0)
        cf_score = float(row.get("cf_rating", row.get("Collaborative score", 0.0)) or 0.0)
        source = str(row.get("blend_source", blend_source_label(content_weight)))
        reasons: list[str] = []
        if user_id is not None:
            if getattr(self.content_model, "movies", None) is None:
                self.content_model.movies = self.movies
            if hasattr(self.content_model, "recommendation_reasons"):
                reasons.extend(
                    self.content_model.recommendation_reasons(
                        int(user_id), int(row["movieId"])
                    )[:4]
                )
        reasons.append(f"{source}: {content_weight:.0%} content / {1.0 - content_weight:.0%} SVD")
        if count <= 15:
            reasons.append(f"Rarely rated ({count} ratings), so content has more influence")
        elif count >= 120:
            reasons.append(f"Popular movie ({count} ratings), so SVD has more influence")
        else:
            reasons.append(f"{count} ratings in the training data")
        gap = content_score - cf_score
        if gap >= 0.35:
            reasons.append(
                f"Content likes it more ({content_score:.2f} vs SVD {cf_score:.2f})"
            )
        elif gap <= -0.35:
            reasons.append(
                f"People like you like it more (SVD {cf_score:.2f} vs content {content_score:.2f})"
            )
        return reasons

    def search_movies(
        self,
        query: str = "",
        genres: list[str] | tuple[str, ...] | None = None,
        limit: int = 25,
    ) -> pd.DataFrame:
        """Find catalog movies by title and/or genre."""
        if self.movies is None:
            raise RuntimeError("Movies catalog is not loaded.")
        text = query.strip()
        wanted = [str(genre).strip() for genre in (genres or []) if str(genre).strip()]
        columns = [column for column in ("movieId", "title", "genres", "year") if column in self.movies.columns]
        catalog = self.movies[columns]
        if not text and not wanted:
            return catalog.head(0)

        mask = pd.Series(True, index=catalog.index)
        if text:
            mask &= catalog["title"].str.contains(text, case=False, na=False, regex=False)
        for genre in wanted:
            mask &= catalog["genres"].str.contains(genre, case=False, na=False, regex=False)
        return catalog.loc[mask].head(limit).reset_index(drop=True)

    def score_movie(self, user_id: int, movie_id: int) -> pd.DataFrame:
        """Hybrid content/SVD breakdown for one searched movie and the current user."""
        content_score = float(self.content_model.predict(int(user_id), int(movie_id)))
        cf_score = float(self.collaborative_model.predict(int(user_id), int(movie_id)))
        details = blend_hybrid_details(
            np.array([content_score]),
            np.array([cf_score]),
            np.array([int(movie_id)]),
            self.item_counts,
            self.alpha,
            shrink=self.item_shrink,
            min_mix=self.min_content_mix,
        )
        your_rating = self.content_model.user_ratings.get(int(user_id), {}).get(int(movie_id))
        row = {
            "movieId": int(movie_id),
            "content_rating": content_score,
            "cf_rating": cf_score,
            "hybrid_rating": float(details["hybrid_rating"][0]),
            "rating_count": int(details["rating_count"][0]),
            "content_weight": float(details["content_weight"][0]),
            "cf_weight": float(details["cf_weight"][0]),
            "blend_source": blend_source_label(float(details["content_weight"][0])),
            "your_rating": your_rating,
        }
        scored = pd.DataFrame([row])
        if self.movies is not None:
            scored = scored.merge(
                self.movies[["movieId", "title", "genres"]],
                on="movieId",
                how="left",
            )
        return scored

    def similar_to_movie(
        self,
        user_id: int,
        movie_id: int,
        n_recommendations: int = 10,
    ) -> pd.DataFrame:
        """Unseen movies with similar content, ranked by the hybrid score."""
        content = self.content_model
        if content.tfidf_matrix is None or content.movie_ids is None:
            raise RuntimeError("Content model is not fitted.")
        seed_index = content.movie_id_to_index.get(int(movie_id))
        if seed_index is None:
            raise ValueError("That movie is not in the content catalog.")

        similarities = content.tfidf_matrix @ content.tfidf_matrix[seed_index]
        seen = set(content.user_ratings.get(int(user_id), {}))
        seen.add(int(movie_id))
        ranked = np.argsort(similarities)[::-1]
        picked: list[int] = []
        sim_values: list[float] = []
        pool_size = max(n_recommendations * 4, 40)
        for index in ranked:
            other_id = int(content.movie_ids[index])
            if other_id in seen:
                continue
            picked.append(other_id)
            sim_values.append(float(similarities[index]))
            if len(picked) >= pool_size:
                break
        if not picked:
            raise ValueError("No similar unseen movies found for that title.")

        content_scores = np.array(
            [self.content_model.predict(int(user_id), other_id) for other_id in picked],
            dtype=float,
        )
        cf_scores = np.array(
            [self.collaborative_model.predict(int(user_id), other_id) for other_id in picked],
            dtype=float,
        )
        details = blend_hybrid_details(
            content_scores,
            cf_scores,
            np.array(picked),
            self.item_counts,
            self.alpha,
            shrink=self.item_shrink,
            min_mix=self.min_content_mix,
        )
        similar = pd.DataFrame(
            {
                "movieId": picked,
                "similarity": sim_values,
                "content_rating": content_scores,
                "cf_rating": cf_scores,
                "hybrid_rating": details["hybrid_rating"],
                "rating_count": details["rating_count"].astype(int),
                "content_weight": details["content_weight"],
                "cf_weight": details["cf_weight"],
            }
        )
        similar["blend_source"] = [
            blend_source_label(float(weight)) for weight in similar["content_weight"]
        ]
        if self.movies is not None:
            similar = similar.merge(
                self.movies[["movieId", "title", "genres"]],
                on="movieId",
                how="left",
            )
        return (
            similar.sort_values(["hybrid_rating", "similarity"], ascending=False)
            .head(n_recommendations)
            .reset_index(drop=True)
        )

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict a hybrid rating for one user-movie pair."""
        content_rating = self.content_model.predict(user_id, movie_id)
        cf_rating = self.collaborative_model.predict(user_id, movie_id)
        return self._hybrid_rating(content_rating, cf_rating, movie_id=movie_id)

    def recommend_content(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """Top-N recommendations from the content-based model only."""
        movie_ids = self.content_model.top_candidates(user_id, n_recommendations)
        if not movie_ids:
            raise ValueError(f"Unable to generate content recommendations for user {user_id}.")
        recommendations = pd.DataFrame(
            {
                "movieId": movie_ids,
                "predicted_rating": [
                    self.content_model.predict(user_id, movie_id) for movie_id in movie_ids
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

    def recommend_collaborative(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """Top-N recommendations from the collaborative model only."""
        movie_ids = self.collaborative_model.top_candidates(user_id, n_recommendations)
        if not movie_ids:
            raise ValueError(
                f"Unable to generate collaborative recommendations for user {user_id}."
            )
        recommendations = pd.DataFrame(
            {
                "movieId": movie_ids,
                "predicted_rating": [
                    self.collaborative_model.predict(user_id, movie_id)
                    for movie_id in movie_ids
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
    actual: np.ndarray,
    predicted: np.ndarray,
    relevance_threshold: float = 4.0,
    decision_threshold: float | None = None,
) -> dict[str, float]:
    """Compute RMSE and liked/not-liked metrics, tuning the decision cutoff if needed."""
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
    if decision_threshold is None:
        best_metrics: dict[str, float] | None = None
        for threshold in CLASSIFICATION_THRESHOLDS:
            candidate = _classification_metrics(
                actual, predicted, relevance_threshold, float(threshold)
            )
            if best_metrics is None or candidate["f1_score"] > best_metrics["f1_score"]:
                best_metrics = candidate
        if best_metrics is None:
            raise RuntimeError("Unable to compute classification metrics.")
        metrics.update(best_metrics)
    else:
        metrics.update(
            _classification_metrics(
                actual, predicted, relevance_threshold, decision_threshold
            )
        )
    return metrics


def _collect_base_predictions(
    content_model: ContentBasedRecommender,
    collaborative_model: CollaborativeFilteringRecommender,
    test_ratings: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actual = test_ratings["rating"].to_numpy(dtype=float)
    user_ids = test_ratings["userId"].astype(int).to_numpy()
    movie_ids = test_ratings["movieId"].astype(int).to_numpy()
    content_scores = content_model.predict_many(user_ids, movie_ids)
    cf_scores = collaborative_model.predict_many(user_ids, movie_ids)
    return actual, content_scores, cf_scores, movie_ids


def evaluate_all_models(
    model: HybridRecommender,
    test_ratings: pd.DataFrame,
    relevance_threshold: float = 4.0,
) -> dict[str, dict[str, float]]:
    """Evaluate content-based, collaborative, and hybrid models on the same test set."""
    if test_ratings.empty:
        raise ValueError("Test set is empty. Cannot compute evaluation metrics.")

    actual, content_scores, cf_scores, movie_ids = _collect_base_predictions(
        model.content_model,
        model.collaborative_model,
        test_ratings,
    )
    hybrid_scores = blend_hybrid_scores(
        content_scores,
        cf_scores,
        movie_ids,
        model.item_counts,
        model.alpha,
        shrink=model.item_shrink,
        min_mix=model.min_content_mix,
    )
    return {
        "content": evaluate_predictions(
            actual, content_scores, relevance_threshold=relevance_threshold
        ),
        "collaborative": evaluate_predictions(
            actual, cf_scores, relevance_threshold=relevance_threshold
        ),
        "hybrid": evaluate_predictions(
            actual, hybrid_scores, relevance_threshold=relevance_threshold
        ),
    }


def evaluate_model(
    model: HybridRecommender,
    test_ratings: pd.DataFrame,
    relevance_threshold: float = 4.0,
    decision_threshold: float | None = None,
) -> dict[str, float]:
    """Evaluate RMSE plus liked/not-liked classification metrics on held-out ratings."""
    return evaluate_all_models(model, test_ratings, relevance_threshold=relevance_threshold)[
        "hybrid"
    ]


def tune_alpha(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    movie_content: pd.DataFrame,
    movies: pd.DataFrame,
    alphas: list[float] | None = None,
    n_factors: int = 20,
    relevance_threshold: float = 4.0,
    metric: str = "f1_score",
) -> tuple[float, pd.DataFrame, HybridRecommender, dict[str, dict[str, float]]]:
    """
    Try multiple alpha values and return the best one.

    Base models are trained once; only the hybrid blend weight changes per alpha.
    """
    candidate_alphas = alphas or DEFAULT_ALPHA_GRID
    content_model = ContentBasedRecommender().fit(train_ratings, movie_content)
    collaborative_model = CollaborativeFilteringRecommender(n_factors=n_factors).fit(
        train_ratings
    )
    print("Scoring test ratings with base models...", flush=True)
    actual, content_scores, cf_scores, movie_ids = _collect_base_predictions(
        content_model,
        collaborative_model,
        test_ratings,
    )
    item_counts = dict(collaborative_model.item_counts)

    tuning_rows: list[dict[str, float]] = []
    best_alpha = candidate_alphas[0]
    best_model: HybridRecommender | None = None
    best_metrics: dict[str, float] | None = None
    best_score: float | None = None

    for alpha in candidate_alphas:
        predicted = blend_hybrid_scores(
            content_scores,
            cf_scores,
            movie_ids,
            item_counts,
            alpha,
        )
        print(f"Evaluating alpha={alpha:.2f}...", flush=True)
        metrics = evaluate_predictions(
            actual,
            predicted,
            relevance_threshold=relevance_threshold,
        )
        tuning_rows.append({"alpha": alpha, **metrics})

        score = float(metrics[metric])
        is_better = (
            best_score is None
            or (metric == "rmse" and score < best_score)
            or (metric != "rmse" and score > best_score)
        )
        if is_better:
            best_score = score
            best_alpha = alpha
            best_metrics = metrics
            best_model = HybridRecommender(
                alpha=alpha, n_factors=n_factors
            ).attach_fitted_models(
                content_model,
                collaborative_model,
                movies,
                item_counts=item_counts,
            )

    if best_model is None or best_metrics is None:
        raise RuntimeError("Alpha tuning did not produce a valid model.")

    tuning_results = pd.DataFrame(tuning_rows).sort_values("alpha").reset_index(drop=True)
    all_metrics = {
        "content": evaluate_predictions(
            actual, content_scores, relevance_threshold=relevance_threshold
        ),
        "collaborative": evaluate_predictions(
            actual, cf_scores, relevance_threshold=relevance_threshold
        ),
        "hybrid": best_metrics,
    }
    return best_alpha, tuning_results, best_model, all_metrics


def print_evaluation_metrics(
    metrics: dict[str, float],
    train_size: int,
    test_size: int,
    alpha: float,
    relevance_threshold: float = 4.0,
) -> None:
    print("Model evaluation (80/20 split)")
    print("-" * 32)
    print("Content:        TF-IDF + cosine similarity")
    print("Collaborative:  SVD matrix factorization")
    print("Hybrid formula: support-aware weighted blend of content and SVD")
    print(f"Alpha:          {alpha:.2f} (max content weight; cold movies get more)")
    if "decision_threshold" in metrics:
        print(
            "Liked cutoff:   "
            f"actual >= {relevance_threshold:.1f}, "
            f"predicted >= {metrics['decision_threshold']:.2f}"
        )
    print("-" * 32)
    print(f"Train ratings: {train_size:,}")
    print(f"Test ratings:  {test_size:,}")
    print(f"RMSE:       {metrics['rmse']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-score:   {metrics['f1_score']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid movie recommender.")
    parser.add_argument("--user-id", type=int, default=1, help="User id to recommend for.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of each user's ratings reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Rating threshold used to define actually liked movies. Predicted likes use a cutoff chosen to maximize F1.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Blend weight for content-based score when alpha tuning is disabled.",
    )
    parser.add_argument(
        "--tune-alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search for the best alpha on the test split.",
    )
    parser.add_argument(
        "--tune-metric",
        choices=["f1_score", "precision", "recall", "accuracy", "rmse"],
        default="f1_score",
        help="Metric used to select the best alpha.",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=20,
        help="Number of latent factors used by SVD.",
    )
    parser.add_argument(
        "--skip-recommendations",
        action="store_true",
        help="Only run evaluation without printing sample recommendations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Loading data...", flush=True)
    ratings, movies, movie_content = load_data()
    train_ratings, test_ratings = split_train_test(
        ratings,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if args.tune_alpha:
        print("Training models and tuning alpha (this can take several minutes)...", flush=True)
        best_alpha, tuning_results, model, all_metrics = tune_alpha(
            train_ratings,
            test_ratings,
            movie_content,
            movies,
            n_factors=args.n_factors,
            relevance_threshold=args.relevance_threshold,
            metric=args.tune_metric,
        )
        print("Alpha tuning results")
        print("-" * 32)
        print(
            tuning_results[
                ["alpha", "rmse", "precision", "recall", "f1_score", "accuracy", "decision_threshold"]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
        print("-" * 32)
        print(f"Best alpha ({args.tune_metric}): {best_alpha:.2f}")
        print()
        metrics = all_metrics
    else:
        print("Training hybrid model...", flush=True)
        best_alpha = args.alpha
        model = HybridRecommender(
            alpha=best_alpha,
            n_factors=args.n_factors,
        ).fit(train_ratings, movie_content, movies)
        print("Evaluating on the test split...", flush=True)
        metrics = evaluate_all_models(
            model,
            test_ratings,
            relevance_threshold=args.relevance_threshold,
        )

    for name, label in (
        ("content", "Content-based (TF-IDF)"),
        ("collaborative", "Collaborative (SVD)"),
        ("hybrid", "Hybrid"),
    ):
        print(label)
        print_evaluation_metrics(
            metrics[name],
            train_size=len(train_ratings),
            test_size=len(test_ratings),
            alpha=best_alpha,
            relevance_threshold=args.relevance_threshold,
        )
        print()

    if not args.skip_recommendations:
        print()
        results = model.recommend(args.user_id, n_recommendations=args.top_n)
        print(f"Hybrid recommendations for user {args.user_id} (trained on 80% split):")
        print(
            results[
                [
                    "movieId",
                    "title",
                    "genres",
                    "content_rating",
                    "cf_rating",
                    "hybrid_rating",
                ]
            ].to_string(index=False)
        )
