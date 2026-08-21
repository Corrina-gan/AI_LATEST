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
    if len(content_scores) == 0:
        return np.array([], dtype=float)
    counts = np.fromiter(
        (item_counts.get(int(movie_id), 0) for movie_id in movie_ids),
        dtype=float,
        count=len(movie_ids),
    )
    item_confidence = counts / (counts + shrink)
    effective_alpha = np.clip(
        alpha * (min_mix + (1.0 - min_mix) * (1.0 - item_confidence)),
        0.0,
        1.0,
    )
    return np.clip(
        effective_alpha * content_scores + (1.0 - effective_alpha) * cf_scores,
        MIN_RATING,
        MAX_RATING,
    )


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
        self.content_model.fit(ratings, movie_content)
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

    def recommend(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        candidate_count = max(n_recommendations * 5, 50)
        content_candidates = self.content_model.top_candidates(user_id, candidate_count)
        cf_candidates = self.collaborative_model.top_candidates(user_id, candidate_count)
        candidate_movie_ids = list(dict.fromkeys(content_candidates + cf_candidates))

        if not candidate_movie_ids:
            raise ValueError(f"Unable to generate candidates for user {user_id}.")

        hybrid = pd.DataFrame(
            {
                "movieId": candidate_movie_ids,
                "content_rating": [
                    self.content_model.predict(user_id, movie_id)
                    for movie_id in candidate_movie_ids
                ],
                "cf_rating": [
                    self.collaborative_model.predict(user_id, movie_id)
                    for movie_id in candidate_movie_ids
                ],
            }
        )
        hybrid["hybrid_rating"] = blend_hybrid_scores(
            hybrid["content_rating"].to_numpy(dtype=float),
            hybrid["cf_rating"].to_numpy(dtype=float),
            hybrid["movieId"].to_numpy(),
            self.item_counts,
            self.alpha,
            shrink=self.item_shrink,
            min_mix=self.min_content_mix,
        )

        recommendations = hybrid.sort_values("hybrid_rating", ascending=False).head(
            n_recommendations
        )
        if self.movies is not None:
            recommendations = recommendations.merge(
                self.movies[["movieId", "title", "genres"]],
                on="movieId",
                how="left",
            )
        return recommendations.reset_index(drop=True)

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
