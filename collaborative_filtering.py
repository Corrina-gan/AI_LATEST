"""Collaborative filtering recommender using SVD matrix factorization.

Standalone module used by the Streamlit app for the Collaborative algorithm path.
Hybrid keeps its own CF implementation and does not import this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

BASE_DIR = Path(__file__).resolve().parent
MIN_RATING = 0.5
MAX_RATING = 5.0


def _clip_rating(value: float) -> float:
    return float(np.clip(value, MIN_RATING, MAX_RATING))


def _find_processed_dir() -> Path:
    for candidate in (BASE_DIR / "processed", BASE_DIR / "dataset" / "processed"):
        if (candidate / "ratings_clean.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Processed data not found. Run `py data_preprocessing.py` first."
    )


class CollaborativeFiltering:
    """Collaborative filtering with SVD matrix factorization + item bias."""

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
        self.item_bias: np.ndarray | None = None
        self.user_train_movies: dict[int, set[int]] = {}
        self.global_mean: float = 3.5
        self.movies: pd.DataFrame | None = None

    def fit(
        self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None
    ) -> "CollaborativeFiltering":
        self.movies = movies
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
        self.item_bias = item_means - self.global_mean

        # Center by user mean + item bias before SVD (matches hybrid CF).
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

        # svds returns singular values in ascending order.
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

    def recommend(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """Return top-N unseen movie recommendations for a user."""
        movie_ids = self.top_candidates(user_id, n_recommendations)
        if not movie_ids:
            raise ValueError(f"No ratings found for user {user_id}.")

        recommendations = pd.DataFrame(
            {
                "movieId": movie_ids,
                "predicted_rating": [
                    self.predict(user_id, movie_id) for movie_id in movie_ids
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

    # ------------------------------------------------------------------
    # Evaluation (standalone; app should use this, not hybrid.py)
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
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

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
        from sklearn.metrics import mean_squared_error

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


def load_data(processed_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = processed_dir or _find_processed_dir()
    ratings = pd.read_csv(processed_dir / "ratings_clean.csv")
    movies = pd.read_csv(processed_dir / "movies_clean.csv")
    return ratings, movies


def demo(user_id: int = 1, top_n: int = 10, n_factors: int = 20) -> pd.DataFrame:
    ratings, movies = load_data()
    model = CollaborativeFiltering(n_factors=n_factors).fit(ratings, movies=movies)
    return model.recommend(user_id, n_recommendations=top_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SVD collaborative filtering.")
    parser.add_argument("--user-id", type=int, default=1, help="User id to recommend for.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations.")
    parser.add_argument(
        "--n-factors",
        type=int,
        default=20,
        help="Number of latent factors used by SVD.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = demo(user_id=args.user_id, top_n=args.top_n, n_factors=args.n_factors)
    print(f"Collaborative filtering recommendations for user {args.user_id}:")
    print(results.to_string(index=False))
