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


def build_movie_content(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Combine genres and tags into one text field per movie."""
    genre_text = (
        movies["genres"]
        .str.replace("(no genres listed)", "", regex=False)
        .str.replace("|", " ")
        .str.strip()
    )

    tag_column = "tag_standardization" if "tag_standardization" in tags.columns else "tag"
    tag_text = (
        tags.groupby("movieId")[tag_column]
        .apply(lambda values: " ".join(values))
        .reset_index(name="tags")
    )

    movie_content = movies[["movieId", "title", "genres"]].copy()
    movie_content["genres_text"] = genre_text
    movie_content = movie_content.merge(tag_text, on="movieId", how="left")
    movie_content["tags"] = movie_content["tags"].fillna("")
    movie_content["content_features"] = (
        movie_content["genres_text"] + " " + movie_content["tags"]
    ).str.strip().fillna("")
    return movie_content


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
        self.vectorizer = TfidfVectorizer(stop_words="english")

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
        """Predict rating via cosine-weighted average of the user's rated movies."""
        if self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")

        movie_index = self.movie_id_to_index.get(movie_id)
        rated_indices = self.user_rated_indices.get(user_id)
        rated_values = self.user_rated_values.get(user_id)
        if movie_index is None or rated_indices is None or rated_values is None:
            return self.user_means.get(user_id, self.global_mean)

        # L2-normalized → cosine similarity is a dot product.
        similarities = self.tfidf_matrix[rated_indices] @ self.tfidf_matrix[movie_index]
        weights = np.clip(similarities, 0.0, None)
        if float(weights.sum()) <= 1e-9:
            return self.user_means.get(user_id, self.global_mean)

        predicted = float(np.dot(weights, rated_values) / weights.sum())
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
