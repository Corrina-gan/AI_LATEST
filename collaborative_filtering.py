"""
Collaborative Filtering for the Movie Recommender System

Model variants:
1. User-based CF        -> cosine similarity between user rating vectors
2. Item-based CF        -> cosine, or adjusted-cosine ("Pearson-style"), similarity between item rating vectors
3. Matrix factorization -> TruncatedSVD on the mean-centered rating matrix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Constants
BASE_DIR = Path(__file__).resolve().parent
MIN_RATING = 0.5
MAX_RATING = 5.0
ALGORITHM_KEY = "collaborative"

# Internal short codes for the three CF variants 
VARIANT_USER = "user"
VARIANT_ITEM = "item"
VARIANT_SVD = "svd"
VARIANT_OPTIONS = {
    "User-Based CF": VARIANT_USER,
    "Item-Based CF": VARIANT_ITEM,
    "Matrix Factorization (SVD)": VARIANT_SVD,
}

# Item-based CF supports two similarity metrics
ITEM_METHOD_COSINE = "cosine"
ITEM_METHOD_PEARSON = "pearson"
ITEM_METHOD_OPTIONS = {
    "Cosine": ITEM_METHOD_COSINE,
    "Pearson (adjusted cosine)": ITEM_METHOD_PEARSON,
}

DEFAULT_NEIGHBORHOOD_K = 30                 # how many similar users/items to use in kNN-style prediction
DEFAULT_N_COMPONENTS = 20                   # number of latent factors for SVD
DEFAULT_ITEM_METHOD = ITEM_METHOD_COSINE

# Module-level fallback so recommend() still has sensible item-method / SVD
# settings even if only variant/k/genres were passed in
_LAST_CONTROLS: dict[str, object] = {
    "item_method": DEFAULT_ITEM_METHOD,
    "n_components": DEFAULT_N_COMPONENTS,
}


def _split_genres(value: object) -> list[str]:
    '''Split a genre string into a list of individual genres, 
    dropping the placeholder (no genres listed)'''
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value)
    if text.startswith("["):
        # Handle the case where genres were stored as a Python list literal
        # (e.g. "['Action', 'Comedy']") rather than a pipe-separated string.
        try:
            import ast

            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                text = "|".join(str(item) for item in parsed)
        except (SyntaxError, ValueError):
            pass
    return [
        part.strip()
        for part in text.replace(",", "|").split("|")
        if part.strip() and part.strip() != "(no genres listed)"
    ]

#Keep a predicted rating inside the valid 0.5-5.0 scale
def _clip_rating(value: float) -> float:
    return float(np.clip(value, MIN_RATING, MAX_RATING))


def _find_processed_dir() -> Path:
    """Locate the folder containing the cleaned/preprocessed CSVs, checking
    a couple of likely locations relative to this file."""
    for candidate in (BASE_DIR / "processed", BASE_DIR / "dataset" / "processed"):
        if (candidate / "ratings_clean.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Processed data not found. Run `py data_preprocessing.py` first."
    )


def _normalize_variant(variant: str) -> str:
    if variant in VARIANT_OPTIONS:
        return VARIANT_OPTIONS[variant]
    if variant in VARIANT_OPTIONS.values():
        return variant
    raise ValueError(f"Unknown collaborative model variant: {variant!r}")


class CollaborativeFiltering:
    """
    Collaborative Filtering recommender with three interchangeable model
    variants, cold-start/business-rule handling, and offline evaluation.

    Usage:
        cf = CollaborativeFiltering(n_factors=20).fit(ratings_df, movies=movies)
        recs = cf.recommend(user_id=1, variant=cf.MODEL_USER_BASED, n_recommendations=10)
    """

    # Valid rating range and tunable thresholds used across the model
    RATING_MIN, RATING_MAX = MIN_RATING, MAX_RATING
    COLD_START_MIN_RATINGS = 5       # users below this many ratings get the popularity fallback
    MIN_ITEM_RATING_COUNT = 5        # movies below this many ratings are excluded from item similarity

    MIN_PREDICTION_SUPPORT = 2       # need at least this many contributing neighbors to trust a prediction

    POPULARITY_PRIOR_VOTES = 10       # "virtual votes" added in the Bayesian popularity average

    # Display labels for each model variant
    MODEL_USER_BASED = "User-Based CF"
    MODEL_ITEM_BASED = "Item-Based CF"
    MODEL_SVD = "Matrix Factorization (SVD)"
    MODELS = [MODEL_USER_BASED, MODEL_ITEM_BASED, MODEL_SVD]

    def __init__(self, n_factors: int = DEFAULT_N_COMPONENTS) -> None:
        # Number of latent factors used by the default SVD model instance.
        self.n_factors = int(n_factors)

        # Populated by fit(): raw data + lookup tables.
        self.ratings_df: pd.DataFrame | None = None
        self.movies: pd.DataFrame | None = None
        self.movie_lookup: pd.DataFrame | None = None

        # The user-item matrices: one with NaN for unrated cells, one with 0.0 for unrated cells (used for similarity/SVD)
        self.ratings_matrix: pd.DataFrame | None = None
        self.filled_matrix: pd.DataFrame | None = None

        self.user_ids: np.ndarray | None = None
        self.movie_ids: np.ndarray | None = None
        self.user_id_to_index: dict[int, int] = {}
        self.movie_id_to_index: dict[int, int] = {}

        # Per-user quick-lookup structures used for cold-start checks, excluding already-rated movies, and explanation generation
        self.user_train_movies: dict[int, set[int]] = {}
        self.user_train_ratings: dict[int, dict[int, float]] = {}
        self.movie_genre_sets: dict[int, set[str]] = {}

        # Caches so repeated calls don't recompute expensive similarity
        # SVD matrices. Cleared whenever fit() is called again
        self.global_mean: float = 3.5
        self._user_similarity: pd.DataFrame | None = None
        self._item_similarity_cache: dict[str, pd.DataFrame] = {}
        self._svd_cache: dict[int, pd.DataFrame] = {}

        # State from the most recent recommend() call, used to build human-readable explanations and to report which strategy was used
        self._popular_movies_cache: pd.DataFrame | None = None
        self._last_scores: pd.DataFrame | None = None
        self.last_is_cold_start = False
        self.last_rationale_label = self.MODEL_SVD
        self.last_item_method = DEFAULT_ITEM_METHOD
        self.last_n_components = int(n_factors)

    def fit(
        self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None
    ) -> "CollaborativeFiltering":
        required = {"userId", "movieId", "rating"}
        missing = required - set(ratings.columns)
        if missing:
            raise ValueError(f"ratings is missing required columns: {missing}")

        # Drop incomplete rows and enforce expected dtypes.
        self.ratings_df = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
        self.ratings_df["userId"] = self.ratings_df["userId"].astype(int)
        self.ratings_df["movieId"] = self.ratings_df["movieId"].astype(int)
        self.ratings_df["rating"] = self.ratings_df["rating"].astype(float)
        self.movies = movies
        self.global_mean = float(self.ratings_df["rating"].mean())

        # Build the user x movie ratings matrix (NaN + zero-filled versions)
        self.ratings_matrix, self.filled_matrix = self.build_user_item_matrix(self.ratings_df)
        self.user_ids = self.ratings_matrix.index.to_numpy()
        self.movie_ids = self.ratings_matrix.columns.to_numpy()
        self.user_id_to_index = {
            int(user_id): index for index, user_id in enumerate(self.user_ids)
        }
        self.movie_id_to_index = {
            int(movie_id): index for index, movie_id in enumerate(self.movie_ids)
        }

        '''Build a movieId -> (title, genres, year) lookup table, preferring
           the movies table if given, otherwise falling back to whatever
           title/genres columns exist on the ratings frame itself'''
        lookup_source = movies if movies is not None else self.ratings_df
        lookup_cols = [column for column in ("movieId", "title", "genres", "year") if column in lookup_source.columns]
        if "movieId" not in lookup_cols:
            raise ValueError("Need a movies table (or title/genres on ratings) to look up recommendations.")
        self.movie_lookup = (
            lookup_source[lookup_cols].drop_duplicates(subset="movieId").set_index("movieId")
        )

        self.user_train_movies = {}
        self.user_train_ratings = {}
        for user_id, user_frame in self.ratings_df.groupby("userId"):
            uid = int(user_id)
            self.user_train_movies[uid] = set(user_frame["movieId"].astype(int))
            self.user_train_ratings[uid] = {
                int(row.movieId): float(row.rating) for row in user_frame.itertuples()
            }

        self.movie_genre_sets = {}
        if self.movie_lookup is not None and "genres" in self.movie_lookup.columns:
            for movie_id, row in self.movie_lookup.iterrows():
                self.movie_genre_sets[int(movie_id)] = set(_split_genres(row.get("genres", "")))

        self._user_similarity = None
        self._item_similarity_cache = {}
        self._svd_cache = {}
        self._popular_movies_cache = None
        self._last_scores = None
        return self

    def _require_fit(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.ratings_matrix is None or self.filled_matrix is None or self.movie_lookup is None:
            raise RuntimeError("Collaborative model is not fitted.")
        return self.ratings_matrix, self.filled_matrix, self.movie_lookup

    @staticmethod
    def build_user_item_matrix(ratings_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Pivot to a (userId x movieId) matrix

        Returns:
            ratings_matrix: NaN where a user hasn't rated a movie
            filled_matrix:  unrated cells filled with 0 (needed for similarity / SVD)
        """
        ratings_matrix = ratings_df.pivot_table(index="userId", columns="movieId", values="rating")
        filled_matrix = ratings_matrix.fillna(0.0)
        return ratings_matrix, filled_matrix

    """Percentage of user-movie cells that are NOT rated (i.e. how
        sparse the ratings matrix is). Used in the EDA section."""
    @staticmethod
    def matrix_sparsity(ratings_matrix: pd.DataFrame) -> float:
        total_cells = ratings_matrix.shape[0] * ratings_matrix.shape[1]
        if total_cells == 0:
            return 0.0
        filled_cells = int(ratings_matrix.notna().sum().sum())
        return 100.0 * (1 - filled_cells / total_cells)

    """Return every distinct genre label present in the movie catalog,
        used to populate the genre-filter dropdown in the UI."""
    def get_all_genres(self) -> list[str]:
        labels: set[str] = set()
        for genres in (self.movie_lookup["genres"] if self.movie_lookup is not None else []):
            labels.update(_split_genres(genres))
        return sorted(labels)

    def user_similarity(self) -> pd.DataFrame:
        """Cosine similarity between every pair of users' rating vectors."""
        _, filled_matrix, _ = self._require_fit()
        if self._user_similarity is None:
            # User-user cosine similarity
            sim = cosine_similarity(filled_matrix.to_numpy())
            self._user_similarity = pd.DataFrame(
                sim, index=filled_matrix.index, columns=filled_matrix.index
            )
        return self._user_similarity

    def item_similarity(self, method: str = DEFAULT_ITEM_METHOD) -> pd.DataFrame:
        """Similarity between movie rating vectors (eligible movies only)."""
        ratings_matrix, filled_matrix, _ = self._require_fit()
        method = ITEM_METHOD_PEARSON if method == ITEM_METHOD_PEARSON else ITEM_METHOD_COSINE
        if method not in self._item_similarity_cache:
            self._item_similarity_cache[method] = self._compute_item_similarity(
                ratings_matrix, filled_matrix, method, self.MIN_ITEM_RATING_COUNT
            )
        return self._item_similarity_cache[method]

    @staticmethod
    def _compute_item_similarity(
        ratings_matrix: pd.DataFrame,
        filled_matrix: pd.DataFrame,
        method: str,
        min_rating_count: int,
    ) -> pd.DataFrame:
        """Compute the item-item similarity matrix, restricted to movies
        with at least `min_rating_count` ratings (too-sparse movies would
        give unreliable similarity scores)."""
        rating_counts = ratings_matrix.notna().sum(axis=0)
        eligible_movies = rating_counts[rating_counts >= min_rating_count].index
        sub_filled = filled_matrix[eligible_movies]
        if method == ITEM_METHOD_PEARSON:
            # Adjusted cosine: subtract each movie's own mean rating first, so similarity reflects rating *pattern* rather than raw level
            item_means = ratings_matrix[eligible_movies].mean(axis=0).fillna(0.0)
            rated_mask = ratings_matrix[eligible_movies].notna()
            centered = (sub_filled - item_means).where(rated_mask, 0.0)
            sim = cosine_similarity(centered.to_numpy().T)
        else:
            # Plain cosine similarity on raw (zero-filled) rating vectors
            sim = cosine_similarity(sub_filled.to_numpy().T)
        return pd.DataFrame(sim, index=eligible_movies, columns=eligible_movies)

    def svd_predictions(self, n_components: int = DEFAULT_N_COMPONENTS) -> pd.DataFrame:
        """r_hat(u, i) = mean(u) + TruncatedSVD reconstruction of the centered matrix."""
        ratings_matrix, filled_matrix, _ = self._require_fit()
        n_components = max(1, int(n_components))
        if n_components not in self._svd_cache:
            self._svd_cache[n_components] = self._fit_svd(
                ratings_matrix, filled_matrix, n_components
            )
        return self._svd_cache[n_components]

    @classmethod
    def _fit_svd(
        cls, ratings_matrix: pd.DataFrame, filled_matrix: pd.DataFrame, n_components: int
    ) -> pd.DataFrame:
        '''Center the matrix by each user's average rating (removes
        generous/harsh-rater bias), factorize with TruncatedSVD, then
        reconstruct predicted ratings by adding the user mean back.'''
        global_mean = float(ratings_matrix.stack().mean())
        user_means = ratings_matrix.mean(axis=1, skipna=True).fillna(global_mean)
        rated_mask = ratings_matrix.notna()

        centered = filled_matrix.sub(user_means, axis=0).where(rated_mask, 0.0)
        n_components = max(1, min(n_components, min(centered.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        u_factors = svd.fit_transform(centered.to_numpy())
        reconstructed = u_factors @ svd.components_

        predicted = reconstructed + user_means.to_numpy().reshape(-1, 1)
        predicted = np.clip(predicted, cls.RATING_MIN, cls.RATING_MAX)
        return pd.DataFrame(predicted, index=filled_matrix.index, columns=filled_matrix.columns)

    def predict_user_based(self, user_id: int, k_neighbors: int = DEFAULT_NEIGHBORHOOD_K) -> pd.DataFrame:
        ratings_matrix, _, _ = self._require_fit()
        if user_id not in ratings_matrix.index:
            return pd.DataFrame(columns=["predicted_rating", "confidence"])

        user_similarity = self.user_similarity()
        user_means = ratings_matrix.mean(axis=1, skipna=True)
        target_mean = float(user_means.loc[user_id])

        # Pick the top-k most similar users
        sims = user_similarity.loc[user_id].drop(index=user_id, errors="ignore")
        top_neighbors = sims.sort_values(ascending=False).head(int(k_neighbors))
        top_neighbors = top_neighbors[top_neighbors > 0]
        if top_neighbors.empty:
            return pd.DataFrame(columns=["predicted_rating", "confidence"])

        # Mean-centered weighted average of neighbor ratings
        neighbor_ratings = ratings_matrix.loc[top_neighbors.index]
        neighbor_means = user_means.loc[top_neighbors.index]
        diff = neighbor_ratings.sub(neighbor_means, axis=0)
        weighted = diff.mul(top_neighbors, axis=0)
        numerator = weighted.sum(axis=0, skipna=True)
        contributed = diff.notna()
        denominator = contributed.mul(top_neighbors.abs(), axis=0).sum(axis=0)
        support_count = contributed.sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            predicted = target_mean + (numerator / denominator)
        predicted = predicted.where(
            (denominator > 0) & (support_count >= self.MIN_PREDICTION_SUPPORT)
        )
        # Only keep predictions backed by enough neighbor support
        predicted = predicted.clip(self.RATING_MIN, self.RATING_MAX)

        # Confidence = average similarity weight behind the prediction
        confidence = (denominator / support_count.replace(0, np.nan)).fillna(0.0)
        result = pd.DataFrame({"predicted_rating": predicted, "confidence": confidence})
        result.index.name = "movieId"

        # Exclude movies the user has already rated
        already_rated = ratings_matrix.loc[user_id].dropna().index
        return result.drop(index=already_rated, errors="ignore").dropna(subset=["predicted_rating"])

    def predict_item_based(
        self,
        user_id: int,
        k_neighbors: int = DEFAULT_NEIGHBORHOOD_K,
        method: str = DEFAULT_ITEM_METHOD,
        item_means: pd.Series | None = None,
    ) -> pd.DataFrame:
        '''Predict ratings for all unseen movies using the k most similar
        movies (to each candidate) that the user has already rated.'''
        ratings_matrix, _, _ = self._require_fit()
        if user_id not in ratings_matrix.index:
            return pd.DataFrame(columns=["predicted_rating", "confidence"])

        item_sim = self.item_similarity(method=method)
        if item_means is None:
            item_means = ratings_matrix.mean(axis=0, skipna=True)
        return self._predict_item_based_with(
            ratings_matrix, item_sim, user_id, int(k_neighbors), item_means
        )

    def _predict_item_based_with(
        self,
        ratings_matrix: pd.DataFrame,
        item_similarity: pd.DataFrame,
        user_id: int,
        k_neighbors: int,
        item_means: pd.Series,
    ) -> pd.DataFrame:
        # Vectorized item-based prediction
        if user_id not in ratings_matrix.index:
            return pd.DataFrame(columns=["predicted_rating", "confidence"])

        user_row = ratings_matrix.loc[user_id].dropna()
        rated_movie_ids = [movie_id for movie_id in user_row.index if movie_id in item_similarity.columns]
        if not rated_movie_ids:
            return pd.DataFrame(columns=["predicted_rating", "confidence"])

        candidate_ids = item_similarity.index

        # similarity between every candidate movie and every movie the user rated
        sim_matrix = item_similarity.loc[candidate_ids, rated_movie_ids].to_numpy()
        rated_means = item_means.loc[rated_movie_ids].to_numpy()

        # how much the user's rating deviated from each rated movie's own average
        deviations = user_row.loc[rated_movie_ids].to_numpy() - rated_means

        # For each candidate movie, keep only its top-k most similar rated movies
        k = min(max(int(k_neighbors), 1), sim_matrix.shape[1])
        abs_sim = np.abs(sim_matrix)
        top_k_idx = np.argpartition(-abs_sim, kth=k - 1, axis=1)[:, :k]
        mask = np.zeros_like(sim_matrix, dtype=bool)
        row_idx = np.arange(sim_matrix.shape[0])[:, None]
        mask[row_idx, top_k_idx] = True
        mask &= abs_sim > 0
        masked_sim = np.where(mask, sim_matrix, 0.0)

        numerator = masked_sim @ deviations
        denominator = np.abs(masked_sim).sum(axis=1)
        support_count = mask.sum(axis=1)
        candidate_means = item_means.reindex(candidate_ids).fillna(item_means.mean()).to_numpy()

        with np.errstate(invalid="ignore", divide="ignore"):
            predicted = candidate_means + numerator / denominator
        # Only keep predictions backed by enough neighbor support
        predicted = np.where(
            (denominator > 0) & (support_count >= self.MIN_PREDICTION_SUPPORT),
            predicted,
            np.nan,
        )
        predicted = np.clip(predicted, self.RATING_MIN, self.RATING_MAX)
        # Confidence = average absolute similarity weight used
        confidence = np.divide(
            denominator,
            np.where(support_count > 0, support_count, np.nan),
            out=np.zeros_like(denominator),
            where=support_count > 0,
        )
        result = pd.DataFrame(
            {"predicted_rating": predicted, "confidence": confidence},
            index=candidate_ids,
        )
        result.index.name = "movieId"
        # Exclude movies the user has already rated
        return result.drop(index=rated_movie_ids, errors="ignore").dropna(subset=["predicted_rating"])

    def predict_svd(self, user_id: int, n_components: int = DEFAULT_N_COMPONENTS) -> pd.DataFrame:
        ratings_matrix, _, _ = self._require_fit()
        svd_predictions = self.svd_predictions(n_components=n_components)
        if user_id not in svd_predictions.index:
            return pd.DataFrame(columns=["predicted_rating", "confidence"])
        predicted = svd_predictions.loc[user_id]
        already_rated = (
            ratings_matrix.loc[user_id].dropna().index if user_id in ratings_matrix.index else []
        )
        predicted = predicted.drop(index=already_rated, errors="ignore")

        # Normalize confidence by the spread of this user's predicted ratings
        spread = float(predicted.max() - predicted.min()) if not predicted.empty else 0.0
        if spread > 0:
            confidence = (predicted - predicted.min()) / spread
        else:
            confidence = pd.Series(0.5, index=predicted.index)
        result = pd.DataFrame({"predicted_rating": predicted, "confidence": confidence})
        result.index.name = "movieId"
        return result

    def get_popular_movies(self) -> pd.DataFrame:
        """Bayesian weighted rating so a few 5-stars don't beat a large good average."""
        if self.ratings_df is None:
            raise RuntimeError("Collaborative model is not fitted.")
        if self._popular_movies_cache is None:
            stats = self.ratings_df.groupby("movieId")["rating"].agg(
                vote_count="count", avg_rating="mean"
            )
            global_mean = float(self.ratings_df["rating"].mean())
            prior = self.POPULARITY_PRIOR_VOTES
            '''Bayesian average: blends each movie's own average with the
            global average, weighted by how many votes it has. Movies
            with few votes get pulled toward the global mean'''
            stats["predicted_rating"] = (
                stats["vote_count"] / (stats["vote_count"] + prior) * stats["avg_rating"]
                + prior / (stats["vote_count"] + prior) * global_mean
            )
            stats["confidence"] = (stats["vote_count"] / stats["vote_count"].max()).clip(upper=1.0)
            self._popular_movies_cache = stats.sort_values("predicted_rating", ascending=False)[
                ["predicted_rating", "confidence"]
            ]
            self._popular_movies_cache.index.name = "movieId"
        return self._popular_movies_cache

    def _genre_ok(self, movie_id: int, genres: list[str] | tuple[str, ...] | None) -> bool:
        if not genres:
            return True
        movie_genres = self.movie_genre_sets.get(int(movie_id), set())
        return any(genre in movie_genres for genre in genres)

    def _finalize_scores(
        self,
        scores: pd.DataFrame,
        top_k: int,
        genres_filter: list[str] | tuple[str, ...] | None,
    ) -> pd.DataFrame:
        _, _, movie_lookup = self._require_fit()
        if scores.empty:
            return pd.DataFrame(columns=["movieId", "title", "genres", "predicted_rating", "confidence"])
        wanted = [str(genre).strip() for genre in (genres_filter or []) if str(genre).strip()]
        if wanted:
            keep_ids = [
                movie_id
                for movie_id in scores.index
                if self._genre_ok(int(movie_id), wanted)
            ]
            scores = scores.loc[keep_ids]
        scores = scores.sort_values(
            ["predicted_rating", "confidence"], ascending=[False, False]
        ).head(int(top_k))
        if scores.empty:
            return pd.DataFrame(columns=["movieId", "title", "genres", "predicted_rating", "confidence"])
        out = scores.join(movie_lookup, how="left")
        out = out.reset_index()
        if "movieId" not in out.columns:
            out = out.rename(columns={out.columns[0]: "movieId"})
        out["movieId"] = out["movieId"].astype(int)
        out["predicted_rating"] = out["predicted_rating"].astype(float).round(4)
        out["confidence"] = out["confidence"].astype(float).round(4)
        columns = ["movieId", "predicted_rating", "confidence"]
        for extra in ("title", "genres", "year"):
            if extra in out.columns:
                columns.append(extra)
        return out[columns].reset_index(drop=True)

    def _scores_for_model(
        self,
        user_id: int,
        variant: str,
        neighborhood_k: int,
        item_method: str,
        n_components: int,
    ) -> pd.DataFrame:
        """Dispatch to the right prediction method based on the chosen variant."""
        variant = _normalize_variant(variant)
        if variant == VARIANT_USER:
            return self.predict_user_based(user_id, neighborhood_k)
        if variant == VARIANT_ITEM:
            return self.predict_item_based(user_id, neighborhood_k, method=item_method)
        return self.predict_svd(user_id, n_components=n_components)

    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        variant: str = VARIANT_USER,
        neighborhood_k: int = DEFAULT_NEIGHBORHOOD_K,
        genres: list[str] | tuple[str, ...] | None = None,
        item_method: str | None = None,
        n_components: int | None = None,
        model: str | None = None,
        top_k: int | None = None,
        min_rating: float = 0.0,
        genres_filter: list[str] | tuple[str, ...] | None = None,
        k_neighbors: int | None = None,
    ) -> pd.DataFrame:
        """Top-N unseen recommendations. Already-watched movies are excluded.

        Accepts both the app kwargs (`variant`, `neighborhood_k`, `genres`)
        and the assignment CLI names (`model`, `top_k`, `genres_filter`).
        """
        self._require_fit()
        user_id = int(user_id)
        top_n = int(top_k if top_k is not None else n_recommendations)
        variant = _normalize_variant(model if model is not None else variant)
        k = int(k_neighbors if k_neighbors is not None else neighborhood_k)
        wanted = genres_filter if genres_filter is not None else genres
        if item_method is None:
            item_method = str(_LAST_CONTROLS.get("item_method") or DEFAULT_ITEM_METHOD)
        if n_components is None:
            n_components = int(_LAST_CONTROLS.get("n_components") or self.n_factors)
        components = int(n_components)

        # Cold-start check
        n_user_ratings = len(self.user_train_ratings.get(user_id, {}))
        is_cold_start = n_user_ratings < self.COLD_START_MIN_RATINGS
        rationale_label = (
            self.MODEL_USER_BASED
            if variant == VARIANT_USER
            else self.MODEL_ITEM_BASED
            if variant == VARIANT_ITEM
            else self.MODEL_SVD
        )

        if is_cold_start:
            scores = self.get_popular_movies().copy()
            scores = scores.drop(index=self.user_train_movies.get(user_id, set()), errors="ignore")
            rationale_label = "Popularity fallback (cold start)"
        else:
            # Normal path: try the chosen CF variant
            try:
                scores = self._scores_for_model(user_id, variant, k, item_method, components)
            except Exception:
                scores = pd.DataFrame(columns=["predicted_rating", "confidence"])
            if min_rating > 0 and not scores.empty:
                scores = scores[scores["predicted_rating"] >= min_rating]
            if scores.empty:
                # Graceful fallback
                scores = self.get_popular_movies().copy()
                scores = scores.drop(index=self.user_train_movies.get(user_id, set()), errors="ignore")
                is_cold_start = True
                rationale_label = "Popularity fallback (no confident model predictions)"

        recommendations = self._finalize_scores(scores, top_n, wanted)
        if recommendations.empty:
            if wanted:
                raise ValueError("No unseen movies match that genre filter.")
            raise ValueError(f"No ratings found for user {user_id}.")

        # Remember the details of this call so recommendation_reasons() can explain it
        self._last_scores = scores
        self.last_is_cold_start = is_cold_start
        self.last_rationale_label = rationale_label
        self.last_item_method = str(item_method)
        self.last_n_components = components
        return recommendations

    def top_candidates(
        self,
        user_id: int,
        n_candidates: int,
        variant: str = VARIANT_SVD,
        neighborhood_k: int = DEFAULT_NEIGHBORHOOD_K,
        genres: list[str] | tuple[str, ...] | None = None,
        item_method: str | None = None,
        n_components: int | None = None,
    ) -> list[int]:
        recs = self.recommend(
            user_id,
            n_recommendations=n_candidates,
            variant=variant,
            neighborhood_k=neighborhood_k,
            genres=genres,
            item_method=item_method,
            n_components=n_components,
        )
        return recs["movieId"].astype(int).tolist()

    def predict(self, user_id: int, movie_id: int) -> float:
        """SVD predicted rating for one cell (used by the 80/20 evaluation)."""
        pred_matrix = self.svd_predictions(n_components=self.n_factors)
        if user_id not in pred_matrix.index or movie_id not in pred_matrix.columns:
            return self.global_mean
        return _clip_rating(float(pred_matrix.loc[user_id, movie_id]))

    def recommendation_reasons(
        self,
        user_id: int,
        movie_id: int,
        variant: str = VARIANT_SVD,
        neighborhood_k: int = DEFAULT_NEIGHBORHOOD_K,
        item_method: str | None = None,
    ) -> list[str]:
        """Short notes explaining a collaborative recommendation."""
        movie_id = int(movie_id)
        user_id = int(user_id)
        variant = _normalize_variant(variant)
        if item_method is None:
            item_method = self.last_item_method
        reasons: list[str] = []

        def movie_title(mid: int) -> str:
            if self.movie_lookup is None or int(mid) not in self.movie_lookup.index:
                return f"movie {mid}"
            return str(self.movie_lookup.loc[int(mid), "title"])

        if self.last_is_cold_start:
            reasons.append(str(self.last_rationale_label))
            reasons.append("Ranked by Bayesian popularity across all users")
        elif variant == VARIANT_USER and self.ratings_matrix is not None and user_id in self.ratings_matrix.index:
            similarity = self.user_similarity().loc[user_id].drop(index=user_id, errors="ignore")
            top_neighbors = similarity.sort_values(ascending=False).head(int(neighborhood_k))
            top_neighbors = top_neighbors[top_neighbors > 0]
            if movie_id in self.ratings_matrix.columns and not top_neighbors.empty:
                neighbor_ratings = self.ratings_matrix.loc[top_neighbors.index, movie_id]
                rated = neighbor_ratings.dropna()
                n_rated = int(rated.size)
                n_liked = int((rated >= 4.0).sum())
                if n_rated:
                    reasons.append(f"{n_rated} similar users rated this movie")
                if n_liked:
                    reasons.append(f"{n_liked} of them rated it 4★ or higher")
                if n_rated:
                    reasons.append(f"Avg. neighbor similarity: {float(top_neighbors.loc[rated.index].mean()):.2f}")
            if not reasons:
                reasons.append("User-based CF: similar users to you")
        elif variant == VARIANT_ITEM:
            # Explain via the single most similar movie the user already rated
            rated = self.user_train_ratings.get(user_id, {})
            item_sim = self.item_similarity(method=item_method)
            best: tuple[float, int, float] | None = None
            if movie_id in item_sim.index:
                for other_id, rating in rated.items():
                    if other_id not in item_sim.columns:
                        continue
                    sim = float(item_sim.loc[movie_id, other_id])
                    if best is None or sim > best[0]:
                        best = (sim, int(other_id), float(rating))
            if best is not None and best[0] >= 0.05:
                reasons.append(
                    f"Similar to '{movie_title(best[1])}' (you rated it {best[2]:.0f}★)"
                )
            method_label = "Pearson" if item_method == ITEM_METHOD_PEARSON else "cosine"
            reasons.append(f"Item-based CF ({method_label}): close to movies you already rated")
        else:
            # SVD: no per-neighbor explanation available
            reasons.append("SVD: matches your rating pattern")
            if self.ratings_matrix is not None and movie_id in self.ratings_matrix.columns:
                n_raters = int(self.ratings_matrix[movie_id].notna().sum())
                if n_raters >= 20:
                    reasons.append(f"Rated by {n_raters} users in training")

        if self._last_scores is not None and movie_id in self._last_scores.index:
            confidence = float(self._last_scores.loc[movie_id, "confidence"])
            reasons.append(f"Confidence {confidence:.2f}")

        rec_genres = self.movie_genre_sets.get(movie_id, set())
        liked_genres: set[str] = set()
        for mid, rating in self.user_train_ratings.get(user_id, {}).items():
            if rating < 4.0:
                continue
            liked_genres.update(self.movie_genre_sets.get(int(mid), set()))
        overlap = [genre for genre in rec_genres if genre in liked_genres][:3]
        return overlap + reasons

    def score_test_ratings(
        self, test_ratings: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Look up SVD-predicted ratings for every (user, movie) pair in a
        held-out test set, for RMSE/precision/recall evaluation"""
        if test_ratings.empty:
            return np.array([]), np.array([])
        pred_matrix = self.svd_predictions(n_components=self.n_factors)
        values = pred_matrix.to_numpy()
        user_index = {int(user_id): index for index, user_id in enumerate(pred_matrix.index)}
        movie_index = {int(movie_id): index for index, movie_id in enumerate(pred_matrix.columns)}
        actual = np.empty(len(test_ratings), dtype=float)
        predicted = np.empty(len(test_ratings), dtype=float)
        for index, row in enumerate(test_ratings.itertuples()):
            actual[index] = float(row.rating)
            ui = user_index.get(int(row.userId))
            mi = movie_index.get(int(row.movieId))
            predicted[index] = (
                float(values[ui, mi]) if ui is not None and mi is not None else self.global_mean
            )
        return actual, predicted

    @staticmethod
    def _classification_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        relevance_threshold: float,
        decision_threshold: float,
    ) -> dict[str, float]:
        """Turn continuous ratings into "liked"/"not liked" labels using
        the given thresholds, then compute standard classification metrics."""
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
        """Compute RMSE/MSE plus classification metrics. If no fixed
        decision_threshold is given, grid-search thresholds between 2.5 and
        4.2 and keep whichever gives the best F1-score."""
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
        """RMSE + liked/not-liked metrics plus raw arrays for plots (SVD)."""
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

    def split_train_test(self, test_size: float = 0.2, random_state: int = 42):
        """80/20 (by default) random split of the ratings for evaluation"""
        if self.ratings_df is None:
            raise RuntimeError("Collaborative model is not fitted.")
        return train_test_split(self.ratings_df, test_size=test_size, random_state=random_state)


def render_controls(movies: pd.DataFrame | None = None) -> tuple[str, list[str], int, str, int]:
    """Sidebar widgets matching the three CF variants and their extra knobs."""
    import streamlit as st

    from content_based import catalog_genres

    st.markdown("**Collaborative Filtering**")
    variant_label = st.radio(
        "Model variant",
        list(VARIANT_OPTIONS.keys()),
        index=0,
        key="cf_variant_label",
        help="User-based uses similar people, item-based uses similar movies, SVD uses latent factors.",
    )
    variant = VARIANT_OPTIONS[variant_label]
    genre_options = catalog_genres(movies) if movies is not None else []
    genres = st.multiselect(
        "Genre filter (optional)",
        options=genre_options,
        placeholder="Choose options",
        key="cf_genre_filter",
        help="Only recommend movies that include at least one selected genre.",
    )
    neighborhood_k = DEFAULT_NEIGHBORHOOD_K
    item_method = DEFAULT_ITEM_METHOD
    n_components = DEFAULT_N_COMPONENTS
    # Only show the neighborhood-size slider for the two kNN-style variants
    if variant in {VARIANT_USER, VARIANT_ITEM}:
        neighborhood_k = int(
            st.slider(
                "Neighborhood size (k)",
                min_value=5,
                max_value=80,
                value=DEFAULT_NEIGHBORHOOD_K,
                step=5,
                key="cf_neighborhood_k",
                help="How many similar users (user-based) or similar movies (item-based) to use.",
            )
        )
    # Only show the similarity-metric choice for item-based CF
    if variant == VARIANT_ITEM:
        item_label = st.radio(
            "Item similarity",
            list(ITEM_METHOD_OPTIONS.keys()),
            index=0,
            key="cf_item_method_label",
            help="Pearson (adjusted cosine) centers each movie by its mean rating first.",
        )
        item_method = ITEM_METHOD_OPTIONS[item_label]
    # Only show the latent-factor slider for SVD
    if variant == VARIANT_SVD:
        n_components = int(
            st.slider(
                "Latent factors (SVD)",
                min_value=5,
                max_value=50,
                value=DEFAULT_N_COMPONENTS,
                step=5,
                key="cf_n_components",
                help="More factors capture finer rating patterns; 20 is a solid default.",
            )
        )
    _LAST_CONTROLS["item_method"] = item_method
    _LAST_CONTROLS["n_components"] = n_components
    return variant, list(genres), neighborhood_k, item_method, n_components

# Load the cleaned ratings and movies CSVs produced by preprocessing
def load_data(processed_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = processed_dir or _find_processed_dir()
    ratings = pd.read_csv(processed_dir / "ratings_clean.csv")
    movies = pd.read_csv(processed_dir / "movies_clean.csv")
    return ratings, movies

# Quick manual smoke-test
def demo(user_id: int = 1, top_n: int = 10, n_factors: int = DEFAULT_N_COMPONENTS) -> pd.DataFrame:
    ratings, movies = load_data()
    model = CollaborativeFiltering(n_factors=n_factors).fit(ratings, movies=movies)
    return model.recommend(user_id, n_recommendations=top_n, variant=VARIANT_USER)

def parse_args() -> argparse.Namespace:
    """CLI arguments for running this module directly, e.g.:
    `py collaborative_filtering.py --user-id 5 --model "Item-Based CF"`"""
    parser = argparse.ArgumentParser(description="Run collaborative filtering.")
    parser.add_argument("--user-id", type=int, default=1, help="User id to recommend for.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations.")
    parser.add_argument(
        "--n-factors",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help="Number of latent factors used by SVD.",
    )
    parser.add_argument(
        "--model",
        choices=list(VARIANT_OPTIONS.keys()),
        default="User-Based CF",
        help="Which CF variant to run.",
    )
    return parser.parse_args()

# Load data, fit the model, print recommendations
if __name__ == "__main__":
    args = parse_args()
    ratings, movies = load_data()
    model = CollaborativeFiltering(n_factors=args.n_factors).fit(ratings, movies=movies)
    results = model.recommend(
        args.user_id,
        n_recommendations=args.top_n,
        variant=args.model,
    )
    print(f"{args.model} recommendations for user {args.user_id}:")
    print(results.to_string(index=False))
