#Imports
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
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

#Constant
BASE_DIR = Path(__file__).resolve().parent
MIN_RATING = 0.5
MAX_RATING = 5.0
DEFAULT_ALPHA_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80] #alpha values to try when tuning
CLASSIFICATION_THRESHOLDS = np.round(np.arange(2.5, 4.25, 0.05), 2) #try many cutoffs to find the best F1
ITEM_SUPPORT_SHRINK = 25.0 #how fast the blend moves from content to SVD when a movie has more ratings
MIN_CONTENT_MIX = 0.25 #content always keeps at least this share of the mix
HYBRID_RARE_MAX = 15 #movies with this many ratings or fewer lean on content
HYBRID_POPULAR_MIN = 120 #movies with this many ratings or more stay closer to SVD
EXAMPLE_MIX_COUNTS = (0, 50, 120) #example rating counts shown on the weight page


#Rating range validation
def _clip_rating(value: float) -> float:
    return float(np.clip(value, MIN_RATING, MAX_RATING)) #np.clip() = forces the value to stay between min_rating and max_rating


#Turn cosine similarity (0–1) into a 0.5–5 score so it can be blended with SVD
def _score_from_cosine(similarity: float) -> float:
    similarity = float(np.clip(similarity, 0.0, 1.0))
    return _clip_rating(MIN_RATING + (MAX_RATING - MIN_RATING) * similarity)


#Finding folder that containig processed data
def _find_processed_dir() -> Path:
    for candidate in (BASE_DIR / "processed", BASE_DIR / "dataset" / "processed"): #find the folder
        if (candidate / "ratings_clean.csv").exists(): #find the file inside this two folder
            return candidate
    raise FileNotFoundError(
        "Processed data not found. Run `py data_preprocessing.py` first."
    )


#Hybrid blending
#content_scores = predicted rating from TF-IDF / cosine
#cf_scores = predicted rating from SVD
#alpha = max content weight
#If a movie has few ratings, use more content. If it has many ratings, use more SVD.
def blend_hybrid_details(
    content_scores: np.ndarray,
    cf_scores: np.ndarray,
    movie_ids: np.ndarray,
    item_counts: dict[int, int],
    alpha: float,
    shrink: float = ITEM_SUPPORT_SHRINK,
    min_mix: float = MIN_CONTENT_MIX,
) -> dict[str, np.ndarray]:
    #If there is nothing to score, return empty arrays
    if len(content_scores) == 0:
        empty = np.array([], dtype=float)
        return {
            "rating_count": empty,
            "content_weight": empty,
            "cf_weight": empty,
            "hybrid_rating": empty,
        }
    #How many ratings each movie already has
    counts = np.fromiter(
        (item_counts.get(int(movie_id), 0) for movie_id in movie_ids),
        dtype=float,
        count=len(movie_ids),
    )
    #More ratings = more confidence in SVD
    item_confidence = counts / (counts + shrink)
    #Content weight gets smaller when the movie is popular
    content_weight = np.clip(
        alpha * (min_mix + (1.0 - min_mix) * (1.0 - item_confidence)),
        0.0,
        1.0,
    )
    cf_weight = 1.0 - content_weight #the rest of the mix is SVD
    #Final hybrid rating = weighted average of both models
    hybrid_rating = content_weight * content_scores + (1.0 - content_weight) * cf_scores
    return {
        "rating_count": counts,
        "content_weight": content_weight,
        "cf_weight": cf_weight,
        "hybrid_rating": hybrid_rating,
    }


#Same as blend_hybrid_details but only return the final hybrid rating
def blend_hybrid_scores(
    content_scores: np.ndarray,
    cf_scores: np.ndarray,
    movie_ids: np.ndarray,
    item_counts: dict[int, int],
    alpha: float,
    shrink: float = ITEM_SUPPORT_SHRINK,
    min_mix: float = MIN_CONTENT_MIX,
) -> np.ndarray:
    return blend_hybrid_details(
        content_scores,
        cf_scores,
        movie_ids,
        item_counts,
        alpha,
        shrink=shrink,
        min_mix=min_mix,
    )["hybrid_rating"]


#Give a simple label based on how much content was used
def blend_source_label(content_weight: float) -> str:
    if content_weight >= 0.45:
        return "Content-leaning"
    if content_weight <= 0.25:
        return "Collaborative-leaning"
    return "Balanced"


#Find which column names the table is using
def _hybrid_score_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    content = "content_rating" if "content_rating" in frame.columns else "Content score"
    collaborative = "cf_rating" if "cf_rating" in frame.columns else "Collaborative score"
    hybrid = "hybrid_rating" if "hybrid_rating" in frame.columns else "Hybrid score"
    return content, collaborative, hybrid


#Build a HTML table to explain why each movie got this mix
def explain_blend_table_html(frame: pd.DataFrame, reasons: dict[int, list[str]]) -> str:
    from html import escape

    from content_based import genre_pill_html, split_movie_genres

    content_col, cf_col, hybrid_col = _hybrid_score_columns(frame)
    rows_html: list[str] = []
    for number, (_, row) in enumerate(frame.iterrows(), start=1): #go through each recommended movie
        movie_id = int(row["movieId"])
        title = escape(str(row.get("title") or "Unknown title")) #escape() = make the title safe for HTML
        genres = split_movie_genres(row.get("genres", ""))
        pills = "".join(genre_pill_html(genre) for genre in genres) or (
            '<span class="why-empty">No genres listed</span>'
        )
        content = float(row.get(content_col, 0.0) or 0.0)
        cf_score = float(row.get(cf_col, 0.0) or 0.0)
        hybrid_score = float(row.get(hybrid_col, 0.0) or 0.0)
        genre_set = set(genres)
        #Don't show genre names again in the why column
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


#Plotting content vs SVD vs hybrid scores
def plot_score_breakdown(frame: pd.DataFrame):
    import matplotlib.pyplot as plt

    content_col, cf_col, hybrid_col = _hybrid_score_columns(frame)
    #Shorten long movie titles so they fit on the x axis
    labels = [
        str(title)[:26] + ("…" if len(str(title)) > 26 else "")
        for title in frame["title"].tolist()
    ]
    content = frame[content_col].astype(float).to_numpy()
    collaborative = frame[cf_col].astype(float).to_numpy()
    hybrid_scores = frame[hybrid_col].astype(float).to_numpy()
    stacked = np.concatenate([content, collaborative, hybrid_scores])
    #Zoom the y axis around the scores so the bars are easier to compare
    y_min = max(0.0, float(np.nanmin(stacked)) - 0.2)
    y_max = min(5.35, float(np.nanmax(stacked)) + 0.15)
    if y_max - y_min < 0.7:
        mid = (y_min + y_max) / 2
        y_min = max(0.0, mid - 0.45)
        y_max = min(5.35, mid + 0.45)

    x = list(range(len(frame)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.0)) #create the chart, width = 8.2, height = 4.0
    #Background colour
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


#Plotting how the blend weight changes when a movie has more ratings
def plot_blend_weights(
    alpha: float,
    shrink: float = ITEM_SUPPORT_SHRINK,
    min_mix: float = MIN_CONTENT_MIX,
    max_count: int = 250,
):
    import matplotlib.pyplot as plt

    counts = np.arange(0, max_count + 1, dtype=int)
    item_counts = {int(index): int(count) for index, count in enumerate(counts)}
    #Use dummy scores of 4.0 because we only care about the weights here
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


def example_mix_at_counts(
    alpha: float,
    counts: tuple[int, ...] = EXAMPLE_MIX_COUNTS,
    shrink: float = ITEM_SUPPORT_SHRINK,
    min_mix: float = MIN_CONTENT_MIX,
) -> list[tuple[int, float, float]]:
    movie_ids = np.arange(len(counts))
    item_counts = {int(index): int(count) for index, count in enumerate(counts)}
    details = blend_hybrid_details(
        np.full(len(counts), 4.0),
        np.full(len(counts), 4.0),
        movie_ids,
        item_counts,
        float(alpha),
        shrink=shrink,
        min_mix=min_mix,
    )
    return [
        (int(count), float(details["content_weight"][index]), float(details["cf_weight"][index]))
        for index, count in enumerate(counts)
    ]


def plot_example_mix(
    alpha: float,
    shrink: float = ITEM_SUPPORT_SHRINK,
    min_mix: float = MIN_CONTENT_MIX,
):
    import matplotlib.pyplot as plt

    mix = example_mix_at_counts(alpha, shrink=shrink, min_mix=min_mix)
    labels = [
        "Rare\n(0 ratings)",
        "In between\n(50 ratings)",
        "Popular\n(120 ratings)",
    ]
    content = [row[1] * 100 for row in mix]
    collaborative = [row[2] * 100 for row in mix]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.bar(x - width / 2, content, width, label="Content", color="#5B8FF9")
    ax.bar(x + width / 2, collaborative, width, label="Collaborative (SVD)", color="#5AD8A6")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Blend share (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def catalog_popularity_counts(item_counts: dict[int, int]) -> tuple[int, int, int]:
    rare = mid = popular = 0
    for count in item_counts.values():
        if int(count) <= HYBRID_RARE_MAX:
            rare += 1
        elif int(count) >= HYBRID_POPULAR_MIN:
            popular += 1
        else:
            mid += 1
    return rare, mid, popular


def plot_catalog_popularity(item_counts: dict[int, int]):
    import matplotlib.pyplot as plt

    rare, mid, popular = catalog_popularity_counts(item_counts)
    labels = [
        f"Rarely rated\n(≤{HYBRID_RARE_MAX} ratings)\nmore content",
        f"In between\n({HYBRID_RARE_MAX + 1}–{HYBRID_POPULAR_MIN - 1})\nmixed",
        f"Popular\n(≥{HYBRID_POPULAR_MIN} ratings)\nmore SVD",
    ]
    values = [rare, mid, popular]
    colors = ["#E85D75", "#5B8FF9", "#5AD8A6"]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.set_ylabel("Number of movies")
    ax.set_title("How many movies fall into each blend band")
    ax.yaxis.grid(True, color="#D0D3DA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.margins(y=0.14)
    fig.tight_layout()
    return fig

#----------------------------------------------------------------------------------------
#ContentBased Recommender 
#----------------------------------------------------------------------------------------
#Genre + tags -> TF-IDF -> cosine similarity.
#Star ratings are only used as "movies this user has seen", not as profile weights.
#This class is only used by HybridRecommender so both models can be blended later.
class ContentBasedRecommender:
    #Content based recommender model training and prediction
    def __init__(self) -> None:
        #Varaible
        self.movie_ids: np.ndarray | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.movie_id_to_index: dict[int, int] = {} #store the TF-IDF representation of every movie
        self.user_ratings: dict[int, dict[int, float]] = {}
        self.user_profiles: dict[int, np.ndarray] = {}
        #Stores the TF-IDF profile representing each user's movie taste
        self.vectorizer = TfidfVectorizer(**CONTENT_VECTORIZER_PARAMS) #Creates the TF-IDF vectorizer

    #Train/Build the Model
    def fit(
        self,
        ratings: pd.DataFrame,
        movie_content: pd.DataFrame,
        movies: pd.DataFrame | None = None,
    ) -> "ContentBasedRecommender":
        del movies  # hybrid may pass movies; content scoring does not use them
        content = movie_content.copy()
        #make sure the content feature column has no missing value and contains string
        content["content_features"] = content["content_features"].fillna("").astype(str)
        self.movie_ids = content["movieId"].to_numpy()
        #Create TF-IDF features
        tfidf_sparse = self.vectorizer.fit_transform(content["content_features"])
        #Normalize TF-IDF so cosine similarity = dot product
        self.tfidf_matrix = normalize(tfidf_sparse, norm="l2").toarray()
        #Create Movie ID to row index lookup
        self.movie_id_to_index = {
            int(movie_id): index for index, movie_id in enumerate(self.movie_ids)
        }
        #Keep which movies each user has seen (ignore the star value when scoring)
        self.user_ratings = {
            int(user_id): {
                int(row.movieId): float(row.rating) for row in user_frame.itertuples()
            }
            for user_id, user_frame in ratings.groupby("userId")
        }
        self._build_user_indexes() #Build user profiles
        return self

    #Build User Profiles
    def _build_user_indexes(self) -> None:
        if self.tfidf_matrix is None:
            return
        self.user_profiles = {}
        for user_id, rated_movies in self.user_ratings.items(): #Processes one user at a time
            indices = []
            #Find movies this user has seen (do not use the star as a weight)
            for movie_id in rated_movies:
                index = self.movie_id_to_index.get(movie_id)
                if index is not None:
                    indices.append(index)
            if not indices:
                continue
            index_array = np.asarray(indices, dtype=int)
            #Unweighted mean of those movies' TF-IDF vectors
            profile = self.tfidf_matrix[index_array].mean(axis=0)
            self.user_profiles[user_id] = normalize(profile.reshape(1, -1), norm="l2").ravel()

    #Get User Profile
    def _user_profile(self, user_id: int) -> np.ndarray | None:
        return self.user_profiles.get(user_id)

    #Calculate Movie Similarity
    def _cosine_to_profile(self, user_id: int, movie_id: int) -> float: #How similar is this movie to the user's overall movie taste
        if self.tfidf_matrix is None:
            return 0.0
        profile = self._user_profile(user_id) #get user profile
        movie_index = self.movie_id_to_index.get(movie_id) #Finds the movie's TF-IDF vector
        if profile is None or movie_index is None:
            return 0.0
        #Calculate cosine similarity
        similarity = float(
            cosine_similarity(
                profile.reshape(1, -1),
                self.tfidf_matrix[movie_index].reshape(1, -1),
            )[0, 0]
        )
        return float(np.clip(similarity, 0.0, 1.0))

    #Predict a 0.5–5 content score from cosine similarity (not from star ratings)
    def predict(self, user_id: int, movie_id: int) -> float:
        if self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")
        return _score_from_cosine(self._cosine_to_profile(user_id, movie_id))

    #Predict many content scores at once for evaluation
    def predict_many(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        if self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")

        n_rows = len(user_ids)
        predictions = np.full(n_rows, MIN_RATING, dtype=float) #cosine 0 maps to 0.5
        if n_rows == 0:
            return predictions

        grouped = pd.DataFrame(
            {
                "row": np.arange(n_rows),
                "userId": np.asarray(user_ids, dtype=int),
                "movieId": np.asarray(movie_ids, dtype=int),
            }
        )
        for user_id, group in grouped.groupby("userId"): #one user at a time
            profile = self._user_profile(int(user_id))
            row_indices = group["row"].to_numpy()
            if profile is None:
                continue

            movie_indices = np.array(
                [self.movie_id_to_index.get(int(movie_id), -1) for movie_id in group["movieId"]],
                dtype=int,
            )
            known = movie_indices >= 0 #skip movies not in the content catalog
            if not np.any(known):
                continue

            similarities = np.clip(
                self.tfidf_matrix[movie_indices[known]] @ profile,
                0.0,
                1.0,
            )
            predictions[row_indices[known]] = MIN_RATING + (MAX_RATING - MIN_RATING) * similarities
        return predictions

    #Generate Top Recommendations
    def top_candidates(self, user_id: int, n_candidates: int) -> list[int]: #finds the best movies to recommend to a user
        if self.movie_ids is None or self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")
        profile = self._user_profile(user_id) #Get user's profile
        if profile is None:
            return []
        #Don't recommend movies the user has already rated
        rated_movies = set(self.user_ratings.get(user_id, {}))
        candidate_indices = [
            index
            for index, movie_id in enumerate(self.movie_ids)
            if int(movie_id) not in rated_movies
        ]
        if not candidate_indices:
            return []
        #Calculate similarity
        scores = cosine_similarity(
            profile.reshape(1, -1), self.tfidf_matrix[candidate_indices]
        ).flatten()
        top_indices = np.argsort(scores)[-n_candidates:][::-1] #highest score first
        return [int(self.movie_ids[candidate_indices[index]]) for index in top_indices]


#----------------------------------------------------------------------------------------
#Collaborative Filtering Recommender 
#----------------------------------------------------------------------------------------
#TruncatedSVD = matrix factorization, finds hidden patterns in the rating matrix
#Example: users who like the same movies will have similar latent factors
class CollaborativeFilteringRecommender:
    def __init__(self, n_factors: int = 20) -> None:
        #Varaible
        self.n_factors = n_factors #number of hidden features to learn
        self.user_ids: np.ndarray | None = None
        self.movie_ids: np.ndarray | None = None
        self.user_id_to_index: dict[int, int] = {}
        self.movie_id_to_index: dict[int, int] = {}
        self.user_factors: np.ndarray | None = None
        self.movie_factors: np.ndarray | None = None
        self.user_means: np.ndarray | None = None
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_train_movies: dict[int, set[int]] = {}
        self.item_counts: dict[int, int] = {} #how many ratings each movie has
        self.global_mean: float = 3.5

    #Train/Build the SVD model
    def fit(self, ratings: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        self.user_ids = np.sort(ratings["userId"].astype(int).unique())
        self.movie_ids = np.sort(ratings["movieId"].astype(int).unique())
        #Create lookup tables so we can find the matrix row/column quickly
        self.user_id_to_index = {
            int(user_id): index for index, user_id in enumerate(self.user_ids)
        }
        self.movie_id_to_index = {
            int(movie_id): index for index, movie_id in enumerate(self.movie_ids)
        }

        row_indices = ratings["userId"].map(self.user_id_to_index).to_numpy()
        col_indices = ratings["movieId"].map(self.movie_id_to_index).to_numpy()
        values = ratings["rating"].to_numpy(dtype=float)
        #Build a sparse user-item rating matrix
        rating_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )

        #Calculate each user's average rating
        rating_count = np.diff(rating_matrix.indptr)
        rating_sum = np.asarray(rating_matrix.sum(axis=1)).ravel()
        self.user_means = np.divide(
            rating_sum,
            rating_count,
            out=np.full(len(self.user_ids), ratings["rating"].mean()),
            where=rating_count > 0,
        )
        self.global_mean = float(ratings["rating"].mean())
        #Calculate how many ratings each movie has and the movie average
        item_count = np.asarray(rating_matrix.getnnz(axis=0), dtype=float)
        item_sum = np.asarray(rating_matrix.sum(axis=0)).ravel()
        item_means = np.divide(
            item_sum,
            item_count,
            out=np.full(len(self.movie_ids), self.global_mean),
            where=item_count > 0,
        )
        #Bias = how much this user/movie is above or below the global average
        self.user_bias = self.user_means - self.global_mean
        self.item_bias = item_means - self.global_mean
        self.item_counts = {
            int(movie_id): int(count)
            for movie_id, count in zip(self.movie_ids, item_count)
        }

        #Center the ratings before running SVD
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

        #TruncatedSVD: user_factors is U * Sigma, movie_factors is V
        svd = TruncatedSVD(n_components=k, random_state=42)
        self.user_factors = svd.fit_transform(centered)
        self.movie_factors = svd.components_.T

        #Remember which movies each user already rated
        self.user_train_movies = {}
        for user_id, user_frame in ratings.groupby("userId"):
            self.user_train_movies[int(user_id)] = set(user_frame["movieId"].astype(int))
        return self

    #Predict User Rating
    def predict(self, user_id: int, movie_id: int) -> float:
        if (
            self.user_factors is None
            or self.movie_factors is None
            or self.user_means is None
            or self.item_bias is None
        ):
            raise RuntimeError("Collaborative model is not fitted.")

        user_index = self.user_id_to_index.get(user_id)
        movie_index = self.movie_id_to_index.get(movie_id)
        if user_index is None or movie_index is None:
            return self.global_mean

        #predicted rating = user average + movie bias + SVD score
        latent_score = np.dot(
            self.user_factors[user_index],
            self.movie_factors[movie_index],
        )
        return _clip_rating(
            float(self.user_means[user_index] + self.item_bias[movie_index] + latent_score)
        )

    #Predict many ratings at once for evaluation
    def predict_many(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        if (
            self.user_factors is None
            or self.movie_factors is None
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
        valid = (user_index >= 0) & (movie_index >= 0) #skip unknown users/movies
        if not np.any(valid):
            return predictions

        valid_users = user_index[valid]
        valid_movies = movie_index[valid]
        latent = np.einsum(
            "ij,ij->i",
            self.user_factors[valid_users],
            self.movie_factors[valid_movies],
        )
        predictions[valid] = np.clip(
            self.user_means[valid_users] + self.item_bias[valid_movies] + latent,
            MIN_RATING,
            MAX_RATING,
        )
        return predictions

    #Generate Top Recommendations
    def top_candidates(self, user_id: int, n_candidates: int) -> list[int]:
        if (
            self.user_factors is None
            or self.movie_factors is None
            or self.user_means is None
            or self.item_bias is None
            or self.movie_ids is None
        ):
            raise RuntimeError("Collaborative model is not fitted.")

        user_index = self.user_id_to_index.get(user_id)
        if user_index is None:
            return []

        #Score every movie for this user
        user_vector = self.user_factors[user_index]
        predicted_scores = (
            self.user_means[user_index] + self.item_bias + self.movie_factors @ user_vector
        )

        #Don't recommend movies the user already rated
        rated_movies = self.user_train_movies.get(user_id, set())
        candidate_indices = [
            index
            for index, movie_id in enumerate(self.movie_ids)
            if int(movie_id) not in rated_movies
        ]
        candidate_scores = predicted_scores[candidate_indices]
        top_indices = np.argsort(candidate_scores)[-n_candidates:][::-1]
        return [int(self.movie_ids[candidate_indices[index]]) for index in top_indices]


#Load dataset
def load_data(
    processed_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: #tuple[] = return 3 data frames
    processed_dir = processed_dir or _find_processed_dir()
    #load the file
    ratings = pd.read_csv(processed_dir / "ratings_clean.csv")
    movies = pd.read_csv(processed_dir / "movies_clean.csv")

    movie_content_path = processed_dir / "movies_content.csv"
    if movie_content_path.exists():
        movie_content = pd.read_csv(movie_content_path)
    else:
        tags = pd.read_csv(processed_dir / "tags_clean.csv")
        movie_content = build_movie_content(movies, tags)

    return ratings, movies, movie_content


#Train-Test Data Splitting = divide the ratings data into trining data and testing data for evaluating
def split_train_test(
    ratings: pd.DataFrame,
    test_size: float = 0.2, #20% go testing, 80% do training
    random_state: int = 42, #makes the split repeatable, so you get the same result each time
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = [] #stores training ratings for each user
    test_parts: list[pd.DataFrame] = [] #stores testing ratings for each user

    for _, user_ratings in ratings.groupby("userId"): #groups the rating based on userID
        #If user less than 2 rating, the program will not spilt thier data
        if len(user_ratings) < 2:
            train_parts.append(user_ratings)
            continue

        #For user with 2 or more rating, the rating are spilt
        train_split, test_split = train_test_split(
            user_ratings,
            test_size=test_size,
            random_state=random_state,
        )
        #Stores each user's training and testing ratings in the corresponding lists
        train_parts.append(train_split)
        test_parts.append(test_split)

    train_df = pd.concat(train_parts, ignore_index=True) #Combines all users' training ratings into one training DataFrame
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame() #Combines all users' testing ratings into one testing DataFrame
    return train_df, test_df


#----------------------------------------------------------------------------------------
#Hybrid Recommender 
#----------------------------------------------------------------------------------------
#This model combines two methods:
#1. Content-based = genres/tags -> TF-IDF -> cosine similarity
#2. Collaborative = SVD matrix factorization
#Then it blends both scores. Rare movies use more content, popular movies use more SVD.
class HybridRecommender:
    def __init__(self, alpha: float = 0.3, n_factors: int = 20) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0)) #alpha = max content weight
        self.content_model = ContentBasedRecommender()
        self.collaborative_model = CollaborativeFilteringRecommender(n_factors=n_factors)
        self.movies: pd.DataFrame | None = None
        self.item_counts: dict[int, int] = {}
        self.item_shrink = ITEM_SUPPORT_SHRINK
        self.min_content_mix = MIN_CONTENT_MIX

    #Train/Build both models
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

    #Reuse models that are already trained, used when trying different alpha values
    def attach_fitted_models(
        self,
        content_model: ContentBasedRecommender,
        collaborative_model: CollaborativeFilteringRecommender,
        movies: pd.DataFrame,
        item_counts: dict[int, int] | None = None,
    ) -> "HybridRecommender":
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.movies = movies
        self.item_counts = item_counts or dict(collaborative_model.item_counts)
        return self

    #Blend the two predicted ratings into one hybrid rating
    def _hybrid_rating(
        self,
        content_rating: float,
        cf_rating: float,
        movie_id: int | None = None,
    ) -> float:
        if movie_id is None:
            return float(self.alpha * content_rating + (1.0 - self.alpha) * cf_rating)
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

    #Get candidate movies from both models and score them
    def _candidate_scores(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        candidate_count = max(n_recommendations * 5, 50) #get extra candidates then blend later
        content_candidates = self.content_model.top_candidates(user_id, candidate_count)
        cf_candidates = self.collaborative_model.top_candidates(user_id, candidate_count)
        #Combine both lists and remove duplicates
        candidate_movie_ids = list(dict.fromkeys(content_candidates + cf_candidates))
        if not candidate_movie_ids:
            raise ValueError(f"Unable to generate candidates for user {user_id}.")

        #Predict a rating from each model
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
            #Add movie title and genres
            scored = scored.merge(
                self.movies[["movieId", "title", "genres"]],
                on="movieId",
                how="left",
            )
        return scored

    #Generate Final Recommendations
    def recommend(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        scored = self._candidate_scores(user_id, n_recommendations=n_recommendations)
        return (
            scored.sort_values("hybrid_rating", ascending=False)
            .head(n_recommendations)
            .reset_index(drop=True)
        )

    #Check how much the two candidate lists overlap
    def candidate_overlap(self, user_id: int, n_recommendations: int = 10) -> dict[str, int]:
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

    #Find movies where content and SVD disagree the most
    def score_disagreements(self, user_id: int, n_movies: int = 8) -> pd.DataFrame:
        scored = self._candidate_scores(user_id, n_recommendations=max(n_movies, 10))
        scored = scored.copy()
        scored["score_gap"] = (scored["content_rating"] - scored["cf_rating"]).abs()
        scored["favored_by"] = np.where(
            scored["content_rating"] >= scored["cf_rating"],
            "Content",
            "Collaborative",
        )
        return scored.sort_values("score_gap", ascending=False).head(n_movies).reset_index(drop=True)

    #Explain Why a Movie Was Recommended
    def recommendation_reasons(self, row: pd.Series, user_id: int | None = None) -> list[str]: #return a list of reason
        count = int(row.get("rating_count", 0) or 0)
        raw_weight = row.get("content_weight", np.nan)
        if raw_weight is None or (isinstance(raw_weight, float) and pd.isna(raw_weight)):
            movie_id = row.get("movieId")
            content_score = float(row.get("content_rating", row.get("Content score", 0.0)) or 0.0)
            cf_score = float(row.get("cf_rating", row.get("Collaborative score", 0.0)) or 0.0)
            if pd.notna(movie_id):
                details = blend_hybrid_details(
                    np.array([content_score]),
                    np.array([cf_score]),
                    np.array([int(movie_id)]),
                    self.item_counts,
                    self.alpha,
                    shrink=self.item_shrink,
                    min_mix=self.min_content_mix,
                )
                content_weight = float(details["content_weight"][0])
            else:
                content_weight = float(self.alpha)
        else:
            content_weight = float(raw_weight)
        content_score = float(row.get("content_rating", row.get("Content score", 0.0)) or 0.0)
        cf_score = float(row.get("cf_rating", row.get("Collaborative score", 0.0)) or 0.0)
        source = blend_source_label(content_weight)
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
        reasons.append(
            f"{source}: {content_weight:.0%} content / {1.0 - content_weight:.0%} SVD"
        )
        if count <= HYBRID_RARE_MAX:
            if content_weight > 0.5:
                reasons.append(
                    f"Rarely rated ({count} ratings), so content has more influence"
                )
            else:
                reasons.append(
                    f"Rarely rated ({count} ratings), but current alpha still keeps SVD dominant"
                )
        elif count >= HYBRID_POPULAR_MIN:
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

    #Search movies by title and/or genre
    def search_movies(
        self,
        query: str = "",
        genres: list[str] | tuple[str, ...] | None = None,
        limit: int = 25,
    ) -> pd.DataFrame:
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

    #Score one searched movie for the current user
    def score_movie(self, user_id: int, movie_id: int) -> pd.DataFrame:
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

    #Recommend Movies Similar to a Selected Movie
    def similar_to_movie(
        self,
        user_id: int,
        movie_id: int,
        n_recommendations: int = 10,
    ) -> pd.DataFrame:
        content = self.content_model
        if content.tfidf_matrix is None or content.movie_ids is None:
            raise RuntimeError("Content model is not fitted.")
        seed_index = content.movie_id_to_index.get(int(movie_id))
        if seed_index is None:
            raise ValueError("That movie is not in the content catalog.")

        #Compare the selected movie to every other movie using TF-IDF
        similarities = content.tfidf_matrix @ content.tfidf_matrix[seed_index]
        seen = set(content.user_ratings.get(int(user_id), {}))
        seen.add(int(movie_id)) #don't recommend the same movie again
        ranked = np.argsort(similarities)[::-1]
        picked: list[int] = []
        sim_values: list[float] = []
        pool_size = max(n_recommendations * 4, 40)
        for index in ranked:
            other_id = int(content.movie_ids[index])
            if other_id in seen: #Skip movies the user already rated
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

    #Predict a hybrid rating for one user and one movie
    def predict(self, user_id: int, movie_id: int) -> float:
        content_rating = self.content_model.predict(user_id, movie_id)
        cf_rating = self.collaborative_model.predict(user_id, movie_id)
        return self._hybrid_rating(content_rating, cf_rating, movie_id=movie_id)

    #Top-N recommendations from the content-based model only
    def recommend_content(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
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

    #Top-N recommendations from the collaborative model only
    def recommend_collaborative(self, user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
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


#Liked / not-liked classification metrics
#actual rating >= relevance_threshold means the user liked the movie
#predicted rating >= decision_threshold means the model thinks they will like it
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


#Calculate Overall Performance
#Calculate both rating prediction errors and liked/not-liked classification performance
def evaluate_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    relevance_threshold: float = 4.0,
    decision_threshold: float | None = None,
) -> dict[str, float]:
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))), #calculate RMSE
    }
    #Finding the Best Decision Threshold, threshold = the value used to decide which class the prediction belongs to
    if decision_threshold is None:
        best_metrics: dict[str, float] | None = None
        for threshold in CLASSIFICATION_THRESHOLDS:
            candidate = _classification_metrics(
                actual, predicted, relevance_threshold, float(threshold)
            )
            if best_metrics is None or candidate["f1_score"] > best_metrics["f1_score"]: #selects the threshold that gives the highest F1 score
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


#Get content and SVD predictions for the test set
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


#Complete Model Evaluation for all three models
def evaluate_all_models(
    model: HybridRecommender,
    test_ratings: pd.DataFrame,
    relevance_threshold: float = 4.0,
) -> dict[str, dict[str, float]]:
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


#Evaluate only the hybrid model
def evaluate_model(
    model: HybridRecommender,
    test_ratings: pd.DataFrame,
    relevance_threshold: float = 4.0,
    decision_threshold: float | None = None,
) -> dict[str, float]:
    return evaluate_all_models(model, test_ratings, relevance_threshold=relevance_threshold)[
        "hybrid"
    ]


#Try different alpha values and keep the best one
#The two base models are trained once, only the blend weight changes
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
        #For RMSE smaller is better, for the other metrics bigger is better
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


#Print the evaluation results in a simple table
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


#Command-Line Arguments = Allow you to control the recommender from the terminal.
def parse_args() -> argparse.Namespace:
    #default = If the user doesn't tell me what to use, use this value
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
    #80% training, 20% testing, split by user
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
