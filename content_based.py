#Imports
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

#Constant
BASE_DIR = Path(__file__).resolve().parent
MIN_RATING = 0.5
MAX_RATING = 5.0
CONTENT_NEIGHBOR_K = 40 #prediction can use 40 similar previous rated movie
CONTENT_SHRINKAGE = 12.0
ALGORITHM_KEY = "content" #identify content_based as content

#Genre colours
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

#Rating range validation
def _clip_rating(value: float) -> float:
    return float(np.clip(value, MIN_RATING, MAX_RATING)) #np.clip() = forces the value to stay between min_rating and max_rating

#Finding folder that containig processed data
def _find_processed_dir() -> Path:
    for candidate in (BASE_DIR / "processed", BASE_DIR / "dataset" / "processed"): #find the folder
        if (candidate / "ratings_clean.csv").exists(): #find the file inside this two folder
            return candidate
    raise FileNotFoundError(
        "Processed data not found. Run `py data_preprocessing.py` first."
    )

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

#Movie Genre Data Cleaning and Splitting
def split_movie_genres(value: object) -> list[str]:
    #check if the genre value is missing
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    #check whether the value is already inside the list
    if isinstance(value, list):
        parts = [str(item).strip() for item in value]
    else:
        text = str(value)
        parts = text.split("|")
        #cleanig step = remove extra spacesm empty valuem remove (no genre listed)
    return [part.strip() for part in parts if part.strip() and part.strip() != "(no genres listed)"]

#Cataloging Unique Movie Genres
def catalog_genres(movies: pd.DataFrame) -> list[str]:
    labels: set[str] = set() #set will automatically prevents duplicates
    for value in movies["genres"].dropna(): #dropna() = ignore missing genre value
        labels.update(split_movie_genres(value)) #clean and split genres
    return sorted(labels)

#User genre preference profile
def user_genre_profile(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    min_rating: float = 3.0,
) -> pd.DataFrame:
    #get the user good rated movies
    user = ratings.loc[
        (ratings["userId"] == int(user_id)) & (ratings["rating"] >= min_rating) #select user and rating >= 3.0
    ].copy()
    #return empty if rating less than 3.0
    if user.empty:
        return pd.DataFrame(columns=["genre", "share", "percent", "color"])
    #Combine rating with movie genre
    merged = user.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    weights: dict[str, float] = {}
    #Process each movie
    for row in merged.itertuples(): #Loops through each movie the user rated
        genres = split_movie_genres(getattr(row, "genres", None)) #Split the movie's genres
        #Ignore movies without genres
        if not genres:
            continue
        #Divide the rating between genres
        #Cth: If movie rating is 5, then the genre have 2 which is action and comedy, the rating will divide two 5/2 - 2.5, so the action and comedy rating is 2.5
        piece = float(row.rating) / len(genres)
        #Add the genre weight
        for genre in genres:
            weights[genre] = weights.get(genre, 0.0) + piece
    #Calculate total weight
    total = sum(weights.values())
    #Check whether the total is valid
    #IF no meaningful genre weight, return an empty data frame
    if total <= 0:
        return pd.DataFrame(columns=["genre", "share", "percent", "color"])

    rows = [
        {
            "genre": genre,
            "share": weight / total, #Calculate each genre's percentage
            "percent": 100.0 * weight / total,
            "color": GENRE_COLORS.get(genre, DEFAULT_GENRE_COLOR),
        }
        for genre, weight in weights.items()
    ]
    return (
        #Sort the results (from high preference to low)
        pd.DataFrame(rows).sort_values("percent", ascending=False).reset_index(drop=True)
    )

#Plotting User Genre Preferences
def plot_genre_profile(
    profile: pd.DataFrame, user_id: int, top_n: int = 10 #top 10 genre
):
    """Horizontal bar chart of a user's genre share (same colours as the pills)."""
    import matplotlib.pyplot as plt

    data = profile.head(top_n).iloc[::-1] #reverse the order
    fig, ax = plt.subplots(figsize=(8.6, 4.8)) #create the chart, width = 8.6, height = 4.8
    #Background colour
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    #If no data
    if data.empty:
        ax.set_title(f"User {user_id}'s Genre Preferences", color="#31333F") #show title
        ax.text(0.5, 0.5, "Not enough liked ratings", ha="center", color="#6B7280")
        ax.set_axis_off()
        fig.tight_layout()
        return fig
    #Create the bars
    bars = ax.barh(
        data["genre"],
        data["percent"],
        color=data["color"],
        height=0.66,
    )
    #Chart title
    ax.set_title(f"User {user_id}'s Genre Preferences", color="#31333F", pad=12) #creates a title based on the userID
    #Set the x-axis range
    xmax = max(25.0, float(data["percent"].max()) * 1.18) 
    ax.set_xlim(0, xmax)
    ax.tick_params(colors="#31333F", labelsize=10)
    #Remove unnecessary chart borders
    ax.spines[:].set_visible(False)
    #Add grid lines
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("")
    #Add percentage labels
    for bar, percent in zip(bars, data["percent"], strict=True):
        ax.text(
            bar.get_width() + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{percent:.0f}%",
            va="center",
            color="#31333F",
            fontsize=9,
        )
    #Return the chart
    fig.tight_layout()
    return fig

#Generating Genre Pills for the UI
def genre_pill_html(genre: str) -> str:
    from html import escape

    color = GENRE_COLORS.get(genre, DEFAULT_GENRE_COLOR) #look the genre colour in GENRE_COLORS
    return (
        f'<span class="genre-pill" style="background:{color}">{escape(genre)}</span>' 
    )

#Calculating “For You” Recommendation Percentage
def _for_you_percent(row) -> int:
    similarity = getattr(row, "similarity", None)
    if similarity is not None and not pd.isna(similarity):
        #converts similarity from a 0–1 scale to a 0–100% scale = cth: 0.85(100) = 85%
        return int(round(float(np.clip(similarity, 0.0, 1.0)) * 100))
    score = None # if similarity does not exist, create function called score
    for name in ("Score", "predicted_rating", "Hybrid_score", "hybrid_rating"):
        if hasattr(row, name):
            value = getattr(row, name)
            #if valid score is found
            if value is not None and not pd.isna(value):
                score = value
                break
    #convert rating score to percentage
    return int(round(float(score) / MAX_RATING * 100)) if score is not None else 0

#Generate the "Why Recommended?" table
#Create html table that display movie number, movie title, genre, rating(for you), average rating, why recommender?
def why_recommended_table_html(
    frame: pd.DataFrame,
    reasons: dict[int, list[str]],
    empty_why: str = "Matches your profile",
) -> str:
    #import html escape
    from html import escape
    #Create an empty list for a table rows
    rows_html: list[str] = []
    #Loop through recommended movies
    for number, row in enumerate(frame.itertuples(), start=1):
        movie_id = int(row.movieId)
        title = escape(str(getattr(row, "title", "Unknown title"))) #Get the movie title
        pills = "".join(genre_pill_html(genre) for genre in split_movie_genres(getattr(row, "genres", ""))) #Create genre pills
        #Calculate “For You” percentage
        for_you = _for_you_percent(row)
        #Get the average rating
        avg = getattr(row, "avg_rating", None)
        #Convert average rating into percentage
        avg_pct = int(round(float(avg) / MAX_RATING * 100)) if avg is not None and pd.notna(avg) else 0
        #Get recommendation reasons
        why_bits = reasons.get(movie_id, [])
        #Display the reasons
        why = "".join(
            f'<div class="why-line">✓ {escape(item)}</div>' for item in why_bits
        ) or f'<span class="why-empty">{escape(empty_why)}</span>'
        #Build each movie row
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
    #Create the table
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

#Generating Movie Title, Genre, and Recommendation Score Cells
def _title_genre_pct_cells(row, *, changed: bool = False) -> str: # * = changed
    from html import escape
    #Get the movie title
    title = escape(str(getattr(row, "title", "Unknown title")))
    #Create genre pills
    pills = "".join(genre_pill_html(genre) for genre in split_movie_genres(getattr(row, "genres", "")))
    #Calculate the “For You” percentage
    pct = _for_you_percent(row)
    #Decide the title CSS class, title-cell is CSS class
    title_class = "title-cell changed" if changed else "title-cell"
    return (
        f"<td class='{title_class}'>{title}</td>"
        f"<td class='genre-cell'>{pills}</td>"
        f"<td class='pct-cell'><span>{pct}%</span>"
        f"<div class='bar'><div class='fill fill-you' style='width:{pct}%'></div></div></td>"
    )

#Before-and-After Diversity Recommendation Comparison
def _empty_cells() -> str: #return html table cell
    return (
        "<td class='title-cell'>—</td><td class='genre-cell'></td>"
        "<td class='pct-cell'><span>—</span></td>"
    )

#Compare recommendations before and after diversity
def diversity_comparison_table_html(
    original: pd.DataFrame, #recommend before diversity
    diversified: pd.DataFrame, #recommend after diversity
) -> str:
    #Convert both DataFrames into rows
    original_rows = list(original.itertuples())
    diversified_rows = list(diversified.itertuples())
    n = max(len(original_rows), len(diversified_rows)) #Find the larger list, cth: before = 10, after = 5, n=10
    rows_html: list[str] = []
    #Loop through each recommendation position
    for index in range(n):
        #Get the original and diversity movie
        orig_row = original_rows[index] if index < len(original_rows) else None
        div_row = diversified_rows[index] if index < len(diversified_rows) else None
        #Get movie IDs
        orig_id = int(orig_row.movieId) if orig_row is not None else None
        div_id = int(div_row.movieId) if div_row is not None else None
        #Detect changed recommendations
        changed = orig_id is not None and div_id is not None and orig_id != div_id
        #Create the left side (before diversity)
        left = _title_genre_pct_cells(orig_row) if orig_row is not None else _empty_cells()
        #Create the right side (after diversity)
        right = (
            _title_genre_pct_cells(div_row, changed=changed) if div_row is not None else _empty_cells()
        )
        #Create each table row
        rows_html.append(f"<tr><td class='num'>{index + 1}</td>{left}{right}</tr>")

    return (
        "<table class='why-table diversity-table'>"
        "<thead>"
        "<tr>"
        "<th rowspan='2'>No.</th>"
        "<th colspan='3' class='grp-orig'>Before diversity</th>"
        "<th colspan='3' class='grp-div'>After diversity</th>"
        "</tr>"
        "<tr>"
        "<th>Movie</th><th>Genres</th><th class='you'>For You</th>"
        "<th>Movie</th><th>Genres</th><th class='you'>For You</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )

#ContentBasedRecommender class
#TF-IDF = Uses movie genres and tags to represent movie content numerically.
#Cosine similarity = Measures how similar two movies are based on their content.
#MMR diversification = MMR (Maximal Marginal Relevance) can select movies that are both: relevant to the user, different from each other
#This helps reduce recommendations that are all too similar.
class ContentBasedRecommender:
    #set minimum and maximum
    RATING_MIN = MIN_RATING
    RATING_MAX = MAX_RATING
    
    #Content based recommender model training and prediction
    def __init__(self) -> None:
        #Varaible
        self.movie_ids: np.ndarray | None = None
        self.tfidf_matrix: np.ndarray | None = None
        self.movie_id_to_index: dict[int, int] = {} #store the TF-IDF representation of every movie
        self.user_ratings: dict[int, dict[int, float]] = {}
        self.user_profiles: dict[int, np.ndarray] = {}
        #Stores the TF-IDF profile representing each user's movie taste
        self.user_rated_indices: dict[int, np.ndarray] = {}
        self.user_rated_values: dict[int, np.ndarray] = {}
        self.user_means: dict[int, float] = {} #store movie position and ratings
        self.global_mean: float = 3.5 #stores each user's average rating
        self.movies: pd.DataFrame | None = None
        self.movie_content: pd.DataFrame | None = None
        self.neighbor_k = CONTENT_NEIGHBOR_K
        self.shrinkage = CONTENT_SHRINKAGE
        self.vectorizer = TfidfVectorizer(**CONTENT_VECTORIZER_PARAMS) #Creates the TF-IDF vectorizer
        
    #Train/Build the Model
    def fit(
        self,
        ratings: pd.DataFrame,
        movie_content: pd.DataFrame,
        movies: pd.DataFrame | None = None,
    ) -> "ContentBasedRecommender":
        content = movie_content.copy()
        #make sure the content feature column has no missing b=value and contains string
        content["content_features"] = content["content_features"].fillna("").astype(str)
        #Store movie information
        self.movie_content = content
        self.movie_ids = content["movieId"].to_numpy()
        #Create TF-IDF features
        tfidf_sparse = self.vectorizer.fit_transform(content["content_features"])
        #Normalize TF-IDF
        self.tfidf_matrix = normalize(tfidf_sparse, norm="l2").toarray()
        #Create Movie ID
        self.movie_id_to_index = {
            int(movie_id): index for index, movie_id in enumerate(self.movie_ids)
        }
        #Calculate global average rating
        self.global_mean = float(ratings["rating"].mean())
        #Store user ratings
        self.user_ratings = {
            int(user_id): {
                int(row.movieId): float(row.rating) for row in user_frame.itertuples()
            }
            for user_id, user_frame in ratings.groupby("userId")
        }
        self.movies = movies
        self._build_user_indexes() #Build user profiles
        return self
    
    #Build User Taste Profiles
    def _build_user_indexes(self) -> None:
        if self.tfidf_matrix is None:
            return

        self.user_profiles = {}
        self.user_rated_indices = {}
        self.user_rated_values = {}
        self.user_means = {}

        for user_id, rated_movies in self.user_ratings.items(): #Processes one user at a time
            indices = []
            values = []
            #Find their rated movies
            for movie_id, rating in rated_movies.items():
                index = self.movie_id_to_index.get(movie_id)
                if index is not None:
                    #Store indexes and ratings
                    indices.append(index)
                    values.append(rating)
            if not indices:
                continue

            index_array = np.asarray(indices, dtype=int)
            value_array = np.asarray(values, dtype=float)
            #Create the user's profile
            profile = np.average(self.tfidf_matrix[index_array], axis=0, weights=value_array)
            #Normalize the user profile
            self.user_profiles[user_id] = normalize(profile.reshape(1, -1), norm="l2").ravel()
            #Store information for prediction
            self.user_rated_indices[user_id] = index_array
            self.user_rated_values[user_id] = value_array
            self.user_means[user_id] = float(value_array.mean())

    #Get User Profile
    def _user_profile(self, user_id: int) -> np.ndarray | None:
        return self.user_profiles.get(user_id)
    #Calculate Movie Similarity
    def similarity_to_profile(self, user_id: int, movie_id: int) -> float: #Calculate How similar is this movie to the user's overall movie taste?
        profile = self._user_profile(user_id) #get user profile
        movie_index = self.movie_id_to_index.get(int(movie_id)) #Finds the movie's TF-IDF vector
        if profile is None or movie_index is None or self.tfidf_matrix is None:
            return 0.0
        #Calculate cosine similarity
        similarity = float(
            cosine_similarity(
                profile.reshape(1, -1),
                self.tfidf_matrix[movie_index].reshape(1, -1),
            )[0, 0]
        )
        return float(np.clip(similarity, 0.0, 1.0))

    #Predict User Rating
    def predict(self, user_id: int, movie_id: int) -> float: #predict What rating is this user likely to give this movie?
        if self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")
        #Find the movie and user's rated movies = target movie, movie the users already rated, rating they gave
        movie_index = self.movie_id_to_index.get(movie_id)
        rated_indices = self.user_rated_indices.get(user_id)
        rated_values = self.user_rated_values.get(user_id)
        user_mean = self.user_means.get(user_id, self.global_mean)
        if movie_index is None or rated_indices is None or rated_values is None:
            return user_mean
        #Calculate similarity with previously rated movies
        similarities = np.clip(
            self.tfidf_matrix[rated_indices] @ self.tfidf_matrix[movie_index],
            0.0,
            None,
        )
        #Select top K similar movies
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
        #Calculate weighted rating = Movies that are more similar have greater influence on the predicted rating
        neighbor = float(np.dot(similarities, rated_values) / mass)
        #Apply shrinkage = reduce or pull a value closer to a safer/average value.
        #more data = less shrinkage, less data = more shrinkage
        predicted = (mass * neighbor + self.shrinkage * user_mean) / (
            mass + self.shrinkage
        )
        #Keep rating within the valid range
        return _clip_rating(predicted)

    #Generate Top Recommendations
    def top_candidates( #finds the best movies to recommend to a user.
        self,
        user_id: int,
        n_candidates: int,
        diversify: bool = False,
        diversity: float = 0.3,
    ) -> list[int]:
        if self.movie_ids is None or self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")

        profile = self._user_profile(user_id) #Get user's profile
        if profile is None:
            return []
        #Remove movies the user already rated
        rated_movies = set(self.user_ratings.get(user_id, {}))
        #Don't recommend movies the user has already rated
        candidate_indices = [
            index
            for index, movie_id in enumerate(self.movie_ids)
            if int(movie_id) not in rated_movies
        ]
        if not candidate_indices:
            return []
        #Calculate similarity
        scores = cosine_similarity(
            profile.reshape(1, -1),
            self.tfidf_matrix[candidate_indices],
        ).flatten()
        #Candidate Pool for Diversity, pool = group of items
        pool_size = (
            int(n_candidates * (4 + 6 * float(diversity))) if diversify else n_candidates
        )
        pool_size = min(max(pool_size, n_candidates), len(candidate_indices))
        top_local = np.argsort(scores)[-pool_size:][::-1] #Get highest-scoring candidates
        pool_indices = [candidate_indices[i] for i in top_local]
        pool_scores = scores[top_local]
        pool_ids = [int(self.movie_ids[i]) for i in pool_indices]
        #If diversity is disabled
        if not diversify or len(pool_ids) <= n_candidates:
            return pool_ids[:n_candidates]
        #If diversity is enabled
        return self._mmr_diversify(
            pool_ids, pool_indices, pool_scores, n_candidates, float(diversity)
        )

    #Diversify Recommendations
    def _mmr_diversify( #uses MMR (Maximal Marginal Relevance) = no only recommedn movie that have the highest similarity to user, recommend the movie are not too similar to each other
        self,
        candidate_ids: list[int],
        candidate_indices: list[int],
        candidate_scores: np.ndarray,
        top_n: int,
        diversity: float,
    ) -> list[int]:
        
        if self.tfidf_matrix is None:
            return candidate_ids[:top_n]
        #Normalize the recommendation scores
        scores = np.asarray(candidate_scores, dtype=float)
        if float(scores.max()) > float(scores.min()):
            #convert the score to 0 = lowest score, 1 = highest score
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            norm_scores = np.ones_like(scores)
        #Select recommendations one by one
        remaining = list(range(len(candidate_ids)))
        selected: list[int] = []
        while remaining and len(selected) < top_n:
            #First movie = highest recommendation movie
            if not selected:
                pick = max(remaining, key=lambda i: float(norm_scores[i]))
            else:
                #Calculate MMR for later movies
                def mmr(i: int) -> float:
                    #Calculate redundancy = How similar is this candidate to the movies I have already selected?
                    redundancy = max(
                        float(
                            cosine_similarity(
                                self.tfidf_matrix[candidate_indices[i]].reshape(1, -1),
                                self.tfidf_matrix[candidate_indices[j]].reshape(1, -1),
                            )[0, 0]
                        )
                        for j in selected
                    )
                    #MMR formula, MMR = relevance - redundancy penalty
                    return (1.0 - diversity) * float(norm_scores[i]) - diversity * redundancy

                pick = max(remaining, key=mmr)
            selected.append(pick)
            remaining.remove(pick)
        #return diversity movie
        return [candidate_ids[i] for i in selected]
    
    #Generate Final Recommendations
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        diversify: bool = False,
        diversity: float = 0.3,
    ) -> pd.DataFrame:
        #Get recommended movie IDs, function finding candidate movie, similarity score, excluding movies already rated, optional MMR diversity
        movie_ids = self.top_candidates(
            user_id,
            n_recommendations,
            diversify=diversify,
            diversity=diversity,
        )
        if not movie_ids:
            raise ValueError(f"No usable ratings found for user {user_id}.")
        #Create recommendation DataFrame
        recommendations = pd.DataFrame(
            {
                "movieId": movie_ids,
                #calculates the predicted rating for every recommended movie
                "predicted_rating": [
                    self.predict(user_id, movie_id) for movie_id in movie_ids
                ],
                #calculates how similar each movie is to the user's overall profile
                "similarity": [
                    self.similarity_to_profile(user_id, movie_id) for movie_id in movie_ids
                ],
            }
        )
        if self.movies is not None:
            #Add movie information, add movie title, genres
            recommendations = recommendations.merge(
                self.movies[["movieId", "title", "genres"]],
                on="movieId",
                how="left",
            )
        return recommendations.reset_index(drop=True)

    #Genre-Based Recommendations
    #Recommend Movies Similar to a Selected Genre
    def similar_to_genre(
        self,
        genre: str | list[str] | tuple[str, ...],
        n_recommendations: int = 10,
        user_id: int | None = None,
    ) -> pd.DataFrame:

        if getattr(self, "movies", None) is None or self.tfidf_matrix is None:
            raise RuntimeError("Content model is not fitted.")

        if isinstance(genre, str):
            wanted = [genre.strip()] #Put the genre into a list, strip() = remove unnecessary spaces
        else:
            wanted = [str(item).strip() for item in genre if str(item).strip()] #if genre is not a string
        wanted = [item for item in wanted if item] #Remove empty values
        #Check whether the user selected a genre
        if not wanted:
            raise ValueError("Pick at least one genre.")
        label = " + ".join(wanted)
        #Create function called matches = Does this movie's genre match the genre(s) the user wants?
        def matches(genres_value: object, require_all: bool) -> bool:
            movie_genres = set(split_movie_genres(genres_value)) #Get the movie's genres
            if require_all: #If ALL genres are required
                return all(name in movie_genres for name in wanted)
            return any(name in movie_genres for name in wanted) #if only ONE genre is required
        #Find movies belonging to the genre
        seed_ids = [ #seed_id = IDs of movies that match the user's selected genre(s)
            int(movie_id)
            for movie_id, genres in zip(
                self.movies["movieId"], self.movies["genres"], strict=False
            )
            if matches(genres, require_all=True)
        ]
        #If no movies match ALL genres
        if not seed_ids and len(wanted) > 1:
            #Try again using ANY genre
            seed_ids = [
                int(movie_id)
                for movie_id, genres in zip(
                    self.movies["movieId"], self.movies["genres"], strict=False
                )
                if matches(genres, require_all=False) #The movie only needs to match at least ONE of the selected genres
            ]
        #Convert movie IDs into indexes
        seed_indices = [
            self.movie_id_to_index[movie_id]
            for movie_id in seed_ids
            if movie_id in self.movie_id_to_index
        ]
        #Check if there are still no movies
        if not seed_indices:
            raise ValueError(f"No movies found for genre '{label}'.")
        #Create a genre centroid
        centroid = self.tfidf_matrix[seed_indices].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-12:
            centroid = centroid / norm
        #Compare every movie to the genre
        scores = self.tfidf_matrix @ centroid
        #Exclude movies already rated
        rated = set(self.user_ratings.get(int(user_id), {})) if user_id is not None else set() #Get movies the user has already rated
        ranked = np.argsort(scores)[::-1] #Sort movies by their scores
        #Create empty lists for the recommendations
        picked: list[int] = [] #picked store movie id
        picked_scores: list[float] = [] #picked store recommendation scores
        for index in ranked: #Go through the movies in ranking order
            movie_id = int(self.movie_ids[index])
            #Skip movies the user already rated
            if movie_id in rated:
                continue
            picked.append(movie_id) #Add the movie to the recommendations
            picked_scores.append(float(np.clip(scores[index], 0.0, 1.0))) #add its score, np.clip() = force the score to stay between 0 and 1
            #Return the top movies
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

    #Explain Why a Movie Was Recommended
    def recommendation_reasons(self, user_id: int, movie_id: int) -> list[str]: #return a list of reason
        #Check whether movie data exists
        if getattr(self, "movies", None) is None or self.tfidf_matrix is None:
            return []
        #Find the selected movie
        movie_row = self.movies.loc[self.movies["movieId"] == int(movie_id)] #.loc() = used to select rows froma pandas DataFrame
        if movie_row.empty:
            return []
        #Find the recommended movie's genres
        rec_genres = split_movie_genres(movie_row.iloc[0]["genres"])
        #Find movies the user liked
        liked = [
            (mid, rating)
            for mid, rating in self.user_ratings.get(int(user_id), {}).items()
            if rating >= 4.0 #movies rated 4 stars or higher as liked
        ]
        #Find liked genres, mid - movie id
        liked_genres: set[str] = set()
        for mid, _rating in liked:
            liked_row = self.movies.loc[self.movies["movieId"] == int(mid)] #find the movie
            if liked_row.empty:
                continue
            liked_genres.update(split_movie_genres(liked_row.iloc[0]["genres"])) #get the genre of that movie
        #Find shared genres
        reasons = [genre for genre in rec_genres if genre in liked_genres][:3] #maximum 3 shared genre are added

        rec_index = self.movie_id_to_index.get(int(movie_id))
        best: tuple[float, int, float] | None = None
        if rec_index is not None:
            for mid, rating in liked:
                other_index = self.movie_id_to_index.get(int(mid))
                if other_index is None:
                    continue
                #Find the Most Similar Highly Rated Movie
                similarity = float(self.tfidf_matrix[rec_index] @ self.tfidf_matrix[other_index]) #compares the recommended movie with the movies the user rated 4★ or higher
                if best is None or similarity > best[0]:
                    best = (similarity, int(mid), float(rating))
        if best is not None and best[0] >= 0.08:
            neighbour = self.movies.loc[self.movies["movieId"] == best[1]]
            if not neighbour.empty:
                title = str(neighbour.iloc[0]["title"])
                #Add the explanation
                reasons.append(
                    f"Similar to '{title}' (you rated it {best[2]:.0f}★)"
                )
        return reasons

#Evaluation
    def score_test_ratings( #generate actual and predicted ratings
        self, test_ratings: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        if test_ratings.empty:
            return np.array([]), np.array([])
        #Create two array
        actual = np.empty(len(test_ratings), dtype=float)
        predicted = np.empty(len(test_ratings), dtype=float)
        for index, row in enumerate(test_ratings.itertuples()):
            actual[index] = float(row.rating) #store the real rating
            predicted[index] = self.predict(int(row.userId), int(row.movieId)) #What rating do you predict this user would give this movie?
        return actual, predicted

    @staticmethod #used to tell python this function does not need to use the object's self data
    #Calculate Classification Metrics
    #Convert ratings into liked/not-liked classes and calculate Precision, Recall, F1 and Accuracy
    def _classification_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        relevance_threshold: float,
        decision_threshold: float,
    ) -> dict[str, float]:
        actual_liked = (actual >= relevance_threshold).astype(int)
        predicted_liked = (predicted >= decision_threshold).astype(int) #does the same thing for predicted ratings.
        #Calculate precision, recall, f1, accuracy
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
        #RMSE, MSE
        mse = float(mean_squared_error(actual, predicted))
        metrics: dict[str, float] = {
            "rmse": float(np.sqrt(mse)), #calculate RMSE
            "mse": mse, 
            "n_ratings_evaluated": float(actual.size),
        }
        #Finding the Best Decision Threshold, threshold = the value used to decide which class the prediction belongs to
        thresholds = np.round(np.arange(2.5, 4.25, 0.05), 2)
        if decision_threshold is None:
            best: dict[str, float] | None = None
            for threshold in thresholds:
                #calculates Precision, Recall and F1
                candidate = self._classification_metrics(
                    actual, predicted, relevance_threshold, float(threshold)
                )
                if best is None or candidate["f1_score"] > best["f1_score"]: #selects the threshold that gives the highest F1 score
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

    #Complete Model Evaluation
    def evaluate(
        self, #current recommender object
        test_ratings: pd.DataFrame, #test data containing the user's actual ratings
        relevance_threshold: float = 4.0, #A rating of 4.0 or higher is considered liked
    ) -> dict[str, float | np.ndarray]:
        actual, predicted = self.score_test_ratings(test_ratings) #Get actual and predicted ratings
        #Calculate evaluation metrics
        metrics = self.evaluate_predictions(
            actual, predicted, relevance_threshold=relevance_threshold
        )
        decision = float(metrics.get("decision_threshold", relevance_threshold)) #Get the decision threshold
        return {
            **metrics, #Return all the evaluation results
            "actual": actual, #Return the actual ratings
            "predicted": predicted, #Return the predicted ratings
            "y_true": (actual >= relevance_threshold).astype(int) if actual.size else np.array([]), #Convert actual ratings into liked/not-liked
            "y_pred": (predicted >= decision).astype(int) if predicted.size else np.array([]), #Convert predicted ratings into predictions
        }
        
    #Top-k Evaluation
    def evaluate_precision_recall_at_k(
        self,
        train_ratings: pd.DataFrame,
        test_ratings: pd.DataFrame,
        k: int = 10,
        threshold: float = 4.0,
        max_users: int = 150,
    ) -> dict[str, float]:
        #Find users who were in the training data
        train_users = set(train_ratings["userId"].astype(int).unique()) #set() is useful because checking whether a user exists is fast
        #Choose users that can actually be evaluated
        #Find users when they appear in the training data, have a user profile in the recommender
        eval_users = [
            int(user_id)
            for user_id in test_ratings["userId"].unique()
            if int(user_id) in train_users and int(user_id) in self.user_profiles
        ]
        #Limit the number of users
        if len(eval_users) > max_users:
            rng = np.random.default_rng(42) #42 = random seed
            eval_users = list(rng.choice(eval_users, size=max_users, replace=False)) #slect 150 different user, replace = False is dont select the same user twice
        #Create lists for the metrics, will store precision and recall
        precisions: list[float] = []
        recalls: list[float] = []
        #Evaluate each user
        for user_id in eval_users:
            #Find movies the user actually liked
            #Rating 4.0 or higher → Relevant / liked
            #Rating below 4.0 → Not relevant
            relevant = set(
                test_ratings.loc[
                    (test_ratings["userId"] == user_id)
                    & (test_ratings["rating"] >= threshold),
                    "movieId",
                ]
                .astype(int)
                .tolist()
            )
            if not relevant: #Skip users with no relevant movies
                continue

            recommended = set(self.top_candidates(user_id, k)) #Get the top K recommendations
            hits = len(recommended & relevant) #Find the movies that appear in BOTH lists
            precisions.append(hits / k) #calculate precision
            recalls.append(hits / len(relevant)) #calculate recall
    
        precision_at_k = float(np.mean(precisions)) if precisions else 0.0 #Calculate average precision
        recall_at_k = float(np.mean(recalls)) if recalls else 0.0 #Calculate average recall
        #Calculate F1
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

#Streamlit Diversity Controls
def render_diversity_controls(*, enabled: bool = True) -> tuple[bool, float]:
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

#Run a Recommendation Demo
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

#Command-Line Arguments = Allow you to control the recommender from the terminal
def parse_args() -> argparse.Namespace:
    #default = If the user doesn't tell me what to use, use this value
    parser = argparse.ArgumentParser(description="Run the content-based recommender.")
    parser.add_argument("--user-id", type=int, default=1, help="User id to recommend for.") #default = 1 is backup value
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
