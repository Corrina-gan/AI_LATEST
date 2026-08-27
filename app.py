#Import Libraries
from __future__ import annotations

import ast #Imports Python's Abstract Syntax Tree module - safely convert a string that looks like a Python data structure into an actual Python object.
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
import trained_model_store

#Reloading the collaborative filtering module
if not hasattr(collaborative_filtering, "render_controls"): #hasattr() is use to checks whether the module contains render_controls
    reload(collaborative_filtering)
if not hasattr(collaborative_filtering, "DEFAULT_ITEM_METHOD"):
    collaborative_filtering.DEFAULT_ITEM_METHOD = "cosine" #if no exits, the program create cosine as default
if not hasattr(collaborative_filtering, "DEFAULT_N_COMPONENTS"):
    collaborative_filtering.DEFAULT_N_COMPONENTS = 20 #set default as 20
if not hasattr(collaborative_filtering, "ITEM_METHOD_OPTIONS"):
    collaborative_filtering.ITEM_METHOD_OPTIONS = { #creates two options for item-based collaborative filtering:Cosine similarity, Pearson/adjusted cosine similarity
        "Cosine": "cosine",
        "Pearson (adjusted cosine)": "pearson",
    }
reload(data_visualization)
reload(hybrid)

#Streamlit page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide", #use the full screen width
    initial_sidebar_state="expanded",
)

#Constants
BASE_DIR = Path(__file__).resolve().parent #find directory if the current python file is located
PROCESSED_FILES = (
    "ratings_clean.csv",
    "movies_clean.csv",
    "movies_content.csv",
)

#Main Navigation tabs
MAIN_TABS = (
    "🔀 Hybrid",
    "🎬 Content-based",
    "👥 Collaborative",
    "📈 Model Comparison",
    "📉 Data Visualization",
    "🔍 Data Explorer",
)
#Algorithm Labels
ALGORITHM_LABELS = { #change internal algorithm name to user friendly name
    "hybrid": "Hybrid (Content + Collaborative)",
    content_based.ALGORITHM_KEY: "Content-based (TF-IDF)",
    collaborative_filtering.ALGORITHM_KEY: "Collaborative (SVD)",
}

#Evaluation metrics
DISPLAY_METRIC_KEYS = (
    "rmse",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "decision_threshold", #Determines what predicted rating is considered "liked"
)

#Three models are compared in Model Comparision page
COMPARISON_ALGORITHMS = (
    "Content-based (TF-IDF)",
    "Collaborative (SVD)",
    "Hybrid",
)
CLASSIFICATION_METRICS = ("Precision", "Recall", "F1-score", "Accuracy")
#Define colour used in the graph
MODEL_COLORS = {
    "Content-based (TF-IDF)": "#8B90A0",
    "Collaborative (SVD)": "#5B8FF9",
    "Hybrid": "#E85D75",
}

#Find precessed files
def _find_processed_file(filename: str) -> Path:
    for candidate in (
        BASE_DIR / "processed" / filename,
        BASE_DIR / "dataset" / "processed" / filename,
    ):
        if candidate.exists(): #if exits return its path
            return candidate
    raise FileNotFoundError(
        f"{filename} not found. Run `py data_preprocessing.py --output-dir processed` first."
    )

#Check whether dataset changed
def processed_data_signature() -> str: #Important = Streamlit uses for caching
    """Cache key that changes when cleaned CSV files are rebuilt."""
    parts: list[str] = []
    for filename in PROCESSED_FILES:
        try:
            path = _find_processed_file(filename)
        except FileNotFoundError:
            parts.append(f"{filename}:missing")
            continue
        stat = path.stat()
        parts.append(f"{filename}:{stat.st_mtime_ns}:{stat.st_size}") #store filename, modification time, file size
    return "|".join(parts)

#Extracting evaluation metrics
def _scalar_metrics(result: dict) -> dict[str, float]: #take the evaluation result and keep only numerical metrics
    return {key: float(result[key]) for key in DISPLAY_METRIC_KEYS if key in result}

#Packing evaluations arrays
def _pack_eval_arrays( #prepare actual and predicted rating for visualization
    metrics: dict,
    actual: np.ndarray, #converts the values on NumPy arrays
    predicted: np.ndarray,
    relevance_threshold: float = 4.0,
) -> dict:
    """Attach predicted/actual arrays so the Evaluation tab can draw plots."""
    packed = {**metrics}
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    decision = float(metrics.get("decision_threshold", relevance_threshold))
    packed["actual"] = actual #store actual rating
    packed["predicted"] = predicted #store predict raating
    packed["y_true"] = (
        (actual >= relevance_threshold).astype(int) if actual.size else np.array([], dtype=int)
    ) #convert actual rating into 1 = liked, 0 = not liked
    packed["y_pred"] = (
        (predicted >= decision).astype(int) if predicted.size else np.array([], dtype=int)
    ) #convert predict rating into 1 = liked, 0 = not liked
    return packed

#Load the dataset
@st.cache_data(show_spinner=False) #tells Streamlit to cache the function's result
def load_dataset(data_sig: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:#load ratings, movies, movie content, link and posters
    del data_sig  # used only as a Streamlit cache key
    ratings, movies, movie_content = content_based.load_data() #call content based to load main dataset
    links = pd.read_csv(_find_processed_file("links_clean.csv"))
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce").astype("Int64") #convert movie id into numeric values
    links["imdbId"] = pd.to_numeric(links["imdbId"], errors="coerce").astype("Int64")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
    links = links.dropna(subset=["movieId"]).drop_duplicates("movieId") #remove missing movie id and duplicate movie id

    try: #If exists, load movie poster
        posters = pd.read_csv(_find_processed_file("posters.csv"))
        posters["movieId"] = pd.to_numeric(posters["movieId"], errors="coerce").astype("Int64")
        posters = posters.dropna(subset=["movieId"]).drop_duplicates("movieId")
        posters = posters[["movieId", "poster_url"]]
    except FileNotFoundError:
        posters = pd.DataFrame(columns=["movieId", "poster_url"])

    return ratings, movies, movie_content, links, posters

#Load tags
@st.cache_data(show_spinner=False)
def load_tags(data_sig: str) -> pd.DataFrame:
    del data_sig
    tags = pd.read_csv(_find_processed_file("tags_clean.csv"))
    if "tagged_at" in tags.columns:
        tags["tagged_at"] = pd.to_datetime(tags["tagged_at"], utc=True) #Converts the tag timestamp into a proper data/ time format
    return tags

#Train all these three models (or load them from trained_model/)
@st.cache_resource(show_spinner=False)
def train_models(
    data_sig: str,
    n_factors: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
):
    ratings, movies, movie_content, links, posters = load_dataset(data_sig) #load dataset
    bundle = trained_model_store.load_bundle(data_sig)
    if bundle is None:
        bundle = trained_model_store.fit_recommenders(
            ratings,
            movies,
            movie_content,
            n_factors=n_factors,
            test_size=test_size,
            random_state=random_state,
        )
        trained_model_store.save_bundle(bundle, data_sig)
    return (
        bundle["hybrid_model"],
        bundle["content_model"],
        bundle["collaborative_model"],
        bundle["all_metrics"],
        bundle["all_eval"],
        bundle["tuning_results"],
        ratings,
        movies,
        movie_content,
        links,
        posters,
        bundle["user_ids"],
        bundle["train_size"],
        bundle["test_size"],
        bundle["best_alpha"],
    )

#IMDb and TMDB URLs
def _imdb_url(imdb_id: object) -> str | None:
    if pd.isna(imdb_id):
        return None
    return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/"


def _tmdb_url(tmdb_id: object) -> str | None:
    if pd.isna(tmdb_id):
        return None
    return f"https://www.themoviedb.org/movie/{int(tmdb_id)}"

#Adding movie metadata
def attach_meta(
    frame: pd.DataFrame,
    links: pd.DataFrame,
    posters: pd.DataFrame,
    rating_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    enriched = frame.merge(links[["movieId", "imdbId", "tmdbId"]], on="movieId", how="left") #Uses Pandas merge() to join the datasets
    enriched = enriched.merge(posters, on="movieId", how="left")
    if rating_stats is not None and not rating_stats.empty:
        enriched = enriched.merge(rating_stats, on="movieId", how="left")
    if "n_ratings" in enriched.columns:
        if "rating_count" in enriched.columns:
            enriched["rating_count"] = enriched["n_ratings"].fillna(enriched["rating_count"])
        else:
            enriched["rating_count"] = enriched["n_ratings"]
    enriched["IMDb"] = enriched["imdbId"].map(_imdb_url)
    enriched["TMDB"] = enriched["tmdbId"].map(_tmdb_url)
    return enriched

#Movie rating statistics, calculate the average rating and number of rating for every movie
def movie_rating_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.groupby("movieId", as_index=False).agg(
        avg_rating=("rating", "mean"),
        n_ratings=("rating", "size"),
    )

#User summary
def user_summary(ratings: pd.DataFrame, user_id: int) -> dict[str, float]:
    user = ratings.loc[ratings["userId"] == user_id]
    return {
        "movies_rated": int(len(user)), #how many movie user rated
        "avg_rating": float(user["rating"].mean()) if len(user) else 0.0, #user average rating
        "liked": int((user["rating"] >= 4.0).sum()), #Number of movie rated
        "disliked": int((user["rating"] <= 2.0).sum()),
    }

#User history
def user_history(
    ratings: pd.DataFrame, movies: pd.DataFrame, user_id: int
) -> pd.DataFrame:
    history = ratings.loc[ratings["userId"] == user_id].copy()
    history = history.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    history = history.sort_values("rating", ascending=False) #Sorts the movies from highest rating to lowest rating
    return history[["movieId", "title", "genres", "rating"]].reset_index(drop=True)

#Getting recommendations
def get_recommendations(
    algorithm: str,
    user_id: int,
    top_n: int,
    alpha: float,
    links: pd.DataFrame,
    posters: pd.DataFrame,
    content_model: content_based.ContentBasedRecommender | None = None,
    collaborative_model: collaborative_filtering.CollaborativeFiltering | None = None,
    hybrid_model: hybrid.HybridRecommender | None = None,
    rating_stats: pd.DataFrame | None = None,
    diversify: bool = False,
    diversity: float = 0.3,
    cf_variant: str = collaborative_filtering.VARIANT_SVD,
    cf_k: int = collaborative_filtering.DEFAULT_NEIGHBORHOOD_K,
    cf_genres: list[str] | tuple[str, ...] | None = None,
    cf_item_method: str = collaborative_filtering.DEFAULT_ITEM_METHOD,
    cf_n_components: int = collaborative_filtering.DEFAULT_N_COMPONENTS,
) -> tuple[pd.DataFrame, list[str]]:
    #Content-based
    if algorithm == content_based.ALGORITHM_KEY:
        if content_model is None:
            raise RuntimeError("Content-based model is not trained.")
        recs = content_model.recommend( #generate recommendation
            user_id,
            n_recommendations=top_n,
            diversify=diversify, #allows the recommendation list to become more diverse
            diversity=diversity,
        )
        recs = attach_meta(recs, links, posters, rating_stats=rating_stats) #adds poster, IMDb, TMDB and rating information
        display = recs.rename(columns={"predicted_rating": "Score"})
        return display, ["Score"]
    
    #Collaborative
    if algorithm == collaborative_filtering.ALGORITHM_KEY:
        if collaborative_model is None:
            raise RuntimeError("Collaborative model is not trained.")
        recs = collaborative_model.recommend( #gets recommendations, support user-based, item-based and SVD
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
    
    #Hybrid
    if hybrid_model is None:
        raise RuntimeError("Hybrid model is not trained.")
    previous_alpha = hybrid_model.alpha #Temporarily changes the Hybrid alpha
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
    finally: #Restores the original alpha after generating recommendations
        hybrid_model.alpha = previous_alpha

#Genre processing
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

#Movie card styling
def _score_pill_class(column: str) -> str:
    name = str(column).lower()
    if "content" in name:
        return "content"
    if "collab" in name or "svd" in name:
        return "collab"
    if "hybrid" in name:
        return "hybrid"
    return "score"

#Creates the HTML card shown for each recommended movie
#Create small statistic cards
def _stat_tiles_html(tiles: list[tuple[str, str, str]]) -> str:
    cards = []
    for label, value, tone in tiles: #Exp: (label, value, tone) = ("RMSE", "0.82", "good")
        cards.append(
            '<div class="stat-card ' #create the html card
            + tone
            + '">'
            f'<div class="stat-label">{escape(label)}</div>' #display name of the statistic
            f'<div class="stat-value">{escape(value)}</div>' #display the statistic value
            "</div>"
        )
    return (
        f'<div class="stat-grid cols-{len(tiles)}">{"".join(cards)}</div>' #after create all card, combine together into one HTML
    )

#Create the actual movie
def _movie_card_html(row: pd.Series, score_columns: list[str]) -> str:
    title = escape(str(row.get("title") or "Unknown title"))
    #Get poster and movie links
    poster_url = row.get("poster_url")
    tmdb = row.get("TMDB") if pd.notna(row.get("TMDB")) else None
    imdb = row.get("IMDb") if pd.notna(row.get("IMDb")) else None
    href = tmdb or imdb

    if pd.notna(poster_url) and str(poster_url).strip(): #display the movie poster
        poster = (
            f'<img class="poster" src="{escape(str(poster_url), quote=True)}" alt="{title}">'
        )
        if href: #make poster clickable
            poster = (
                f'<a href="{escape(str(href), quote=True)}" target="_blank" rel="noopener">' #target="_blank" means the link opens in a new browser tab
                f"{poster}</a>"
            )
    else:
        poster = '<div class="poster-fallback">No poster</div>'

    genres = _genre_labels(row.get("genres"))
    chips = "".join( #create genre labels
        '<span class="chip" style="background:'
        + content_based.GENRE_COLORS.get(genre, content_based.DEFAULT_GENRE_COLOR) #content_based.DEFAULT_GENRE_COLOR = different genre have different colour
        + f';color:#fff">{escape(genre)}</span>'
        for genre in genres
    )

    meta: list[str] = [] #Creating the movie metadata
    avg = row.get("avg_rating") #Average of rating
    count = row.get("n_ratings") if pd.notna(row.get("n_ratings")) else row.get("rating_count")
    your_rating = row.get("your_rating")
    if pd.notna(your_rating):
        meta.append(f'<span class="star yours">★ {float(your_rating):.1f}</span>')
    elif pd.notna(avg):
        meta.append(f'<span class="star">★ {float(avg):.1f}</span>')
    if pd.notna(avg) and pd.notna(your_rating):
        meta.append(f'<span class="count">avg {float(avg):.1f}</span>')
    if pd.notna(count):
        n_ratings = int(count)
        label = "rating" if n_ratings == 1 else "ratings" #make the grammar correct
        meta.append(f'<span class="count">{n_ratings:,} {label}</span>')
    for column in score_columns: #Displaying recommendation scores
        if column in row.index and pd.notna(row[column]):
            short = str(column).replace(" score", "") #Shortening the score name, exp: hybrid score = hybrid
            meta.append( #Creating the score pill = displayt eh recommendation score
                f'<span class="pred {_score_pill_class(column)}">'
                f"{escape(short)} {float(row[column]):.2f}</span>"
            )
    #Creating IMDb and TMDB links
    links: list[str] = []
    if imdb:
        links.append(
            f'<a href="{escape(str(imdb), quote=True)}" target="_blank" rel="noopener">IMDb</a>' #add IMDB linl if exists
        )
    if tmdb:
        links.append(
            f'<a href="{escape(str(tmdb), quote=True)}" target="_blank" rel="noopener">TMDB</a>' #add TMDB link is exists
        )
    #Build the different sections
    chip_row = f'<div class="chips">{chips}</div>' #create genre section
    meta_row = f'<div class="meta">{"".join(meta)}</div>' #create rating and score section
    link_row = f'<div class="links">{" · ".join(links)}</div>' #create external link
    return ( #Returning the complete movie card
        '<div class="movie-grid-card">'
        f"{poster}"
        f'<div class="title">{title}</div>'
        f"{chip_row}{meta_row}{link_row}"
        "</div>"
    )

#Rendering recommendation cards
def render_recommendation_cards(frame: pd.DataFrame, score_columns: list[str]) -> None:
    if frame.empty:
        st.info("No recommendations available.")
        return
    cards = "".join( #Create the HTML for all cards
        _movie_card_html(row, score_columns) for _, row in frame.iterrows()
    )
    st.html(f'<div class="movie-grid">{cards}</div>') #Displays the HTML inside Streamlit

#Metric cards, create 6 box
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

#Classification chart
def _evaluation_classification_chart(metrics: dict) -> plt.Figure:
    labels = ["Precision", "Recall", "F1-score", "Accuracy"]
    values = [
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["f1_score"]),
        float(metrics["accuracy"]),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.0)) #Create the bar
    bars = ax.bar(labels, values, color=["#5B8FF9", "#5AD8A6", "#E85D75", "#FFC857"], width=0.55)
    _add_bar_labels(ax, bars, values, ".3f", 0.02)
    _style_chart_axes(ax, "Score (higher is better)")
    ax.set_ylim(0, 1.08) #set the y-axis
    fig.tight_layout()
    return fig

#Predict vs Actual Chart
def _evaluation_pred_vs_actual_chart(actual: np.ndarray, predicted: np.ndarray) -> plt.Figure: 
    sample_actual = np.asarray(actual, dtype=float)
    sample_predicted = np.asarray(predicted, dtype=float)
    if sample_actual.size > 2500:
        idx = np.random.default_rng(42).choice(sample_actual.size, size=2500, replace=False)
        sample_actual = sample_actual[idx]
        sample_predicted = sample_predicted[idx]
    fig, ax = plt.subplots(figsize=(6.2, 3.4)) #create a Matplotlib figure
    ax.scatter(sample_actual, sample_predicted, alpha=0.22, s=10, color="#E85D75", linewidths=0) #Create a scatter plot
    ax.plot([0.5, 5.0], [0.5, 5.0], color="#444444", linestyle="--", linewidth=1)
    _style_chart_axes(ax, "Predicted rating") #style the chart
    ax.set_xlabel("Actual rating", color=_CHART_TEXT, fontsize=10) #set x-axis
    ax.set_xlim(0.4, 5.2) #set axis limit
    ax.set_ylim(0.4, 5.2)
    fig.tight_layout()
    return fig

#Error chart
def _evaluation_residual_chart(actual: np.ndarray, predicted: np.ndarray) -> plt.Figure:
    residuals = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float) #calculate prediction error
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.hist(residuals, bins=30, color="#5B8FF9") #create error distribution histogram
    ax.axvline(0, color="#E85D75", linewidth=1.2) #create the line
    _style_chart_axes(ax, "Number of ratings")
    ax.set_xlabel("Predicted − actual", color=_CHART_TEXT, fontsize=10)
    fig.tight_layout()
    return fig

#Confusion Matrix
def _evaluation_confusion_chart(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    y_true = np.asarray(y_true, dtype=int) #actual classification
    y_pred = np.asarray(y_pred, dtype=int) #predicted classification
    matrix = np.array( #create the confusion matirx
        #calculate the four value of the confusion matrix
        [ #(y_true == 0) & (y_pred == 0) = true negative, (y_true == 0) & (y_pred == 1) = false positive, (y_true == 1) & (y_pred == 0)= false negative, (y_true == 1) & (y_pred == 1) = true positive
            [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
            [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.4)) #create the chart
    ax.imshow(matrix, cmap="Blues") #Display the matrix as colours
    peak = float(matrix.max()) if matrix.max() else 1.0 #Find the largest value
    for row in range(2): #Add the numbers to the chart
        for col in range(2):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="#FFFFFF" if matrix[row, col] > peak * 0.55 else _CHART_TEXT, #Change text colour depending on the background
            )
    ax.set_xticks([0, 1], ["Pred. not liked", "Pred. liked"], color=_CHART_TEXT) #set x-axis label
    ax.set_yticks([0, 1], ["Actual not liked", "Actual liked"], color=_CHART_TEXT) #set y-axis label
    ax.set_title("Liked vs not liked", color=_CHART_TEXT, fontsize=11) #set the chart title
    #Set the background colour
    fig.patch.set_facecolor(_CHART_BG)
    ax.set_facecolor(_CHART_BG)
    #Set the border colour
    for spine in ax.spines.values():
        spine.set_color(_CHART_GRID)
    fig.tight_layout()
    return fig

#he chart colour settings
_CHART_BG = "#FFFFFF"
_CHART_TEXT = "#222222"
_CHART_GRID = "#D0D3DA"
_CHART_PANEL = "#FFFFFF"
#Dictionary used to create shorter names for the recommendation algorithms
SHORT_ALGORITHM_LABELS = {
    "Content-based (TF-IDF)": "Content-based",
    "Collaborative (SVD)": "Collaborative",
    "Hybrid": "Hybrid",
}

#Chart Styling and Bar Labeling Functions
def _style_chart_axes(ax, ylabel: str) -> None: #applies a common style to a chart's axes.
    ax.set_facecolor(_CHART_BG) #set the chart background
    ax.figure.set_facecolor(_CHART_BG) #set the figure background
    ax.set_ylabel(ylabel, color=_CHART_TEXT, fontsize=10) #Set y-axis label
    ax.tick_params(colors=_CHART_TEXT, labelsize=10) #Styling the tick labels
    ax.spines["top"].set_visible(False) #Hide the top border
    ax.spines["right"].set_visible(False) #Hide the right border
    ax.spines["left"].set_color(_CHART_GRID) #Hide the left border
    ax.spines["bottom"].set_color(_CHART_GRID) #Hide the bottom border
    ax.yaxis.grid(True, color=_CHART_GRID, linewidth=0.8)
    ax.set_axisbelow(True)

#Chart Formatting and Visualization Helpers
def _add_bar_labels(ax, bars, values: list[float], fmt: str, offset: float) -> None: #adds numbers on top of bars in a bar chart
    for bar, value in zip(bars, values, strict=True): #zip() = pair them together
        ax.text( #add text to the chart
            bar.get_x() + bar.get_width() / 2, #horizontal position
            bar.get_height() + offset, #vertical position
            format(value, fmt),
            ha="center", #center horizontal
            va="bottom", #bottom vertical
            color=_CHART_TEXT,
            fontsize=9,
            fontweight="bold",
        )

#Model comparison chart
def _classification_comparison_chart(comparison: pd.DataFrame) -> plt.Figure:
    metrics = list(CLASSIFICATION_METRICS)
    algorithms = list(COMPARISON_ALGORITHMS)
    by_algo = comparison.set_index("Algorithm") #Organize the comparison data, changes the DataFrame so that the "Algorithm" column becomes the index
    x = list(range(len(metrics)))
    width = 0.24
    n = len(algorithms) #Count the number of algorithms

    fig, ax = plt.subplots(figsize=(8.8, 4.4)) #Create the chart
    for index, algorithm in enumerate(algorithms):
        values = [float(by_algo.loc[algorithm, metric]) for metric in metrics] #get the metric value
        offsets = [pos + (index - (n - 1) / 2) * width for pos in x] #position
        bars = ax.bar(offsets, values, width, label=algorithm, color=MODEL_COLORS[algorithm]) #create the bar
        _add_bar_labels(ax, bars, values, ".3f", 0.012)

    _style_chart_axes(ax, "Score (higher is better)")
    ax.set_xticks(x, metrics) #set x-axis label
    ax.set_ylim(0, 1.08) #set y-axis range
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

#Alpha sweep chart, creates a chart showing how different alpha values affect the Hybrid model's performance
#In Hybrid, alpha controls the balance between Content-Based Filtering and Collaborative Filtering
def _alpha_sweep_chart(tune_display: pd.DataFrame, best_alpha: float) -> plt.Figure:
    metrics = ["Precision", "Recall", "F1-score", "Accuracy"]
    colors = {
        "Precision": "#5B8FF9",
        "Recall": "#5AD8A6",
        "F1-score": "#E85D75",
        "Accuracy": "#FFC857",
    }
    alphas = tune_display["alpha"].to_numpy(dtype=float) #get the alpha value
    fig, ax = plt.subplots(figsize=(7.6, 4.0)) #create the chart
    for metric in metrics: #draw the metric linese
        values = tune_display[metric].to_numpy(dtype=float) 
        ax.plot( #plot the line
            alphas,
            values,
            color=colors[metric],
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            label=metric,
        )
    ax.axvline( #mark the best alpha
        float(best_alpha),
        color="#31333F",
        linestyle="--",
        linewidth=1.3,
        label=f"F1-tuned alpha ({best_alpha:.2f})",
    )
    stacked = np.concatenate([tune_display[metric].to_numpy(dtype=float) for metric in metrics]) #calculate y-axis range
    pad = max(0.04, float(stacked.max() - stacked.min()) * 0.35) #add space arounf=d the lines
    ax.set_ylim(max(0.0, float(stacked.min()) - pad), min(1.02, float(stacked.max()) + pad)) #set y-axis limits
    ax.set_xlim(float(alphas.min()) - 0.03, float(alphas.max()) + 0.03) #set x-axis limits
    if len(alphas) > 10: #contrl x-axis tick labels
        ax.set_xticks(alphas[::2])
    else:
        ax.set_xticks(alphas)
    _style_chart_axes(ax, "Score (higher is better)")
    ax.set_xlabel("Alpha — maximum content weight (0 = more SVD, 1 = more content)", color=_CHART_TEXT, fontsize=10) #set x-axis label
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


def _alpha_rmse_chart(tune_display: pd.DataFrame, best_alpha: float) -> plt.Figure:
    alphas = tune_display["alpha"].to_numpy(dtype=float)
    rmse = tune_display["RMSE"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.plot(
        alphas,
        rmse,
        color="#E85D75",
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        label="RMSE",
    )
    ax.axvline(
        float(best_alpha),
        color="#31333F",
        linestyle="--",
        linewidth=1.3,
        label=f"F1-tuned alpha ({best_alpha:.2f})",
    )
    pad = max(0.04, float(rmse.max() - rmse.min()) * 0.25)
    ax.set_ylim(max(0.0, float(rmse.min()) - pad), float(rmse.max()) + pad)
    ax.set_xlim(float(alphas.min()) - 0.03, float(alphas.max()) + 0.03)
    if len(alphas) > 10:
        ax.set_xticks(alphas[::2])
    else:
        ax.set_xticks(alphas)
    _style_chart_axes(ax, "RMSE (lower is better)")
    ax.set_xlabel(
        "Alpha — maximum content weight (0 = more SVD, 1 = more content)",
        color=_CHART_TEXT,
        fontsize=10,
    )
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=9,
        facecolor=_CHART_PANEL,
        edgecolor=_CHART_GRID,
        labelcolor=_CHART_TEXT,
    )
    legend.get_frame().set_alpha(1)
    fig.tight_layout()
    return fig

#F1 comparison chart
def _f1_comparison_chart(comparison: pd.DataFrame) -> plt.Figure:
    algorithms = list(COMPARISON_ALGORITHMS)
    by_algo = comparison.set_index("Algorithm") #makes "Algorithm" the index so the function can easily find the F1-score for each algorithm
    values = [float(by_algo.loc[algorithm, "F1-score"]) for algorithm in algorithms]
    colors = [MODEL_COLORS[algorithm] for algorithm in algorithms]
    labels = [SHORT_ALGORITHM_LABELS[algorithm] for algorithm in algorithms]
    #set y-axis range
    f1_min = min(values)
    f1_max = max(values)
    pad = max(0.02, (f1_max - f1_min) * 0.7) #creates extra space around the values
    y_min = max(0.0, f1_min - pad) #sets the minimum and maximum Y-axis values
    y_max = min(1.05, f1_max + pad)

    fig, ax = plt.subplots(figsize=(5.6, 4.4)) #create the bar chart
    bars = ax.bar(labels, values, color=colors, width=0.55)
    _add_bar_labels(ax, bars, values, ".4f", (y_max - y_min) * 0.02) #Add the exact F1-score
    _style_chart_axes(ax, "F1-score (higher is better)")
    ax.set_ylim(y_min, y_max) #set the y-axis limit
    fig.tight_layout()
    return fig

#Ranking models
def _comparison_ranking(comparison: pd.DataFrame) -> pd.DataFrame:
    ranked = comparison[["Algorithm"]].copy() #Create the ranking DataFrame
    directions = {"RMSE": True, **{name: False for name in CLASSIFICATION_METRICS}} #Determine whether lower or higher is better
    rank_columns = []
    for metric, lower_better in directions.items(): #Rank each metric
        column = f"{metric} rank"
        ranked[column] = comparison[metric].rank(method="min", ascending=lower_better).astype(int) #calculate the ranking
        rank_columns.append(column)
    ranked["wins"] = (ranked[rank_columns] == 1).sum(axis=1) #Count how many metrics each model wins
    ranked["avg_rank"] = ranked[rank_columns].mean(axis=1) #calculate the average rank
    ranked = ranked.sort_values(["avg_rank", "wins"], ascending=[True, False]).reset_index(drop=True) #Sort the models
    ranked["place"] = range(1, len(ranked) + 1) #Assign places
    winners = [] #Find which metrics each model won
    for _, row in ranked.iterrows():
        won = [metric for metric, col in zip(directions, rank_columns, strict=True) if row[col] == 1]
        winners.append(won)
    ranked["won_metrics"] = winners
    return ranked.merge(comparison, on="Algorithm")

#short explanation for why a model received its ranking
def _rank_reason(row: pd.Series) -> str:
    won = row["won_metrics"]
    if len(won) == 5:
        return "Wins every metric"
    if won:
        return "Best on " + ", ".join(won)
    if row["place"] == 2:
        return "Second on every metric"
    return "Third on every metric"

#takes the ranking DataFrame and converts it into HTML ranking cards for your Streamlit interface
def _ranking_board_html(ranked: pd.DataFrame) -> str:
    n = max(len(ranked), 1) #find the number model
    best = { #find the best value
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
        chips = ( #Display metrics won
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
            extra = " best" if abs(value - best[key]) < 1e-9 else "" #Highlight the best value
            stats.append(
                f'<div class="rank-stat{extra}">'
                f'<span class="lbl">{escape(label)}</span>'
                f'<span class="val">{value:{fmt}}</span>'
                "</div>"
            )
        cards.append( #Create the final HTML card
            f'<article class="rank-card place-{place}">'
            f'<div class="rank-name">{escape(title)}</div>'
            f'<div class="rank-why">{escape(_rank_reason(row))}</div>'
            f'<div class="rank-chips">{chips}</div>'
            f'<div class="rank-stats">{"".join(stats)}</div>'
            "</article>"
        )
    return ( #Arrange the cards
        f'<div class="rank-board" style="grid-template-columns: repeat({n}, minmax(0, 1fr));">'
        f"{''.join(cards)}</div>"
    )

#Ranking conclusion
#displays the ranking section of your Streamlit app
def _render_comparison_ranking(ranked: pd.DataFrame) -> None: 
    st.markdown("#### Ranking")
    st.caption("Overall place uses average rank across RMSE, precision, recall, F1-score, and accuracy. Green scores are the best in that metric.")
    st.html(_ranking_board_html(ranked)) #display the ranking board

#creates the final conclusion section based on the model ranking
def _render_comparison_conclusion(ranked: pd.DataFrame) -> None:
    winner = ranked.iloc[0] 
    second = ranked.iloc[1] if len(ranked) > 1 else None
    third = ranked.iloc[2] if len(ranked) > 2 else None
    won = winner["won_metrics"]
    st.markdown("#### Conclusion") #display heading
    with st.container(border=True):
        st.markdown(f":material/emoji_events: **{winner['Algorithm']}** is the best model") #display the winning model
        if len(won) == 5:
            st.caption(
                "On the same 80/20 test split it wins every metric: lowest RMSE and the highest "
                "precision, recall, F1-score, and accuracy."
            )
        else:
            st.caption(
                f"Average rank {winner['avg_rank']:.1f}. Best on {', '.join(won) or 'none'}."
            )
        st.markdown( #Display the winner's scores
            f":green-badge[RMSE {winner['RMSE']:.4f}] "
            f":green-badge[F1 {winner['F1-score']:.4f}] "
            f":green-badge[Precision {winner['Precision']:.4f}] "
            f":green-badge[Recall {winner['Recall']:.4f}] "
            f":green-badge[Accuracy {winner['Accuracy']:.4f}]"
        )
        notes: list[str] = [] #creates an empty list for additional information
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

#Poster notice, Checks whether movie posters are available.
def _poster_notice(posters: pd.DataFrame) -> None:
    if posters.empty or posters["poster_url"].isna().all():
        st.info(
            "Posters not found. Run `py fetch_posters.py` once to create "
            "`processed/posters.csv`. Links still work without posters."
        )

#Recommendation caching
def _load_recs( #checks whether movie poster information is available
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
    cf_variant: str = collaborative_filtering.VARIANT_USER, #type of collaborative filtering
    cf_k: int = collaborative_filtering.DEFAULT_NEIGHBORHOOD_K, #neighbourhood size
    cf_genres: list[str] | tuple[str, ...] | None = None, #selected genres
    cf_item_method: str = collaborative_filtering.DEFAULT_ITEM_METHOD, #item-based method
    cf_n_components: int = collaborative_filtering.DEFAULT_N_COMPONENTS, #number of components used in
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
    cached = st.session_state.get("_recs_cache") #Get the cached recommendations
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
                diversify=bool(diversify), #Controls whether the recommendation results should be more diverse
                diversity=float(diversity), #Controls the degree of diversification
                cf_variant=str(cf_variant), #Specifies which Collaborative Filtering variant should be used
                cf_k=int(cf_k), #Specifies the number of neighbours (k) used by a neighbourhood-based Collaborative Filtering method
                cf_genres=cf_genres, #Passes the selected movie genres to the recommendation function
                cf_item_method=str(cf_item_method), #Specifies the method used for item-based Collaborative Filtering
                cf_n_components=int(cf_n_components), #Specifies the number of components used in a dimensionality-reduction method such as SVD
            )
    except ValueError as error:
        st.error(str(error))
        return None
    st.session_state["_recs_cache"] = (cache_key, display, score_cols)
    return display, score_cols

#User panel
def render_user_panel(ratings: pd.DataFrame, movies: pd.DataFrame, user_id: int) -> None:
    summary = user_summary(ratings, int(user_id)) #calculate information about the selected user
    st.html(
        _stat_tiles_html( #display the statistic
            [
                ("Movies Rated", f"{summary['movies_rated']:,}", "rose"),
                ("Avg Rating", f"{summary['avg_rating']:.2f}", "amber"),
                ("Liked (≥4★)", f"{summary['liked']:,}", "mint"),
                ("Disliked (≤2★)", f"{summary['disliked']:,}", "violet"),
            ]
        )
    )
    with st.expander(f"👤 User {user_id}'s Rating History"): #display rating history
        history = user_history(ratings, movies, int(user_id))
        st.dataframe(history, use_container_width=True, hide_index=True)

#Taable view
def _recs_table(display: pd.DataFrame) -> None:
    with st.expander("Table view"): #Create an expandable table
        table = display.drop(columns=["imdbId", "tmdbId", "poster_url"], errors="ignore") #Remove unnecessary columns
        st.dataframe( #Display the recommendation table
            table,
            hide_index=True,
            column_config={
                "IMDb": st.column_config.LinkColumn("IMDb"),
                "TMDB": st.column_config.LinkColumn("TMDB"),
            },
        )

#Hybrid recommendation page
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
    st.subheader(f"🏆 Top {top_n} hybrid recommendations") #display the page title
    st.caption( #explain the Hybrid model
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
    display, score_cols = loaded #get the recommendations
    render_recommendation_cards(display, score_cols) #Display movie cards
    reasons = { #Generate recommendation reasons
        int(row["movieId"]): hybrid_model.recommendation_reasons(row, user_id=int(user_id))
        for _, row in display.iterrows()
    }
    st.html(hybrid.explain_blend_table_html(display, reasons)) #Display the explanation
    _recs_table(display)

#Hybrid weight page
def render_hybrid_weight(
    *,
    alpha: float,
    best_alpha: float,
    tuning_results: pd.DataFrame,
    hybrid_model,
) -> None:
    st.subheader("⚖️ Content vs Collaborative weight") #display the heading
    st.write( #Explain alpha
        "The hybrid rating is `content_weight × content score + collaborative_weight × SVD score`. "
        "Alpha is the **maximum** content weight. Rarely rated movies lean on content; "
        "popular movies stay closer to SVD."
    )
    st.html( #Display alpha values
        _stat_tiles_html(
            [
                ("Current alpha", f"{alpha:.2f}", "rose"), #The alpha currently being used
                ("F1-tuned alpha", f"{best_alpha:.2f}", "sky"), #The alpha that achieved the best F1-score during tuning
            ]
        )
    )
    mix = hybrid.example_mix_at_counts(
        float(alpha),
        shrink=hybrid_model.item_shrink,
        min_mix=hybrid_model.min_content_mix,
    )
    mix_labels = ("Rare (0 ratings)", "In between (50 ratings)", "Popular (120 ratings)")
    mix_tones = ("sky", "violet", "mint")
    st.markdown("#### Mix at this alpha")
    st.caption(
        "Alpha is the **maximum** content share. These three examples use the same "
        "formula as the curve below. Slide alpha to see the mix change."
    )
    st.html(
        _stat_tiles_html(
            [
                (
                    label,
                    f"{content * 100:.0f}% content · {cf * 100:.0f}% SVD",
                    tone,
                )
                for (label, tone), (_count, content, cf) in zip(
                    zip(mix_labels, mix_tones, strict=True),
                    mix,
                    strict=True,
                )
            ]
        )
    )
    example_col, catalog_col = st.columns(2, gap="large")
    with example_col:
        st.pyplot(
            hybrid.plot_example_mix(
                float(alpha),
                shrink=hybrid_model.item_shrink,
                min_mix=hybrid_model.min_content_mix,
            ),
            clear_figure=True,
            width="content",
        )
        st.caption("Same three examples as the tiles: 0, 50, and 120 ratings.")
    with catalog_col:
        st.pyplot(
            hybrid.plot_catalog_popularity(hybrid_model.item_counts),
            clear_figure=True,
            width="content",
        )
        rare_n, mid_n, popular_n = hybrid.catalog_popularity_counts(hybrid_model.item_counts)
        catalog_n = max(1, rare_n + mid_n + popular_n)
        st.caption(
            f"{100 * rare_n / catalog_n:.0f}% of movies are rarely rated (more content), "
            f"{100 * popular_n / catalog_n:.0f}% are popular (more SVD)."
        )

    st.pyplot( #Plot the Hybrid weights
        hybrid.plot_blend_weights(
            float(alpha),
            shrink=hybrid_model.item_shrink,
            min_mix=hybrid_model.min_content_mix,
        ),
        clear_figure=True,
        width="content",
    )
    st.caption("As a movie collects more ratings, the blend shifts toward collaborative filtering.")
    #Alpha performance analysis
    st.markdown("#### How alpha changes liked / not-liked metrics")
    st.caption(
        "Each point is Hybrid on the 80/20 test set at that alpha. "
        "**Alpha** is the maximum content weight: **0 = more SVD**, **1 = more content**. "
        "The dashed line is the F1-tuned alpha (the default). "
        "The y-axis is zoomed in so small gaps are visible — nearly flat lines mean alpha barely changes these scores."
    )
    tune_display = tuning_results.rename( #Prepare tuning results
        columns={ #rename columns
            "rmse": "RMSE",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-score",
            "accuracy": "Accuracy",
            "decision_threshold": "Pred. liked cutoff",
        }
    ).round(4)
    if not tune_display.empty and "alpha" in tune_display.columns: #Display the Alpha sweep chart
        st.pyplot(
            _alpha_sweep_chart(tune_display, float(best_alpha)),
            clear_figure=True,
        )
        st.markdown("#### RMSE by alpha")
        st.caption(
            "Lower is better. RMSE usually rises as alpha increases because extra "
            "content weight pulls predicted stars away from actual ratings. "
            "This chart is from the same test split; it does not change when you move the slider."
        )
        st.pyplot(
            _alpha_rmse_chart(tune_display, float(best_alpha)),
            clear_figure=True,
        )
    st.dataframe(tune_display, use_container_width=True, hide_index=True) #Display the tuning table

#Hybrid movie search
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
    genre_options = content_based.catalog_genres(movies) #Get available genres
    with st.form("hybrid_search_form", border=False): #reate the search form
        search_query = st.text_input("Movie title", placeholder="e.g. Toy Story") #Movie title input
        search_genres = st.multiselect( #Genre filter
            "Genres",
            options=genre_options,
            placeholder="Optional. Pick one or more genres",
        )
        search_submitted = st.form_submit_button("Search") #Submit search
    if search_submitted:
        if not search_query.strip() and not search_genres:
            st.warning("Enter a title and/or pick at least one genre.")
        else:
            st.session_state["hybrid_search_query"] = search_query.strip() #Save the search
            st.session_state["hybrid_search_genres"] = tuple(search_genres)
            st.session_state.pop("hybrid_search_run", None)
    stored_query = st.session_state.get("hybrid_search_query", "")
    stored_genres = tuple(st.session_state.get("hybrid_search_genres") or ())
    if stored_query or stored_genres:
        matches = hybrid_model.search_movies(stored_query, genres=stored_genres) #Search the movies
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
            picked_label = st.selectbox( #Let the user choose a movie
                "Pick a movie from the results",
                matches["label"].tolist(),
                key="hybrid_search_title",
            )
            picked_id = int( #Get the selected movie ID
                matches.loc[matches["label"] == picked_label, "movieId"].iloc[0]
            )
            st.caption(
                "This does not add a rating. It estimates a 0.5–5★ score for the User ID in the sidebar."
            )
            if st.button("Predict rating", type="primary"): #Predict the rating
                st.session_state["hybrid_search_run"] = (
                    int(user_id),
                    picked_id,
                    stored_query,
                    int(top_n),
                    float(alpha),
                )
    run = st.session_state.get("hybrid_search_run") #Retrieve the search request
    if not (run and run[0] == int(user_id)):
        return
    _, movie_id, _, similar_n, search_alpha = run
    previous_alpha = hybrid_model.alpha #Temporarily change alpha
    hybrid_model.alpha = float(search_alpha)
    try:
        scored = hybrid_model.score_movie(int(user_id), int(movie_id)) #Calculate the movie score
        scored = attach_meta(scored, links, posters, rating_stats)
        scored = scored.rename(
            columns={ #Rename the scores
                "content_rating": "Content score",
                "cf_rating": "Collaborative score",
                "hybrid_rating": "Hybrid score",
            }
        )
        st.markdown(f"**Predicted rating for user {user_id}**") #Display the predicted rating
        st.caption("Content = genres/tags · SVD = people with similar ratings · Hybrid = the mix.")
        render_recommendation_cards(
            scored, ["Content score", "Collaborative score", "Hybrid score"]
        )
        row = scored.iloc[0]
        your_rating = row.get("your_rating") #Check whether the user already rated it
        if pd.notna(your_rating):
            shown = f"{float(your_rating):.1f}"
            st.caption(f"You already rated this movie **{shown}★**.")
        else:
            st.caption("You have not rated this movie yet.")
        reasons = {int(row["movieId"]): hybrid_model.recommendation_reasons(row)}
        st.html(hybrid.explain_blend_table_html(scored, reasons))

        similar = hybrid_model.similar_to_movie( #Find similar unseen movies
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
        st.caption("Closest in genres/tags/title, then ranked by this user's hybrid score.") #Display similar movies
        render_recommendation_cards(
            similar, ["Content score", "Collaborative score", "Hybrid score"]
        )
    except ValueError as error:
        st.info(str(error))
    finally:
        hybrid_model.alpha = previous_alpha #Restore the original alpha

#Content-Based recommendations
#displays the Top-N Content-Based movie recommendations for a selected user
def render_content_recs(
    *,
    user_id: int,
    top_n: int,
    links, #IMDb/TMDB links
    posters,
    content_model,
    rating_stats,
    diversify: bool, #whether to apply recommendation diversity
    diversity: float, #strength of diversification
) -> None:
    st.subheader(f"🏆 Top {top_n} content-based recommendations") #Display the title
    st.caption("TF-IDF on genres + tags, ranked by cosine similarity to this user's profile.")
    loaded = _load_recs(
        algorithm=content_based.ALGORITHM_KEY,
        user_id=user_id,
        top_n=top_n,
        alpha=0.0, #this is content based, no need to combine it with Collaborative Filtering
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
    render_recommendation_cards(display, []) #Display recommendation cards
    reasons = { #Generate recommendation reasons
        #generates an explanation for each recommendation
        int(row.movieId): content_model.recommendation_reasons(int(user_id), int(row.movieId))
        for row in display.itertuples()
    }
    st.html( #Display "Why recommended?"
        content_based.why_recommended_table_html( #creates an explanation table
            display, reasons, empty_why="Matches your content profile"
        )
    )
    #Diversity comparison
    if diversify: #only runs when recommendation diversification is enabled
        baseline = content_model.recommend(user_id, n_recommendations=top_n, diversify=False) #generates the normal recommendations without diversification
        st.markdown("**🔀 Before vs after diversity**")
        st.caption(
            "Same rank, two rankings: relevance-only on the left, MMR-diversified "
            "(current diversity strength) on the right. Highlighted titles on the "
            "right replaced a different movie at that rank."
        )
        st.html(content_based.diversity_comparison_table_html(baseline, display))
    _recs_table(display)

#Allows the user to select one or more genres and find similar movies
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
    genre_options = content_based.catalog_genres(movies) #Get available genres
    with st.form("like_genre_form", border=False): #Create the genre selection form
        picked_genres = st.multiselect( #allows the user to select multiple genres.
            "Pick genres",
            options=genre_options,
            placeholder="Search and select one or more genres",
            help="Type to search. Nothing runs until you click Find similar movies.",
        )
        submitted = st.form_submit_button("Find similar movies") #Submit the search
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
        similar = content_model.similar_to_genre( #Find similar movies
            genre_names,
            n_recommendations=similar_n,
            user_id=int(user_id),
        )
    except ValueError as error:
        st.info(str(error))
        return
    similar = attach_meta(similar, links, posters, rating_stats) #Attach movie information
    similar = similar.rename(columns={"predicted_rating": "Score"}) #Rename the score
    label = " + ".join(genre_names) #Display the selected genres
    st.caption(
        f"Unseen movies closest to **{label}** (TF-IDF centroid) for user {user_id}."
    )
    render_recommendation_cards(similar, []) #Display similar movies

#Genre profile
def render_content_profile(ratings, movies, user_id: int) -> None:
    st.subheader(f"👤 Genre profile · user {user_id}")
    st.caption(
        "Built from the user's ratings ≥ 3.0, weighted by rating and split "
        "evenly across each movie's genres, then shown as a share of 100%."
    )
    profile = content_based.user_genre_profile(ratings, movies, int(user_id)) #Generate the genre profile
    if profile.empty:
        st.info("This user has no ratings ≥ 3.0 to build a genre profile.")
        return
    st.pyplot( #Display the genre profile chart
        content_based.plot_genre_profile(profile, int(user_id)),
        clear_figure=True,
    )

#Collaborative recommendations
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
    variant_names = {value: key for key, value in collaborative_filtering.VARIANT_OPTIONS.items()} #Identify the selected CF method, onvert an internal value into a readable name
    variant_name = variant_names.get(cf_variant, "Matrix Factorization (SVD)") #gets the name of the selected method
    st.subheader(f"🏆 Top {top_n} collaborative recommendations")
    if cf_variant == collaborative_filtering.VARIANT_USER: #Explain User-Based CF
        st.caption(
            f"{variant_name}: similar users (k={cf_k}) vote on movies you have not rated." #k represents the number of similar users considered
        )
    elif cf_variant == collaborative_filtering.VARIANT_ITEM: #Explain Item-Based CF
        method_names = {
            value: key for key, value in collaborative_filtering.ITEM_METHOD_OPTIONS.items() #recommendations are based on movies similar to those the user already liked
        }
        method_name = method_names.get(cf_item_method, cf_item_method) #displays the selected item similarity method
        st.caption(
            f"{variant_name}: similar movies (k={cf_k}, {method_name}) to ones you already liked."
        )
    else:
        st.caption( #Explain SVD
            f"Matrix factorization (SVD) with {cf_n_components} latent factors "
            "on the user–movie rating matrix."
        )
    if cf_genres: #Display genre filter
        st.caption("Genre filter: " + ", ".join(cf_genres))
    loaded = _load_recs( #Load Collaborative recommendations
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
    if getattr(collaborative_model, "last_is_cold_start", False): #Handle cold-start users, Cold start means there is not enough rating information about the user or item to make a reliable Collaborative Filtering recommendation
        st.info(str(getattr(collaborative_model, "last_rationale_label", "Popularity fallback")))
    display, _score_cols = loaded
    render_recommendation_cards(display, [])
    reasons = {  #Generate recommendation reasons
        int(row.movieId): collaborative_model.recommendation_reasons(
            int(user_id),
            int(row.movieId),
            variant=cf_variant,
            neighborhood_k=cf_k,
            item_method=cf_item_method,
        )
        for row in display.itertuples()
    }
    st.html( #displays a table explaining why each movie was recommended
        content_based.why_recommended_table_html(
            display, reasons, empty_why="Matches users like you"
        )
    )
    _recs_table(display)

#Evaluation page
def render_evaluation(algorithm_label: str, algorithm: str, all_eval, train_size, test_size) -> None:
    st.subheader("Evaluation (80/20 split)")
    st.caption( #Display dataset information
        f"Showing **{algorithm_label}** only. Open another algorithm tab to see a different model. "
        f"Train ratings: {train_size:,} · Test ratings: {test_size:,} · "
        "Liked = actual rating ≥ 4.0."
    )
    selected_eval = all_eval[algorithm] #Get the selected model's results
    _metric_cards(selected_eval) #Display metric cards
    #Get actual and predicted values
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
    st.pyplot(_evaluation_classification_chart(selected_eval), clear_figure=True, width="content") #Classification metrics chart
    if not has_arrays:
        return
    st.markdown("**Predicted vs actual ratings**")
    st.caption("Dashed line is a perfect prediction. Points are a sample of the test set.")
    st.pyplot(_evaluation_pred_vs_actual_chart(actual, predicted), clear_figure=True, width="content") #Classification metrics chart
    st.markdown("**Error distribution**")
    st.caption("How far predictions miss the true rating. Closer to 0 is better.")
    st.pyplot(_evaluation_residual_chart(actual, predicted), clear_figure=True, width="content") #Error distribution
    if y_true is not None and y_pred is not None and np.asarray(y_true).size > 0:
        st.markdown("**Liked vs not liked**")
        st.caption("Actual liked = rating ≥ 4.0. Predicted liked uses this model's cutoff.")
        st.pyplot(_evaluation_confusion_chart(y_true, y_pred), clear_figure=True, width="content") #Confusion matrix

#Model Comparison
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
    ).rename( #Rename metrics
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
    st.dataframe(comparison, width="stretch", hide_index=True) #Display comparison table
    st.caption(
        "Bars are grouped side by side (not stacked) so each 0–1 metric can be compared across models. "
        "Values are labeled on the bars."
    )
    chart_cols = st.columns((3, 2), gap="large")
    with chart_cols[0]:
        st.markdown("**Classification metrics** — higher is better")
        st.pyplot(_classification_comparison_chart(comparison), clear_figure=True) #Classification comparison chart
    with chart_cols[1]:
        st.markdown("**F1-score** — higher is better")
        st.caption("Axis is zoomed in so the small gaps between models are easier to see.")
        st.pyplot(_f1_comparison_chart(comparison), clear_figure=True) #F1 comparison chart
    ranked = _comparison_ranking(comparison) #Rank the models
    _render_comparison_ranking(ranked)
    _render_comparison_conclusion(ranked)

#Data Visualization page, provides Exploratory Data Analysis (EDA)
def render_visualization(data_sig: str, ratings, movies) -> None:
    st.subheader("Data Visualization")
    st.caption("Exploratory analysis of the cleaned MovieLens dataset.")
    tags = load_tags(data_sig)

    st.markdown("#### 1.1 Distribution of rating scores")
    st.caption("How users rate movies and which scores are most common.")
    st.pyplot(data_visualization.make_rating_distribution(ratings), clear_figure=True, width="content") #Shows how frequently different rating values occur
    st.dataframe(
        data_visualization.rating_distribution_stats(ratings),
        width="content",
        hide_index=True,
    )

    st.markdown("#### 1.2 User rating activity")
    st.caption(
        "Each bar is a group of users. Taller bar = more users in that group. "
        "Most people rated 50 movies or fewer. Only a few users rated more than 1,000."
    )
    st.pyplot(data_visualization.make_user_activity(ratings), clear_figure=True, width="content") #Shows how many movies each user has rated
    st.dataframe(
        data_visualization.user_activity_stats(ratings).to_frame("value"),
        width="content",
    )

    st.markdown("#### 1.3 Movie genre distribution")
    st.caption("Genres with the largest number of movies in the catalog.")
    st.pyplot(data_visualization.make_genre_distribution(movies), clear_figure=True, width="content") #Shows how many movies belong to each genre
    st.dataframe(
        data_visualization.genre_distribution_stats(movies),
        width="content",
        hide_index=True,
    )

    st.markdown("#### 1.4 Movie release trends")
    st.caption("How the number of movies in the dataset changes across release years.")
    st.pyplot(data_visualization.make_movies_by_year(movies), clear_figure=True, width="content") #Shows how the number of movies changes across release years
    st.dataframe(
        data_visualization.movies_by_year_stats(movies),
        width="content",
        hide_index=True,
        height=280,
    )

    st.markdown("#### 1.5 Top 20 most common tags")
    tag_counts = data_visualization.all_tag_counts(tags)
    st.caption(
        f"The chart shows the 20 most used tags. The table lists every unique tag "
        f"({len(tag_counts):,} tags in total)."
    )
    st.pyplot(data_visualization.make_top_tags(tags), clear_figure=True, width="content")
    st.dataframe(
        tag_counts,
        width="content",
        hide_index=True,
        height=420,
    )

    st.markdown("#### 1.6 Average rating by genre")
    st.caption("Genres that tend to receive higher audience ratings.")
    st.pyplot(
        data_visualization.make_average_rating_by_genre(movies, ratings), #Shows which genres tend to receive higher average ratings
        clear_figure=True,
        width="content",
    )
    st.dataframe(
        data_visualization.genre_rating_stats(movies, ratings), #Shows how many ratings were submitted in each year.
        width="content",
        hide_index=True,
    )

    st.markdown("#### 1.7 Rating activity over time")
    st.caption("How many ratings were submitted in each year.")
    st.pyplot(data_visualization.make_ratings_over_time(ratings), clear_figure=True, width="content") #Shows how much of the user–movie rating matrix is empty
    st.dataframe(
        data_visualization.ratings_over_time_stats(ratings),
        width="content",
        hide_index=True,
        height=280,
    )

    st.markdown("#### 1.8 Popularity vs average rating")
    st.caption(
        "Each dot is a movie. Left = few ratings, right = many ratings (log scale). "
        "Colour uses the same rare / mixed / popular groups as the hybrid."
    )
    st.pyplot(
        data_visualization.make_popularity_rating_scatter(ratings),
        clear_figure=True,
        width="content",
    )

    st.markdown("#### 1.9 Average rating by genre and decade")
    st.caption(
        "Top 10 most-rated genres. Darker = higher average rating. "
        "Blank cells mean no rated movies in that decade."
    )
    st.pyplot(
        data_visualization.make_genre_decade_heatmap(ratings, movies),
        clear_figure=True,
        width="content",
    )

    st.markdown("#### Collaborative filtering context")
    st.caption(
        "These charts show why collaborative filtering is needed: the rating matrix is "
        "mostly empty, and a small set of popular movies accounts for most ratings."
    )
    filled_pct, n_users, n_movies = data_visualization.sparsity_stats(ratings)
    st.markdown("**1.10 User–item matrix sparsity**")
    st.caption(
        f"Blue = rated, grey = missing. Sample of 80 users × 80 movies. "
        f"The full matrix is {filled_pct:.2f}% filled ({n_users:,} users × {n_movies:,} movies)."
    )
    st.pyplot(
        data_visualization.make_matrix_sparsity_heatmap(ratings),
        clear_figure=True,
        width="content",
    )
    st.markdown("**1.11 Long-tail popularity**")
    st.caption("Most movies have few ratings; a small number of titles are rated often.")
    st.pyplot(
        data_visualization.make_ratings_per_movie_distribution(ratings),
        clear_figure=True,
        width="content",
    )
    st.markdown("**1.12 Cold-start segment**")
    coldstart_fig, pct_users_below, pct_movies_below = (
        data_visualization.make_coldstart_segment_bars(ratings)
    )
    st.pyplot(coldstart_fig, clear_figure=True, width="content")
    st.caption(
        f"{pct_users_below:.1f}% of users and {pct_movies_below:.1f}% of movies "
        "have 5 or fewer ratings."
    )

#Data Explorer
def render_explorer(movies, ratings, links, posters) -> None:
    st.subheader("Data Explorer")
    filled_pct, n_users, _n_rated_movies = data_visualization.sparsity_stats(ratings) #Display dataset statistics
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
    explore_choice = st.selectbox("Dataset", ["Movies", "Ratings", "Links", "Posters"]) #elect a dataset
    if explore_choice == "Movies": #display the movie dataset
        query = st.text_input("Search title / genres") #search check movie title and genres
        view = movies.copy()
        if query.strip():
            mask = view["title"].str.contains(query, case=False, na=False) | view[
                "genres"
            ].str.contains(query, case=False, na=False)
            view = view.loc[mask]
        st.dataframe(view, use_container_width=True, hide_index=True)
    elif explore_choice == "Ratings":
        st.dataframe(ratings.head(1000), use_container_width=True, hide_index=True) #Displays the first 1,000 rating records
        st.caption("Showing first 1,000 rating rows.")
    elif explore_choice == "Links":
        st.dataframe(links, use_container_width=True, hide_index=True) #Displays movie IDs and their external IMDb/TMDB links
    else:
        if posters.empty:
            st.info("No posters.csv found. Run `py fetch_posters.py`.")
        else:
            st.dataframe(posters, use_container_width=True, hide_index=True) #displays the poster information

#Main navigation
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
    render_user_panel(ratings, movies, int(user_id)) #Display user information
    section = st.pills( #Create the main navigation, st.pills() creates clickable navigation buttons
        "Page",
        MAIN_TABS,
        default=MAIN_TABS[0],
        key="main_tab",
        label_visibility="collapsed",
    )
    if section is None: #Default page
        section = MAIN_TABS[0]

    if section == "🔀 Hybrid": #Hybrid page selection
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
    elif section == "🎬 Content-based": #Content-Based page selection
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
    elif section == "👥 Collaborative": #Collaborative page selection
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
    elif section == "📈 Model Comparison": #Model Comparison
        render_comparison(all_metrics, float(best_alpha), tuning_results)
    elif section == "📉 Data Visualization": #Data Visualization
        render_visualization(data_sig, ratings, movies)
    else:
        render_explorer(movies, ratings, links, posters) #Data Explorer

#Hybrid Recommendation page
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
    _poster_notice(posters) #Check poster availability
    features = ( #Create Hybrid features
        "🏆 Top 10 recommendations",
        "🔍 Search a movie",
        "⚖️ Content vs Collaborative weight",
        "📊 Evaluation",
    )
    feature = st.pills( #Create feature navigation
        "Hybrid features",
        features,
        default=features[0],
        key="hybrid_feature",
        label_visibility="collapsed",
    )
    if feature is None:
        feature = features[0]
    if feature == "🏆 Top 10 recommendations": #generates the top Hybrid recommendations
        render_hybrid_recs(
            user_id=user_id,
            top_n=top_n,
            alpha=alpha,
            links=links,
            posters=posters,
            hybrid_model=hybrid_model,
            rating_stats=rating_stats,
        )
    elif feature == "🔍 Search a movie": #Search a movie
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
    elif feature == "⚖️ Content vs Collaborative weight": #Weight analysis
        render_hybrid_weight(
            alpha=alpha,
            best_alpha=best_alpha,
            tuning_results=tuning_results,
            hybrid_model=hybrid_model,
        )
    else:
        render_evaluation( #Displays the Hybrid model's evaluation results
            ALGORITHM_LABELS["hybrid"],
            "hybrid",
            all_eval,
            train_size,
            test_size,
        )

#Contetn-Based Filtering page
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
    diversify: bool, #Whether recommendation diversity is enabled
    diversity: float, #Controls the strength of diversification
    all_eval, #Evaluation results for all models
    train_size, #Number of training ratings
    test_size, #Number of testing ratings
) -> None:
    st.caption("Uses movie content such as title, year, genres, and tags (TF-IDF).")
    _poster_notice(posters) #Check movie posters
    features = (
        "🏆 Top recommendations",
        "🎬 More like this genre",
        "👤 Genre profile",
        "📊 Evaluation",
    )
    feature = st.pills( #Create the feature navigation, st.pills() creates clickable navigation buttons
        "Content-based features",
        features,
        default=features[0],
        key="content_feature",
        label_visibility="collapsed",
    )
    if feature is None:
        feature = features[0]
    if feature == "🏆 Top recommendations": #Top recommendations
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
    elif feature == "🎬 More like this genre": #More like this genre
        render_content_like_genre(
            user_id=user_id,
            top_n=top_n,
            movies=movies,
            links=links,
            posters=posters,
            content_model=content_model,
            rating_stats=rating_stats,
        )
    elif feature == "👤 Genre profile": #Genre profile
        render_content_profile(ratings, movies, user_id)
    else:
        render_evaluation( #Evaluation
            ALGORITHM_LABELS[content_based.ALGORITHM_KEY],
            content_based.ALGORITHM_KEY,
            all_eval,
            train_size,
            test_size,
        )

#Collaborative Filtering page
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
    cf_variant: str, #Selects the Collaborative Filtering approach
    cf_k: int, #Number of similar users/movies considered
    cf_genres: list[str] | tuple[str, ...] | None, #Optional genre filter
    cf_item_method: str, #Method used for item similarity
    cf_n_components: int, #Number of latent factors for SVD
) -> None:
    st.caption("Uses user ratings and similar-user / similar-movie patterns.")
    _poster_notice(posters) #Check posters
    features = (
        "🏆 Top recommendations",
        "📊 Evaluation",
    )
    feature = st.pills( #Create the navigation buttons
        "Collaborative features",
        features,
        default=features[0],
        key="collab_feature",
        label_visibility="collapsed",
    )
    if feature is None:
        feature = features[0]
    if feature == "🏆 Top recommendations": #Display recommendations
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
        st.caption("Evaluation numbers below are from Matrix Factorization (SVD) on the 80/20 split.") #Display evaluation
        render_evaluation(
            ALGORITHM_LABELS[collaborative_filtering.ALGORITHM_KEY],
            collaborative_filtering.ALGORITHM_KEY,
            all_eval,
            train_size,
            test_size,
        )

#main function of the application
def main() -> None:
    st.markdown( #Set the website design
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
        .movie-grid-card .star.yours { color: #C9A227; }
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

    st.title("🎬 Movie Recommendation System") #Display the application title
    st.html('<div class="hero-bar"></div>')

    data_sig = processed_data_signature() #Get the processed data signature

    try: #Train or load the recommendation models
        spinner_text = (
            "Loading trained models from trained_model..."
            if trained_model_store.is_bundle_current(data_sig)
            else "Training models (content + collaborative + hybrid)^_^"
        )
        with st.spinner(spinner_text):
            (
                #Recommendation models
                hybrid_model,
                content_model,
                collaborative_model,
                #Evaluation information
                all_metrics,
                all_eval,
                tuning_results,
                #Dataset information
                ratings,
                movies,
                _movie_content,
                links,
                posters,
                user_ids,
                #Training information
                train_size,
                test_size,
                best_alpha,
            ) = train_models(data_sig)
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Run `py data_preprocessing.py --output-dir processed` first.")
        return

    #Fix cached model classes, module reloading and Streamlit caching
    hybrid_model.__class__ = hybrid.HybridRecommender
    hybrid_model.content_model.__class__ = hybrid.ContentBasedRecommender
    content_model.__class__ = content_based.ContentBasedRecommender
    collaborative_model.__class__ = collaborative_filtering.CollaborativeFiltering
    if getattr(content_model, "movies", None) is None:#Content-Based model has access to the movie dataset
        content_model.movies = movies

    rating_stats = movie_rating_stats(ratings) #Calculate movie rating statistics

    with st.sidebar: #Create the Sidebar
        st.header("Settings")
        user_id = st.selectbox( #creates a dropdown for selecting the user.
            "User ID",
            user_ids,
            index=user_ids.index(1) if 1 in user_ids else 0,
        )
        top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=10) #Number of recommendations
        section = st.session_state.get("main_tab", MAIN_TABS[0]) #Get the current page
        #Set default settings
        diversify = False #Controls whether Content-Based recommendations use diversification
        diversity_strength = 0.3 #Controls how strongly recommendations should be diversified
        alpha = float(st.session_state.get("hybrid_alpha", best_alpha)) #Controls the balance between Content-Based and Collaborative Filtering in the Hybrid model
        #Set Collaborative Filtering defaults
        cf_variant = collaborative_filtering.VARIANT_USER #User-Based / Item-Based / SVD
        cf_k = collaborative_filtering.DEFAULT_NEIGHBORHOOD_K #number of neighbours
        cf_genres: list[str] = [] #selected genre filters
        cf_item_method = collaborative_filtering.DEFAULT_ITEM_METHOD #item similarity method
        cf_n_components = collaborative_filtering.DEFAULT_N_COMPONENTS #number of SVD latent factors
        if section == "🔀 Hybrid": #Hybrid settings
            alpha_token = (data_sig, round(float(best_alpha), 4)) #creates a token based on the dataset and tuned alpha
            if st.session_state.get("_alpha_token") != alpha_token: #checks whether the dataset or tuned alpha has changed
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
        if section == "🎬 Content-based": #Content-Based settings
            diversify, diversity_strength = content_based.render_diversity_controls(enabled=True)
        if section == "👥 Collaborative": #Collaborative settings
            cf_variant, cf_genres, cf_k, cf_item_method, cf_n_components = (
                collaborative_filtering.render_controls(movies)
            )
        if st.button("Clear model cache"): #Clear model cache button
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
            
    #Call render_home()
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