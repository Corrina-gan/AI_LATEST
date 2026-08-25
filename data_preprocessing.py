"""Data cleaning and preprocessing for the MovieLens ml-latest-small dataset."""

#Imports
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

#Constant
DATA_DIR = Path(__file__).resolve().parent / "dataset" #raw MovieLens CSV files
PROCESSED_DIR = Path(__file__).resolve().parent / "processed" #cleaned files go here

#Official MovieLens genre labels. Anything else is flagged as unexpected.
VALID_GENRES = {
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "(no genres listed)",
}

#Screening formats, not story genres. Strip these from genre_list / TF-IDF.
DROPPED_GENRE_LABELS = frozenset({"IMAX"})

#Repeat genres so TF-IDF does not get drowned out by noisy free-text tags.
GENRE_REPEAT = 3
MIN_TAG_COUNT = 2 #ignore tags that appear only once in the whole dataset
CONTENT_VECTORIZER_PARAMS = {
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

#Hyphenated genres become one token so TF-IDF does not split them apart.
GENRE_TOKEN_MAP = {
    "Sci-Fi": "SciFi",
    "Film-Noir": "FilmNoir",
    "Children's": "Children",
}

#Watch-status / queue tags add no movie content signal.
TAG_STOPWORDS = frozenset(
    {
        "in netflix queue",
        "netflix queue",
        "netflix",
        "watched",
        "seen",
        "own",
        "library",
        "dvd",
        "blu ray",
        "bluray",
        "to see",
        "want to see",
        "movie",
        "movies",
        "film",
        "films",
        "imdb",
        "tmdb",
    }
)


#Load the 4 raw MovieLens CSV files from the dataset folder
def load_raw_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load raw MovieLens CSV files."""
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    tags = pd.read_csv(data_dir / "tags.csv")
    links = pd.read_csv(data_dir / "links.csv")

    print("Movies shape:", movies.shape)
    print("Movies duplicates:", movies.duplicated().sum())
    print(movies.head())
    movies.info()
    print(movies.dtypes)
    print(movies.isnull().sum())
    print(movies.nunique())

    print("Ratings shape:", ratings.shape)
    print("Ratings duplicates:", ratings.duplicated().sum())
    print(ratings.head())
    ratings.info()
    print(ratings.dtypes)
    print(ratings.isnull().sum())
    print(ratings.nunique())

    print("Tags shape:", tags.shape)
    print("Tags duplicates:", tags.duplicated().sum())
    print(tags.head())
    tags.info()
    print(tags.dtypes)
    print(tags.isnull().sum())
    print(tags.nunique())

    print("Links shape:", links.shape)
    print("Links duplicates:", links.duplicated().sum())
    print(links.head())
    links.info()
    print(links.dtypes)
    print(links.isnull().sum())
    print(links.nunique())

    return {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
        "links": links,
    }

#--------------------------------------------------------------------------------
# Movie Cleaning
#--------------------------------------------------------------------------------
#Turn "Action|Adventure|IMAX" into a clean list of story genres
def _normalize_genre_list(genres: object) -> list[str]:
    """Drop non-genre labels such as IMAX; keep a placeholder if nothing remains."""
    if isinstance(genres, list):
        raw = genres
    else:
        raw = str(genres).split("|") #MovieLens stores genres as Action|Comedy|Drama
    cleaned = [
        genre.strip()
        for genre in raw
        if str(genre).strip()
        and str(genre).strip() not in DROPPED_GENRE_LABELS #drop IMAX and empty labels
    ]
    return cleaned or ["(no genres listed)"] #never leave a movie with zero genres


#Clean movie titles, extract the release year, and split genres
def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Clean movie metadata and derive useful columns."""
    movies = movies.copy()
    movies["title"] = movies["title"].astype(str).str.strip()
    movies["genres"] = movies["genres"].fillna("(no genres listed)").astype(str).str.strip()

    #Extract release year from titles like "Toy Story (1995)"
    year_match = movies["title"].str.extract(r"\((\d{4})\)\s*$")
    movies["year"] = pd.to_numeric(year_match[0], errors="coerce").astype("Int64")

    #Split pipe-separated genres, drop IMAX, and rebuild the joined string
    movies["genre_list"] = movies["genres"].str.split("|").map(_normalize_genre_list)
    movies["genres"] = movies["genre_list"].map(lambda parts: "|".join(parts))
    #Warn if any leftover label is not in the official MovieLens genre set
    invalid_genre_mask = movies["genre_list"].apply(
        lambda genres: any(genre not in VALID_GENRES for genre in genres)
    )
    if invalid_genre_mask.any():
        invalid_count = int(invalid_genre_mask.sum())
        print(f"Warning: {invalid_count} movies contain unexpected genre labels.")

    return movies.drop_duplicates(subset="movieId", keep="first").reset_index(drop=True)


#-------------------------------------------------------------------------------
# Ratings Cleaning
#-------------------------------------------------------------------------------
#Clean ratings: keep valid scores, known movies, and one rating per user-movie pair
def clean_ratings(
    ratings: pd.DataFrame,
    valid_movie_ids: set[int],
    min_rating: float = 0.5,
    max_rating: float = 5.0,
) -> pd.DataFrame:
    """Clean ratings and keep only records tied to known movies."""
    ratings = ratings.copy()

    #Force IDs and scores to numbers; bad values become NaN
    ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce").astype("Int64")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce").astype("Int64")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce").astype("Int64")

    ratings = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"])

    #MovieLens uses half-star scores from 0.5 to 5.0
    valid_rating_mask = ratings["rating"].between(min_rating, max_rating)
    half_star_mask = (ratings["rating"] * 2).round().eq(ratings["rating"] * 2)
    ratings = ratings.loc[valid_rating_mask & half_star_mask]

    #Drop ratings for movies that are not in the movies table
    ratings = ratings.loc[ratings["movieId"].isin(valid_movie_ids)]

    #Keep the latest rating when the same user rated the same movie more than once
    ratings = (
        ratings.sort_values("timestamp")
        .drop_duplicates(subset=["userId", "movieId"], keep="last")
        .reset_index(drop=True)
    )

    ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    return ratings


#-------------------------------------------------------------------------------
# Tags Cleaning
#-------------------------------------------------------------------------------
#Clean tags: lowercase/normalize text, drop empty tags and unknown movies
def clean_tags(tags: pd.DataFrame, valid_movie_ids: set[int]) -> pd.DataFrame:
    """Normalize tags and remove rows that do not map to known movies."""
    tags = tags.copy()

    tags["userId"] = pd.to_numeric(tags["userId"], errors="coerce").astype("Int64")
    tags["movieId"] = pd.to_numeric(tags["movieId"], errors="coerce").astype("Int64")
    tags["timestamp"] = pd.to_numeric(tags["timestamp"], errors="coerce").astype("Int64")
    tags["tag"] = tags["tag"].astype(str).str.strip()

    tags = tags.dropna(subset=["userId", "movieId", "tag", "timestamp"])
    tags = tags.loc[tags["tag"].ne("")]
    tags = tags.loc[tags["movieId"].isin(valid_movie_ids)]

    #Keep the latest copy of the same user/movie/tag
    tags = (
        tags.sort_values("timestamp")
        .drop_duplicates(subset=["userId", "movieId", "tag"], keep="last")
        .reset_index(drop=True)
    )

    tags["tag_standardization"] = tags["tag"].map(_normalize_tag)
    tags = tags.loc[tags["tag_standardization"].ne("")]
    tags = tags.loc[~tags["tag_standardization"].isin(TAG_STOPWORDS)] #drop "watched", "netflix", etc.
    tags["tagged_at"] = pd.to_datetime(tags["timestamp"], unit="s", utc=True)
    return tags.reset_index(drop=True)


#--------------------------------------------------------------------------------
# Links Cleaning
#--------------------------------------------------------------------------------
#Clean IMDb / TMDb IDs used to look movies up on other sites
def clean_links(links: pd.DataFrame, valid_movie_ids: set[int]) -> pd.DataFrame:
    """Clean external movie identifiers."""
    links = links.copy()

    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce").astype("Int64")
    links["imdbId"] = pd.to_numeric(links["imdbId"], errors="coerce").astype("Int64")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")

    links = links.dropna(subset=["movieId", "imdbId"])
    links = links.loc[links["movieId"].isin(valid_movie_ids)]
    links = links.drop_duplicates(subset="movieId", keep="first").reset_index(drop=True)
    return links

#Drop inactive users and rarely rated movies, then repeat until nothing else is removed
def filter_by_activity(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 5,
) -> pd.DataFrame:
    """
    Iteratively filter sparse users and movies.

    MovieLens already includes users with at least 20 ratings. Dropping movies
    with fewer than 5 ratings removes long-tail noise that hurts both SVD and
    content similarity.
    """
    filtered = ratings.copy()

    while True:
        #Keep users who still have enough ratings
        user_counts = filtered["userId"].value_counts()
        active_users = set(user_counts[user_counts >= min_user_ratings].index)
        filtered = filtered.loc[filtered["userId"].isin(active_users)]

        #Then keep movies that still have enough ratings
        movie_counts = filtered["movieId"].value_counts()
        active_movies = set(movie_counts[movie_counts >= min_movie_ratings].index)
        next_filtered = filtered.loc[filtered["movieId"].isin(active_movies)]

        #Stop when a pass removes nobody
        if len(next_filtered) == len(filtered):
            break
        filtered = next_filtered

    return filtered.reset_index(drop=True)


#Lowercase a tag and strip punctuation so "Sci-Fi!" and "sci fi" match
def _normalize_tag(tag: object) -> str:
    text = str(tag).lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


#Title text for TF-IDF: drop the year, keep letters/numbers only
def _title_feature_text(title: object) -> str:
    text = str(title)
    text = re.sub(r"\s*\(\d{4}\)\s*$", "", text) #remove "(1995)" from the end
    text = re.sub(r"[^a-zA-Z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


#Genre text for TF-IDF: repeat each genre so it outweighs noisy tags
def _genre_feature_text(genres: object) -> str:
    raw = str(genres).strip()
    if not raw or raw == "(no genres listed)":
        return ""
    tokens: list[str] = []
    for genre in raw.split("|"):
        genre = genre.strip()
        if not genre or genre == "(no genres listed)" or genre in DROPPED_GENRE_LABELS:
            continue
        token = GENRE_TOKEN_MAP.get(genre, re.sub(r"[^A-Za-z0-9]+", "", genre))
        if token:
            tokens.extend([token] * GENRE_REPEAT) #repeat 3 times so genres stay important
    return " ".join(tokens)


#Year text for TF-IDF, e.g. "year_1995 decade_1990s"
def _year_feature_text(year: object) -> str:
    try:
        if pd.isna(year):
            return ""
        year_int = int(year)
    except (TypeError, ValueError):
        return ""
    if year_int < 1870 or year_int > 2030:
        return ""
    decade = (year_int // 10) * 10
    return f"year_{year_int} decade_{decade}s"


#Build one text field per movie for the content-based model
def build_movie_content(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """
    Combine title, year, boosted genres, and cleaned tags into one TF-IDF field.

    Example:
        toy story year_1995 decade_1990s Adventure Adventure Adventure ... pixar fun
    """
    movie_content = movies[["movieId", "title", "genres"]].copy()
    if "year" in movies.columns:
        movie_content["year"] = movies["year"]
    else:
        #If year was not cleaned yet, pull it from the title
        year_match = movie_content["title"].str.extract(r"\((\d{4})\)\s*$")
        movie_content["year"] = pd.to_numeric(year_match[0], errors="coerce").astype("Int64")

    movie_content["title_text"] = movie_content["title"].map(_title_feature_text)
    movie_content["year_text"] = movie_content["year"].map(_year_feature_text)
    movie_content["genres_text"] = movie_content["genres"].map(_genre_feature_text)

    #Attach cleaned tags, then fill movies that have none
    tag_text = _prepare_tag_text(tags)
    movie_content = movie_content.merge(tag_text, on="movieId", how="left")
    movie_content["tags"] = movie_content["tags"].fillna("")

    #Join title + year + genres + tags into one string for TF-IDF
    movie_content["content_features"] = (
        movie_content[["title_text", "year_text", "genres_text", "tags"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return movie_content


#Group tags by movie: drop rare/stopword tags, keep unique tags in one string
def _prepare_tag_text(tags: pd.DataFrame) -> pd.DataFrame:
    if tags.empty:
        return pd.DataFrame(columns=["movieId", "tags"])

    tag_column = "tag_standardization" if "tag_standardization" in tags.columns else "tag"
    prepared = tags[["movieId", tag_column]].copy()
    prepared[tag_column] = prepared[tag_column].map(_normalize_tag)
    prepared = prepared.loc[prepared[tag_column].ne("")]
    prepared = prepared.loc[~prepared[tag_column].isin(TAG_STOPWORDS)]

    #Drop tags that appear only once in the whole dataset
    tag_counts = prepared[tag_column].value_counts()
    keep_tags = set(tag_counts[tag_counts >= MIN_TAG_COUNT].index)
    prepared = prepared.loc[prepared[tag_column].isin(keep_tags)]

    return (
        prepared.groupby("movieId")[tag_column]
        .apply(lambda values: " ".join(dict.fromkeys(values))) #keep unique tags, original order
        .reset_index(name="tags")
    )


#Print a short overview after cleaning
def summarize_data(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    tags: pd.DataFrame,
    links: pd.DataFrame,
) -> None:
    """Print a short summary of the cleaned dataset."""
    print("Cleaned dataset summary")
    print("-" * 28)
    print(f"Movies:  {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")
    print(f"Tags:    {len(tags):,}")
    print(f"Links:   {len(links):,}")
    print(f"Users:   {ratings['userId'].nunique():,}")
    print(
        "Rating distribution:\n"
        f"{ratings['rating'].value_counts().sort_index().to_string()}"
    )
    print(f"Movies with year: {int(movies['year'].notna().sum()):,}" if "year" in movies.columns else "")


#Full pipeline: load -> clean -> filter -> build content text -> save CSV files
def preprocess_dataset(
    data_dir: Path = DATA_DIR,
    output_dir: Path = PROCESSED_DIR,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 5,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the full cleaning pipeline and optionally save processed files."""
    raw = load_raw_data(data_dir)

    movies = clean_movies(raw["movies"])
    movie_ids = set(movies["movieId"].dropna().astype(int))

    ratings = clean_ratings(raw["ratings"], movie_ids)
    ratings = filter_by_activity(
        ratings,
        min_user_ratings=min_user_ratings,
        min_movie_ratings=min_movie_ratings,
    )

    #After activity filtering, drop movies that no longer have ratings
    active_movie_ids = set(ratings["movieId"].astype(int))
    movies = movies.loc[movies["movieId"].isin(active_movie_ids)].reset_index(drop=True)

    tags = clean_tags(raw["tags"], active_movie_ids)
    links = clean_links(raw["links"], active_movie_ids)
    movie_content = build_movie_content(movies, tags)

    processed = {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
        "links": links,
        "movie_content": movie_content,
    }

    summarize_data(movies, ratings, tags, links)

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        movies.to_csv(output_dir / "movies_clean.csv", index=False)
        ratings.to_csv(output_dir / "ratings_clean.csv", index=False)
        tags.to_csv(output_dir / "tags_clean.csv", index=False)
        links.to_csv(output_dir / "links_clean.csv", index=False)
        movie_content.to_csv(output_dir / "movies_content.csv", index=False)
        print(f"\nSaved cleaned files to: {output_dir}")

    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean MovieLens dataset files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing raw MovieLens CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory where cleaned CSV files will be written.",
    )
    parser.add_argument(
        "--min-user-ratings",
        type=int,
        default=20,
        help="Minimum number of ratings required per user.",
    )
    parser.add_argument(
        "--min-movie-ratings",
        type=int,
        default=5,
        help="Minimum number of ratings required per movie.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_user_ratings=args.min_user_ratings,
        min_movie_ratings=args.min_movie_ratings,
    )
