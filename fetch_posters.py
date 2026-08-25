"""
Fetch TMDB poster images for every movie in processed/links_clean.csv.

Requires a (free) TMDB API key in C:\\AI2\\.env as:
  TMDB_API_KEY=your_key_here

Or set the environment variable TMDB_API_KEY.

Resumable: progress is saved to processed/posters.csv every SAVE_EVERY
movies, and a re-run only fetches movies missing from that file.

A movie whose lookup fails is recorded with an empty poster_url, so a
plain re-run will not pick it up again. Pass --retry-missing to re-query
those rows.

Run:
  py fetch_posters.py
  py fetch_posters.py --retry-missing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

#Constant
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed"
LINKS_PATH = DATA_DIR / "links_clean.csv" #TMDb / IMDb IDs from preprocessing
POSTERS_PATH = DATA_DIR / "posters.csv" #output the Streamlit app reads
ENV_PATH = BASE_DIR / ".env"

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w342" #medium poster size for the movie cards
REQUEST_TIMEOUT = 10
REQUESTS_PER_SECOND = 10 #stay under TMDB's rate limit
SAVE_EVERY = 100 #write posters.csv often so a crash does not lose progress


#Read TMDB_API_KEY from .env without overwriting a key already set in the shell
def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()

API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
if not API_KEY:
    sys.exit(
        "TMDB_API_KEY is not set.\n"
        "Put it in C:\\AI2\\.env as:\n"
        "  TMDB_API_KEY=your_key_here\n"
        "Or in PowerShell:\n"
        '  $env:TMDB_API_KEY = "your_key_here"; py fetch_posters.py\n'
    )


#Look up a poster by TMDb movie id
def fetch_poster_path(tmdb_id, session):
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
    try:
        resp = session.get(url, params={"api_key": API_KEY}, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None

    if resp.status_code == 429:
        #TMDB asked us to slow down; wait, then retry this movie
        retry_after = float(resp.headers.get("Retry-After", 1))
        time.sleep(retry_after)
        return fetch_poster_path(tmdb_id, session)

    if resp.status_code != 200:
        return None

    return resp.json().get("poster_path")


#Fallback: find the TMDb movie using its IMDb id (tt0000001 style)
def fetch_poster_path_imdb(imdb_id, session):
    if pd.isna(imdb_id):
        return None
    imdb = f"tt{int(imdb_id):07d}"
    try:
        resp = session.get(
            f"https://api.themoviedb.org/3/find/{imdb}",
            params={"api_key": API_KEY, "external_source": "imdb_id"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    results = resp.json().get("movie_results") or []
    if not results:
        return None
    return results[0].get("poster_path")


#Last fallback: search TMDb by title (drop the "(1995)" year suffix)
def fetch_poster_path_title(title, session):
    if not isinstance(title, str) or not title.strip():
        return None
    query = title.rsplit(" (", 1)[0] if " (" in title else title
    try:
        resp = session.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": API_KEY, "query": query},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    results = resp.json().get("results") or []
    if not results:
        return None
    return results[0].get("poster_path")


#Try TMDb id, then IMDb id, then title until one returns a poster path
def resolve_poster_path(row, session):
    path = None
    if pd.notna(row.get("tmdbId")):
        path = fetch_poster_path(row["tmdbId"], session)
    if not path and pd.notna(row.get("imdbId")):
        path = fetch_poster_path_imdb(row["imdbId"], session)
    if not path and pd.notna(row.get("title")):
        path = fetch_poster_path_title(row["title"], session)
    return path


#Write posters.csv through a temp file so a crash cannot leave a half-written CSV
def save(done, rows):
    """Merge freshly fetched rows into done and write posters.csv atomically."""
    merged = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    merged = merged.drop_duplicates(subset="movieId", keep="last")
    merged = merged.sort_values("movieId", ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = POSTERS_PATH.with_suffix(".csv.tmp")
    merged.to_csv(tmp_path, index=False)
    tmp_path.replace(POSTERS_PATH)
    return merged


#True when this row has no usable poster URL
def _missing_poster_mask(frame: pd.DataFrame) -> pd.Series:
    urls = frame["poster_url"]
    return urls.isna() | urls.astype(str).str.strip().isin(["", "nan", "None"])


#Download posters for every movie, skipping ones already in posters.csv
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="also re-query movies already recorded with no poster",
    )
    args = parser.parse_args()

    if not LINKS_PATH.exists():
        sys.exit(
            f"Missing {LINKS_PATH}\n"
            "Run: py data_preprocessing.py --output-dir processed"
        )

    #Load IMDb / TMDb ids, then attach titles for the search fallback
    movies = pd.read_csv(LINKS_PATH)
    movies = movies.dropna(subset=["movieId"]).copy()
    movies["movieId"] = movies["movieId"].astype(int)
    if "tmdbId" in movies.columns:
        movies["tmdbId"] = pd.to_numeric(movies["tmdbId"], errors="coerce")
    if "imdbId" in movies.columns:
        movies["imdbId"] = pd.to_numeric(movies["imdbId"], errors="coerce")
    movies = movies.drop_duplicates(subset="movieId", keep="first")

    movies_meta_path = DATA_DIR / "movies_clean.csv"
    if movies_meta_path.exists():
        titles = pd.read_csv(movies_meta_path)[["movieId", "title"]]
        titles["movieId"] = titles["movieId"].astype(int)
        movies = movies.merge(titles, on="movieId", how="left")
    else:
        movies["title"] = pd.NA

    #Prefer rows that at least have tmdb or imdb for primary lookup.
    movies = movies.loc[movies["tmdbId"].notna() | movies["imdbId"].notna()].copy()

    #Resume from posters.csv if a previous run already fetched some movies
    if POSTERS_PATH.exists():
        try:
            done = pd.read_csv(POSTERS_PATH)
        except pd.errors.ParserError:
            sys.exit(
                f"{POSTERS_PATH} is corrupted.\n"
                "Delete it or restore from posters.csv.bak, then re-run this script."
            )
        done_ids = set(done["movieId"].astype(int))
        print(f"Resuming: {len(done_ids):,} posters already fetched.")
    else:
        done = pd.DataFrame(columns=["movieId", "tmdbId", "poster_path", "poster_url"])
        done_ids = set()

    todo = movies[~movies["movieId"].isin(done_ids)]

    #Re-query movies that were saved with a blank poster (failed last time)
    if args.retry_missing and not done.empty:
        blank_ids = set(done.loc[_missing_poster_mask(done), "movieId"].astype(int))
        retry = movies[movies["movieId"].isin(blank_ids)]
        print(f"Retrying {len(retry):,} movies recorded with no poster.")
        todo = pd.concat([todo, retry], ignore_index=True).drop_duplicates(
            subset="movieId", keep="first"
        )

    print(f"Fetching posters for {len(todo):,} movies...")

    if todo.empty:
        print("Nothing to do.")
        return

    session = requests.Session()
    rows = []
    delay = 1.0 / REQUESTS_PER_SECOND #pause between API calls

    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        poster_path = resolve_poster_path(row, session)
        poster_url = POSTER_BASE_URL + poster_path if poster_path else None
        rows.append(
            {
                "movieId": int(row["movieId"]),
                "tmdbId": int(row["tmdbId"]) if pd.notna(row.get("tmdbId")) else None,
                "poster_path": poster_path,
                "poster_url": poster_url,
            }
        )
        time.sleep(delay)

        #Checkpoint every SAVE_EVERY movies (and once at the end)
        if i % SAVE_EVERY == 0 or i == len(todo):
            batch = save(done, rows)
            found = (~_missing_poster_mask(batch)).sum()
            print(
                f"  {i:,}/{len(todo):,} fetched this run - "
                f"{found:,}/{len(batch):,} total have a poster"
            )

    print(f"Done. Saved to {POSTERS_PATH}")


if __name__ == "__main__":
    main()
