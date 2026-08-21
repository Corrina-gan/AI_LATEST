# Movie Recommendation System

Streamlit app that recommends movies from the [MovieLens](https://grouplens.org/datasets/movielens/) `ml-latest-small` dataset. It compares three approaches:

- **Content-based** — TF-IDF on title, year, genres, and tags; cosine similarity to the user profile
- **Collaborative filtering** — user-based kNN, item-based kNN (cosine or Pearson), and matrix factorization (TruncatedSVD)
- **Hybrid** — support-aware blend of content + SVD (more content when a movie has few ratings)

All three models share the same 80/20 train/test split. The app reports RMSE plus liked/not-liked classification (precision, recall, F1, accuracy).

## Setup

Python 3.13 is what this project was run with. Create a virtual environment if you want, then:

```bash
py -m pip install -r requirements.txt
```

## Data

Raw MovieLens files live in `dataset/` (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`).

Clean once (users with fewer than 20 ratings and movies with fewer than 5 ratings are dropped):

```bash
py data_preprocessing.py --output-dir processed
```

That writes `processed/movies_clean.csv`, `ratings_clean.csv`, `tags_clean.csv`, `links_clean.csv`, and `movies_content.csv`.

Optional poster images (needs a free [TMDB](https://www.themoviedb.org/) API key):

```powershell
$env:TMDB_API_KEY = "your_key_here"
py fetch_posters.py
```

Posters are saved to `processed/posters.csv`. The app still runs without them.

## Run the app

```bash
py -m streamlit run app.py
```

In the sidebar pick a **User ID** and how many recommendations to show. Pages:

| Page | What it does |
| --- | --- |
| Hybrid | Top-N list, search a title for a predicted rating, content vs SVD weight, evaluation |
| Content-based | Top-N, more-like-this genre, genre profile, evaluation |
| Collaborative | User-based / item-based / SVD, genre filter, neighborhood *k*, evaluation |
| Model Comparison | Side-by-side metrics for the three algorithms |
| Data Visualization | EDA charts from the cleaned ratings |
| Data Explorer | Browse movies, ratings, links, posters |

## Project layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit UI |
| `content_based.py` | Content-based recommender |
| `collaborative_filtering.py` | Collaborative recommender (standalone; hybrid does not import this) |
| `hybrid.py` | Content + SVD hybrid and alpha tuning |
| `data_preprocessing.py` | Clean and feature-build the MovieLens CSVs |
| `data_visualization.py` | EDA plots |
| `fetch_posters.py` | Optional TMDB poster download |
| `dataset/` | Raw MovieLens files |
| `processed/` | Cleaned CSVs used by the app |

You can also run a module on its own, for example:

```bash
py content_based.py --user-id 1 --top-n 10
py collaborative_filtering.py --user-id 1 --top-n 10
py data_visualization.py
```

## Dataset citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4. https://doi.org/10.1145/2827872
