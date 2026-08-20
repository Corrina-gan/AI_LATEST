"""Streamlit movie recommendation app (MovieLens assignment UI)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import hybrid

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

importlib.reload(hybrid)

HybridRecommender = hybrid.HybridRecommender
BASE_DIR = Path(__file__).resolve().parent

ALGORITHM_OPTIONS = {
    "Hybrid (Content + Collaborative)": "hybrid",
    "Content-based (TF-IDF)": "content",
    "Collaborative (SVD)": "collaborative",
}


def _find_processed_file(filename: str) -> Path:
    for candidate in (
        BASE_DIR / "processed" / filename,
        BASE_DIR / "dataset" / "processed" / filename,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{filename} not found. Run `py data_preprocessing.py --output-dir processed` first."
    )


@st.cache_data(show_spinner=False)
def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings, movies, movie_content = hybrid.load_data()
    links = pd.read_csv(_find_processed_file("links_clean.csv"))
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce").astype("Int64")
    links["imdbId"] = pd.to_numeric(links["imdbId"], errors="coerce").astype("Int64")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
    links = links.dropna(subset=["movieId"]).drop_duplicates("movieId")

    try:
        posters = pd.read_csv(_find_processed_file("posters.csv"))
        posters["movieId"] = pd.to_numeric(posters["movieId"], errors="coerce").astype("Int64")
        posters = posters.dropna(subset=["movieId"]).drop_duplicates("movieId")
        posters = posters[["movieId", "poster_url"]]
    except FileNotFoundError:
        posters = pd.DataFrame(columns=["movieId", "poster_url"])

    return ratings, movies, movie_content, links, posters


@st.cache_resource(show_spinner="Training models (content + collaborative + hybrid)^_^")
def train_models(n_factors: int = 20, test_size: float = 0.2, random_state: int = 42):
    ratings, movies, movie_content, links, posters = load_dataset()
    train_ratings, test_ratings = hybrid.split_train_test(
        ratings, test_size=test_size, random_state=random_state
    )

    # Fit base models once; hybrid alpha is applied later from the sidebar.
    base_model = HybridRecommender(alpha=0.5, n_factors=n_factors).fit(
        train_ratings, movie_content, movies
    )
    all_metrics = hybrid.evaluate_all_models(
        base_model, test_ratings, relevance_threshold=4.0
    )

    # Also score hybrid across a few alphas for the comparison tab.
    tuning_rows = []
    actual, content_scores, cf_scores = hybrid._collect_base_predictions(
        base_model.content_model,
        base_model.collaborative_model,
        test_ratings,
    )
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        predicted = (alpha * content_scores) + ((1 - alpha) * cf_scores)
        metrics = hybrid.evaluate_predictions(actual, predicted, relevance_threshold=4.0)
        tuning_rows.append({"alpha": alpha, **metrics})
    tuning_results = pd.DataFrame(tuning_rows)

    user_ids = sorted(ratings["userId"].astype(int).unique().tolist())
    return (
        base_model,
        all_metrics,
        tuning_results,
        ratings,
        movies,
        movie_content,
        links,
        posters,
        user_ids,
        len(train_ratings),
        len(test_ratings),
    )


def _imdb_url(imdb_id: object) -> str | None:
    if pd.isna(imdb_id):
        return None
    return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/"


def _tmdb_url(tmdb_id: object) -> str | None:
    if pd.isna(tmdb_id):
        return None
    return f"https://www.themoviedb.org/movie/{int(tmdb_id)}"


def attach_meta(
    frame: pd.DataFrame, links: pd.DataFrame, posters: pd.DataFrame
) -> pd.DataFrame:
    enriched = frame.merge(links[["movieId", "imdbId", "tmdbId"]], on="movieId", how="left")
    enriched = enriched.merge(posters, on="movieId", how="left")
    enriched["IMDb"] = enriched["imdbId"].map(_imdb_url)
    enriched["TMDB"] = enriched["tmdbId"].map(_tmdb_url)
    return enriched


def user_summary(ratings: pd.DataFrame, user_id: int) -> dict[str, float]:
    user = ratings.loc[ratings["userId"] == user_id]
    return {
        "movies_rated": int(len(user)),
        "avg_rating": float(user["rating"].mean()) if len(user) else 0.0,
        "liked": int((user["rating"] >= 4.0).sum()),
        "disliked": int((user["rating"] <= 2.0).sum()),
    }


def user_history(
    ratings: pd.DataFrame, movies: pd.DataFrame, user_id: int
) -> pd.DataFrame:
    history = ratings.loc[ratings["userId"] == user_id].copy()
    history = history.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    history = history.sort_values("rating", ascending=False)
    return history[["movieId", "title", "genres", "rating"]].reset_index(drop=True)


def get_recommendations(
    model: HybridRecommender,
    algorithm: str,
    user_id: int,
    top_n: int,
    alpha: float,
    links: pd.DataFrame,
    posters: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    model.alpha = float(alpha)

    if algorithm == "content":
        recs = model.recommend_content(user_id, n_recommendations=top_n)
        recs = attach_meta(recs, links, posters)
        display = recs.rename(columns={"predicted_rating": "Score"})
        return display, ["Score"]

    if algorithm == "collaborative":
        recs = model.recommend_collaborative(user_id, n_recommendations=top_n)
        recs = attach_meta(recs, links, posters)
        display = recs.rename(columns={"predicted_rating": "Score"})
        return display, ["Score"]

    recs = model.recommend(user_id, n_recommendations=top_n)
    recs = attach_meta(recs, links, posters)
    display = recs.rename(
        columns={
            "content_rating": "Content score",
            "cf_rating": "Collaborative score",
            "hybrid_rating": "Hybrid score",
        }
    )
    return display, ["Content score", "Collaborative score", "Hybrid score"]


def render_recommendation_cards(frame: pd.DataFrame, score_columns: list[str]) -> None:
    if frame.empty:
        st.info("No recommendations available.")
        return

    for index, row in frame.iterrows():
        left, right = st.columns([1, 3])
        with left:
            poster_url = row.get("poster_url")
            if pd.notna(poster_url) and str(poster_url).strip():
                st.image(str(poster_url), use_container_width=True)
            else:
                st.markdown(
                    "<div style='background:#2b2b2b;border-radius:8px;padding:28px 8px;"
                    "text-align:center;color:#aaa;font-size:0.85rem;'>No poster</div>",
                    unsafe_allow_html=True,
                )
        with right:
            st.markdown(f"**{row.get('title', 'Unknown title')}**")
            if pd.notna(row.get("genres")):
                st.caption(str(row["genres"]))
            score_bits = []
            for column in score_columns:
                if column in row and pd.notna(row[column]):
                    score_bits.append(f"{column}: `{float(row[column]):.2f}`")
            if score_bits:
                st.markdown(" · ".join(score_bits))
            link_bits = []
            if pd.notna(row.get("IMDb")):
                link_bits.append(f"[IMDb]({row['IMDb']})")
            if pd.notna(row.get("TMDB")):
                link_bits.append(f"[TMDB]({row['TMDB']})")
            if link_bits:
                st.markdown(" · ".join(link_bits))
        if index != frame.index[-1]:
            st.divider()


def _metric_cards(metrics: dict[str, float]) -> None:
    cols = st.columns(6)
    cols[0].metric("RMSE", f"{metrics['rmse']:.4f}")
    cols[1].metric("Precision", f"{metrics['precision']:.4f}")
    cols[2].metric("Recall", f"{metrics['recall']:.4f}")
    cols[3].metric("F1-score", f"{metrics['f1_score']:.4f}")
    cols[4].metric("Accuracy", f"{metrics['accuracy']:.4f}")
    cols[5].metric("Pred. liked cutoff", f"{metrics.get('decision_threshold', 4.0):.2f}")


def _expand_genres(movies: pd.DataFrame) -> pd.DataFrame:
    if "genre_list" in movies.columns:
        genre_lists = movies["genre_list"].apply(
            lambda value: ast.literal_eval(value)
            if isinstance(value, str) and value.startswith("[")
            else str(value).split("|")
        )
    else:
        genre_lists = movies["genres"].astype(str).str.split("|")
    expanded = movies[["movieId"]].copy()
    expanded["genre"] = genre_lists
    expanded = expanded.explode("genre")
    expanded["genre"] = expanded["genre"].str.strip()
    return expanded.loc[expanded["genre"].ne("(no genres listed)")]


def plot_rating_distribution(ratings: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    counts = ratings["rating"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#E85D75")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_genre_frequency(movies: pd.DataFrame):
    genre_rows = _expand_genres(movies)
    counts = genre_rows["genre"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(counts.index, counts.values, color="#5B8FF9")
    ax.set_title("Top Genres")
    ax.set_xlabel("Number of movies")
    fig.tight_layout()
    return fig


def plot_user_activity(ratings: pd.DataFrame):
    user_counts = ratings.groupby("userId").size()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(user_counts, bins=30, color="#5AD8A6")
    ax.set_title("Ratings per User")
    ax.set_xlabel("Ratings")
    ax.set_ylabel("Users")
    fig.tight_layout()
    return fig


def main() -> None:
    st.markdown(
        """
        <style>
        div.stButton > button[kind="primary"] {
            background-color: #E85D75;
            border-color: #E85D75;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🎬 Movie Recommendation System")

    try:
        (
            model,
            all_metrics,
            tuning_results,
            ratings,
            movies,
            _movie_content,
            links,
            posters,
            user_ids,
            train_size,
            test_size,
        ) = train_models()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Run `py data_preprocessing.py --output-dir processed` first.")
        return

    # ---------------------------------------------------------------------
    # Sidebar settings
    # ---------------------------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        algorithm_label = st.selectbox("Algorithm", list(ALGORITHM_OPTIONS.keys()))
        algorithm = ALGORITHM_OPTIONS[algorithm_label]
        user_id = st.selectbox("User ID", user_ids, index=user_ids.index(1) if 1 in user_ids else 0)
        top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)
        alpha = st.slider(
            "Content-based weight (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            help="Hybrid score = alpha * content + (1 - alpha) * collaborative. "
            "Only used when Algorithm is Hybrid.",
            disabled=algorithm != "hybrid",
        )
        if algorithm == "hybrid":
            st.caption(f"Collaborative weight = {1 - alpha:.2f}")
        if st.button("Clear model cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    # Diversity knobs only render for the content-based model; the defaults
    # below keep both names defined so later branches can pass them through
    # unconditionally (diversify=False is a no-op re-rank).
    diversify = False
    diversity_strength = 0.3
    if algorithm == "Content-Based":
        diversify = st.checkbox(
            "Diversify recommendations",
            value=False,
            help="Re-ranks the top results with MMR so they aren't near-duplicates "
                 "of each other (e.g. 10 Pixar sequels back to back).",
        )
        diversity_strength = st.slider(
            "Diversity strength",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            disabled=not diversify,
            help="0 = pure relevance ranking, 1 = maximise spread over relevance.",
        )


    # ---------------------------------------------------------------------
    # User summary + rating history
    # ---------------------------------------------------------------------
    summary = user_summary(ratings, int(user_id))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Movies Rated", f"{summary['movies_rated']:,}")
    metric_cols[1].metric("Avg Rating", f"{summary['avg_rating']:.2f}")
    metric_cols[2].metric("Liked (≥4★)", f"{summary['liked']:,}")
    metric_cols[3].metric("Disliked (≤2★)", f"{summary['disliked']:,}")

    with st.expander(f"👤 User {user_id}'s Rating History"):
        history = user_history(ratings, movies, int(user_id))
        st.dataframe(history, use_container_width=True, hide_index=True)

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "🎯 Get Recommendation",
            "📊 Evaluation",
            "📈 Model Comparison",
            "📉 Data Visualization",
            "🔍 Data Explorer",
        ]
    )

    # ---------------------------------------------------------------------
    # Tab 1: Get Recommendation - selected algorithm + posters
    # ---------------------------------------------------------------------
    with t1:
        if algorithm == "hybrid":
            st.write(
                "Recommendations blend a content-based score (genres/tags similarity) "
                "with a collaborative-filtering score (SVD on the user-movie rating matrix)."
            )
        elif algorithm == "content":
            st.write(
                "Content-based recommendations use TF-IDF on genres + tags and cosine similarity."
            )
        else:
            st.write(
                "Collaborative recommendations use SVD matrix factorization on user-movie ratings."
            )

        if st.button("Get Recommendations", type="primary"):
            st.session_state["show_recs"] = True
            st.session_state["recs_key"] = (algorithm, int(user_id), int(top_n), float(alpha))

        should_show = st.session_state.get("show_recs", False)
        current_key = (algorithm, int(user_id), int(top_n), float(alpha))
        if should_show and st.session_state.get("recs_key") == current_key:
            if posters.empty or posters["poster_url"].isna().all():
                st.info(
                    "Posters not found. Run `py fetch_posters.py` once to create "
                    "`processed/posters.csv`. Links still work without posters."
                )
            with st.spinner("Generating recommendations..."):
                try:
                    display, score_cols = get_recommendations(
                        model=model,
                        algorithm=algorithm,
                        user_id=int(user_id),
                        top_n=int(top_n),
                        alpha=float(alpha),
                        links=links,
                        posters=posters,
                    )
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.subheader(f"Top {top_n} · {algorithm_label}")
                    render_recommendation_cards(display, score_cols)
                    with st.expander("Table view"):
                        table = display.drop(
                            columns=["imdbId", "tmdbId", "poster_url"], errors="ignore"
                        )
                        st.dataframe(
                            table,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "IMDb": st.column_config.LinkColumn("IMDb"),
                                "TMDB": st.column_config.LinkColumn("TMDB"),
                            },
                        )

    # ---------------------------------------------------------------------
    # Tab 2: Evaluation - RMSE / precision / recall / F1 / accuracy
    # ---------------------------------------------------------------------
    with t2:
        st.subheader("Evaluation (80/20 split)")
        st.caption(
            f"Train ratings: {train_size:,} · Test ratings: {test_size:,} · "
            "Liked = actual rating ≥ 4.0"
        )

        selected_metrics = all_metrics[algorithm]
        st.markdown(f"#### Selected algorithm: {algorithm_label}")
        _metric_cards(selected_metrics)

        st.markdown("#### All algorithms")
        for key, label in (
            ("content", "Content-based (TF-IDF)"),
            ("collaborative", "Collaborative (SVD)"),
            ("hybrid", "Hybrid"),
        ):
            st.markdown(f"**{label}**")
            _metric_cards(all_metrics[key])

    # ---------------------------------------------------------------------
    # Tab 3: Model Comparison - every implemented model side by side
    # ---------------------------------------------------------------------
    with t3:
        st.subheader("Model Comparison")
        comparison = pd.DataFrame(
            [
                {"Algorithm": "Content-based (TF-IDF)", **all_metrics["content"]},
                {"Algorithm": "Collaborative (SVD)", **all_metrics["collaborative"]},
                {"Algorithm": "Hybrid", **all_metrics["hybrid"]},
            ]
        ).rename(
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
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.caption("Classification metrics (higher is better)")
            st.bar_chart(
                comparison.set_index("Algorithm")[
                    ["Precision", "Recall", "F1-score", "Accuracy"]
                ]
            )
        with chart_cols[1]:
            st.caption("RMSE (lower is better)")
            st.bar_chart(comparison.set_index("Algorithm")[["RMSE"]])

        st.markdown("#### Hybrid alpha sweep")
        tune_display = tuning_results.rename(
            columns={
                "rmse": "RMSE",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1-score",
                "accuracy": "Accuracy",
                "decision_threshold": "Pred. liked cutoff",
            }
        ).round(4)
        st.dataframe(tune_display, use_container_width=True, hide_index=True)
        st.line_chart(
            tune_display.set_index("alpha")[["Precision", "Recall", "F1-score", "Accuracy"]]
        )

    # ---------------------------------------------------------------------
    # Tab 4: Data Visualization - EDA charts for MovieLens
    # ---------------------------------------------------------------------
    with t4:
        st.subheader("Data Visualization")
        viz1, viz2 = st.columns(2)
        with viz1:
            st.pyplot(plot_rating_distribution(ratings), clear_figure=True)
            st.pyplot(plot_user_activity(ratings), clear_figure=True)
        with viz2:
            st.pyplot(plot_genre_frequency(movies), clear_figure=True)

    # ---------------------------------------------------------------------
    # Tab 5: Data Explorer - browse movies / ratings / links / posters
    # ---------------------------------------------------------------------
    with t5:
        st.subheader("Data Explorer")
        explore_choice = st.selectbox(
            "Dataset",
            ["Movies", "Ratings", "Links", "Posters"],
        )
        if explore_choice == "Movies":
            query = st.text_input("Search title / genres")
            view = movies.copy()
            if query.strip():
                mask = view["title"].str.contains(query, case=False, na=False) | view[
                    "genres"
                ].str.contains(query, case=False, na=False)
                view = view.loc[mask]
            st.dataframe(view, use_container_width=True, hide_index=True)
        elif explore_choice == "Ratings":
            st.dataframe(ratings.head(1000), use_container_width=True, hide_index=True)
            st.caption("Showing first 1,000 rating rows.")
        elif explore_choice == "Links":
            st.dataframe(links, use_container_width=True, hide_index=True)
        else:
            if posters.empty:
                st.info("No posters.csv found. Run `py fetch_posters.py`.")
            else:
                st.dataframe(posters, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
