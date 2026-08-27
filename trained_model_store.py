"""Save and load fitted recommenders under 04_Trained_Model/."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import collaborative_filtering
import content_based
import hybrid

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "trained_model"

CONTENT_PATH = MODEL_DIR / "content_based.joblib"
COLLAB_PATH = MODEL_DIR / "collaborative_filtering.joblib"
HYBRID_PATH = MODEL_DIR / "hybrid.joblib"
EVAL_PATH = MODEL_DIR / "evaluation.joblib"
META_PATH = MODEL_DIR / "metadata.json"

DISPLAY_METRIC_KEYS = (
    "rmse",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "decision_threshold",
)
REQUIRED_FILES = (CONTENT_PATH, COLLAB_PATH, HYBRID_PATH, EVAL_PATH, META_PATH)


def _scalar_metrics(result: dict) -> dict[str, float]:
    return {key: float(result[key]) for key in DISPLAY_METRIC_KEYS if key in result}


def _pack_eval_arrays(
    metrics: dict,
    actual: np.ndarray,
    predicted: np.ndarray,
    relevance_threshold: float = 4.0,
) -> dict:
    packed = {**metrics}
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    decision = float(metrics.get("decision_threshold", relevance_threshold))
    packed["actual"] = actual
    packed["predicted"] = predicted
    packed["y_true"] = (
        (actual >= relevance_threshold).astype(int) if actual.size else np.array([], dtype=int)
    )
    packed["y_pred"] = (
        (predicted >= decision).astype(int) if predicted.size else np.array([], dtype=int)
    )
    return packed


def bundle_files_exist() -> bool:
    return all(path.exists() for path in REQUIRED_FILES)


def _read_meta() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def is_bundle_current(data_sig: str) -> bool:
    if not bundle_files_exist():
        return False
    try:
        meta = _read_meta()
    except Exception:
        return False
    return str(meta.get("data_sig", "")) == str(data_sig)


def fit_recommenders(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    movie_content: pd.DataFrame,
    n_factors: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    train_ratings, test_ratings = content_based.split_train_test(
        ratings, test_size=test_size, random_state=random_state
    )
    print("Fitting content-based model...", flush=True)
    content_model = content_based.ContentBasedRecommender().fit(
        train_ratings, movie_content, movies=movies
    )
    print("Fitting collaborative model...", flush=True)
    collaborative_model = collaborative_filtering.CollaborativeFiltering(
        n_factors=n_factors
    ).fit(train_ratings, movies=movies)
    print("Tuning hybrid alpha (this can take several minutes)...", flush=True)
    best_alpha, tuning_results, hybrid_model, hybrid_metrics = hybrid.tune_alpha(
        train_ratings,
        test_ratings,
        movie_content,
        movies,
        n_factors=n_factors,
        relevance_threshold=4.0,
        metric="f1_score",
    )

    content_eval = content_model.evaluate(test_ratings, relevance_threshold=4.0)
    collaborative_eval = collaborative_model.evaluate(test_ratings, relevance_threshold=4.0)
    test_user_ids = test_ratings["userId"].astype(int).to_numpy()
    test_movie_ids = test_ratings["movieId"].astype(int).to_numpy()
    hybrid_actual = test_ratings["rating"].to_numpy(dtype=float)
    hybrid_predicted = hybrid.blend_hybrid_scores(
        hybrid_model.content_model.predict_many(test_user_ids, test_movie_ids),
        hybrid_model.collaborative_model.predict_many(test_user_ids, test_movie_ids),
        test_movie_ids,
        hybrid_model.item_counts,
        hybrid_model.alpha,
        shrink=hybrid_model.item_shrink,
        min_mix=hybrid_model.min_content_mix,
    )
    all_eval = {
        content_based.ALGORITHM_KEY: content_eval,
        collaborative_filtering.ALGORITHM_KEY: collaborative_eval,
        "hybrid": _pack_eval_arrays(hybrid_metrics["hybrid"], hybrid_actual, hybrid_predicted),
    }
    all_metrics = {key: _scalar_metrics(value) for key, value in all_eval.items()}
    user_ids = sorted(ratings["userId"].astype(int).unique().tolist())
    return {
        "hybrid_model": hybrid_model,
        "content_model": content_model,
        "collaborative_model": collaborative_model,
        "all_metrics": all_metrics,
        "all_eval": all_eval,
        "tuning_results": tuning_results,
        "user_ids": user_ids,
        "train_size": len(train_ratings),
        "test_size": len(test_ratings),
        "best_alpha": best_alpha,
        "n_factors": n_factors,
        "test_size_frac": test_size,
        "random_state": random_state,
    }


def save_bundle(bundle: dict, data_sig: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle["content_model"], CONTENT_PATH, compress=3)
    joblib.dump(bundle["collaborative_model"], COLLAB_PATH, compress=3)
    joblib.dump(bundle["hybrid_model"], HYBRID_PATH, compress=3)
    joblib.dump(
        {
            "all_metrics": bundle["all_metrics"],
            "all_eval": bundle["all_eval"],
            "tuning_results": bundle["tuning_results"],
            "user_ids": bundle["user_ids"],
            "train_size": bundle["train_size"],
            "test_size": bundle["test_size"],
            "best_alpha": bundle["best_alpha"],
        },
        EVAL_PATH,
        compress=3,
    )
    META_PATH.write_text(
        json.dumps(
            {
                "data_sig": data_sig,
                "n_factors": bundle.get("n_factors", 20),
                "test_size": bundle.get("test_size_frac", 0.2),
                "random_state": bundle.get("random_state", 42),
                "best_alpha": float(bundle["best_alpha"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved trained models to {MODEL_DIR}", flush=True)


def load_bundle(data_sig: str) -> dict | None:
    if not is_bundle_current(data_sig):
        return None
    try:
        content_model = joblib.load(CONTENT_PATH)
        collaborative_model = joblib.load(COLLAB_PATH)
        hybrid_model = joblib.load(HYBRID_PATH)
        evaluation = joblib.load(EVAL_PATH)
        meta = _read_meta()
    except Exception as error:
        print(f"Could not load 04_Trained_Model ({error}). Retraining.", flush=True)
        return None
    return {
        "hybrid_model": hybrid_model,
        "content_model": content_model,
        "collaborative_model": collaborative_model,
        "all_metrics": evaluation["all_metrics"],
        "all_eval": evaluation["all_eval"],
        "tuning_results": evaluation["tuning_results"],
        "user_ids": evaluation["user_ids"],
        "train_size": evaluation["train_size"],
        "test_size": evaluation["test_size"],
        "best_alpha": evaluation["best_alpha"],
        "n_factors": meta.get("n_factors", 20),
        "test_size_frac": meta.get("test_size", 0.2),
        "random_state": meta.get("random_state", 42),
    }
