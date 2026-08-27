"""Train the three recommenders and write them to trained_model/.

Run from the project root (the folder that contains app.py):

    py save_trained_models.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import content_based
import trained_model_store

PROCESSED_FILES = (
    "ratings_clean.csv",
    "movies_clean.csv",
    "movies_content.csv",
)


def _find_processed_file(filename: str) -> Path:
    base = Path(__file__).resolve().parent
    for candidate in (base / "processed" / filename, base / "dataset" / "processed" / filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{filename} not found. Run `py data_preprocessing.py --output-dir processed` first."
    )


def processed_data_signature() -> str:
    parts: list[str] = []
    for filename in PROCESSED_FILES:
        path = _find_processed_file(filename)
        stat = path.stat()
        parts.append(f"{filename}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return content_based.load_data()


def main() -> None:
    print("Loading cleaned CSVs from processed/...", flush=True)
    ratings, movies, movie_content = load_tables()
    data_sig = processed_data_signature()
    bundle = trained_model_store.fit_recommenders(ratings, movies, movie_content)
    trained_model_store.save_bundle(bundle, data_sig)
    print(f"Best alpha (F1): {bundle['best_alpha']:.2f}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
