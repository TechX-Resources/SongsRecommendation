import os
import json
import pickle
import pathlib as p
import gzip
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

from IPython.display import HTML, display
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import (
    train_test_split,
    precision_at_k,
    ndcg_at_k,
    mean_average_precision_at_k,
    AUC_at_k,
)

#Initial commit

# ==========================================================
# Configuration
# ==========================================================
DATA_DIR = p.Path(r"C:\Users\adtra\OneDrive\Documents\Tech X\Data")
SLICE_SIZE = 1_000  # each mpd.slice.*.json holds 1 k playlists
MAX_SLICES = 1_000  # stop after this many slices (≈ full dataset)

# Model hyper‑parameters
ALS_FACTORS = 100
ALS_REGULARISATION = 3.0
ALS_ALPHA = 5.0
ALS_ITERATIONS = 1

# Artifact locations (relative to notebook working dir)
MODEL_PATH = p.Path("als_model.npz")
MATRIX_PATH = p.Path("interaction_matrix.npz")
LOOKUPS_PATH = p.Path("lookup_dicts.pkl")

# Evaluation settings
K_RECOMMENDATIONS = 10


# ==========================================================
# Data‑loading helpers
# ==========================================================
def load_playlist_slice(file_path, keep=("pid", "tracks")):
    """Read one *mpd.slice.*.json* into a flat DataFrame."""
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    pl_df = (pd.json_normalize(data["playlists"])
               .loc[:, list(keep)]
               .explode("tracks", ignore_index=True))

    track_df = pd.json_normalize(pl_df["tracks"])
    track_df["playlist_id"] = pl_df["pid"].astype("int32")

    for col in ("artist_uri", "artist_name", "track_uri", "track_name"):
        track_df[col] = track_df[col].astype("category")

    return track_df[["playlist_id", "track_uri", "artist_uri", "artist_name", "track_name"]]


def load_mpd(base_path=DATA_DIR, pattern="*.zip"):
    """Extract and load playlist data from ZIP files containing mpd.slice.*.json."""
    dfs = []
    files = sorted(base_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern} in {base_path}")

    for i, zip_path in enumerate(files):
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for name in archive.namelist():
                if name.endswith('.json') and 'mpd.slice' in name:
                    with archive.open(name) as f:
                        data = json.load(f)
                    df = pd.json_normalize(data["playlists"])
                    df = df.loc[:, ["pid", "tracks"]]
                    df = df.explode("tracks", ignore_index=True)
                    track_df = pd.json_normalize(df["tracks"])
                    track_df["playlist_id"] = df["pid"].astype("int32")
                    for col in ("artist_uri", "artist_name", "track_uri", "track_name"):
                        track_df[col] = track_df[col].astype("category")
                    dfs.append(track_df[["playlist_id", "track_uri", "artist_uri", "artist_name", "track_name"]])
        if i % 5 == 0:
            print(f"✓ Loaded: {zip_path.name}")

    return pd.concat(dfs, ignore_index=True)


# ==========================================================
# Interaction matrix
# ==========================================================
def build_interaction_matrix(df):
    """Return (csr_matrix, track_uri→code, code→track_uri)."""
    # Guarantee categorical dtypes ----------------------------
    track_uri_cat = df["track_uri"].astype("category")
    user_codes = df["playlist_id"].astype("category").cat.codes
    item_codes = track_uri_cat.cat.codes
    data = np.ones(len(df), dtype=np.float32)

    csr = sp.coo_matrix((data, (user_codes, item_codes))).tocsr()

    uri2code = {uri: code for code, uri in enumerate(track_uri_cat.cat.categories)}
    code2uri = {code: uri for uri, code in uri2code.items()}

    return csr, uri2code, code2uri


# ==========================================================
# Model‑training pipeline
# ==========================================================
def train_als(interactions):
    """Train an implicit ALS model and return it."""
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARISATION,
        iterations=ALS_ITERATIONS,
        alpha=ALS_ALPHA,
        calculate_training_loss=True,
        num_threads=1,
    )
    model.fit(interactions)
    return model


# ==========================================================
# Evaluation helpers
# ==========================================================
def evaluate_model(model, train_mat, val_mat, k=K_RECOMMENDATIONS):
    """Compute ranking metrics (precision / NDCG / MAP / AUC)."""
    return pd.Series({
        f"precision@{k}": precision_at_k(model, train_mat, val_mat, k),
        f"ndcg@{k}": ndcg_at_k(model, train_mat, val_mat, k),
        f"map@{k}": mean_average_precision_at_k(model, train_mat, val_mat, k),
        f"auc@{k}": AUC_at_k(model, train_mat, val_mat, k),
    })


# ==========================================================
# Save & Load helpers
# ==========================================================
def save_artifacts(model, interactions, lookup_dicts):
    """Save model, matrix & auxiliary dicts to disk."""
    sp.save_npz(MATRIX_PATH, interactions)
    model.save(str(MODEL_PATH))
    LOOKUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOOKUPS_PATH.open("wb") as f:
        pickle.dump(lookup_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("💾 Artifacts saved:", MODEL_PATH, MATRIX_PATH, LOOKUPS_PATH)


def load_artifacts():
    """Load model, matrix & lookup dicts from disk."""
    from implicit.cpu.bpr import BayesianPersonalizedRanking  # lazy import

    model = BayesianPersonalizedRanking.load(str(MODEL_PATH)) if MODEL_PATH.exists() else None
    interactions = sp.load_npz(MATRIX_PATH) if MATRIX_PATH.exists() else None
    with LOOKUPS_PATH.open("rb") as f:
        lookup_dicts = pickle.load(f)
    return model, interactions, lookup_dicts


# ==========================================================
# Recommendation utilities (inference‑time)
# ==========================================================
def build_track_metadata(df):
    """Aggregate artist & track names – handy for searches & display."""
    return (df[["track_uri", "artist_name", "track_name"]]
              .value_counts()
              .reset_index()
              .set_index("track_uri"))


def build_track_to_artist(df):
    return (df[["track_uri", "artist_uri"]]
              .drop_duplicates()
              .set_index("track_uri")["artist_uri"].to_dict())


def search_music(track_metadata, query, search_type="artist", limit=10):
    """Case‑insensitive search over artists or tracks."""
    query = query.lower()
    col = "artist_name" if search_type == "artist" else "track_name"
    mask = track_metadata[col].str.lower().str.contains(query, regex=False)
    return (track_metadata[mask]
              .sort_values("count", ascending=False)
              .reset_index()
              .head(limit))


def recommend_similar_tracks(query_uri, model, track_uri_to_code, code_to_track_uri,
                             track_to_artist, track_metadata, top_n=10, oversample=50):
    """Return up to *top_n* tracks similar to *query_uri*, skipping same artist."""
    if query_uri not in track_uri_to_code:
        raise KeyError(f"Track {query_uri!r} not found in training data.")

    query_code = track_uri_to_code[query_uri]
    query_artist = track_to_artist.get(query_uri)

    ids, scores = model.similar_items(query_code, N=oversample)

    recs = []
    for code, score in zip(ids, scores):
        uri = code_to_track_uri[code]
        if uri == query_uri or track_to_artist.get(uri) == query_artist:
            continue  # same track or same artist → skip
        recs.append({
            "artist_name": track_metadata.loc[uri, "artist_name"],
            "track_name": track_metadata.loc[uri, "track_name"],
            "similarity": round(float(score), 4),
            "url": f"https://open.spotify.com/track/{uri.split(':')[-1]}",
        })
        if len(recs) == top_n:
            break

    return pd.DataFrame(recs)


def show_clickable(df):
    """Render a DataFrame in‑notebook with working <a> links."""
    html = df.to_html(index=False, escape=False, render_links=True)
    display(HTML(html))


# ==========================================================
# Main Function
# ==========================================================
def main():
    # === 1. Load and prepare dataset ===
    print("📥 Loading Spotify MPD...")
    df_full = load_mpd()

    print("📊 Building interaction matrix...")
    interactions_csr, uri2code, code2uri = build_interaction_matrix(df_full)

    print("🧠 Training ALS model...")
    als_model = train_als(interactions_csr)

    print("🗂️ Building track metadata...")
    track_metadata = build_track_metadata(df_full)
    track_to_artist = build_track_to_artist(df_full)

    # === 2. Save artifacts ===
    print("💾 Saving model & data...")
    save_artifacts(
        model=als_model,
        interactions=interactions_csr,
        lookup_dicts={
            "track_uri_to_code": uri2code,
            "code_to_track_uri": code2uri,
            "track_metadata": track_metadata,
            "track_to_artist": track_to_artist
        },
    )

    print("✅ Done.")

if __name__ == "__main__":
    main()