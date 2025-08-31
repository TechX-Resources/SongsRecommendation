#!/usr/bin/env python3
"""
Hybrid CF + GPT recommender

Usage:
  python hybrid_cf_gpt_recommender.py --playlist playlist.json --profile profile.json [--rerank-with-gpt]

Requires:
  - OPENAI_API_KEY environment variable set
  - ALS model file (implicit saved model or .npz with item_factors)
  - lookup_dicts.pkl OR MPD zip files path to build minimal metadata
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI client (the "OpenAI" import / interface)
from openai import OpenAI

# --------- CONFIG ----------
ALS_MODEL_PATH = Path("als_model.npz")           # try this first (change if needed)
LOOKUPS_PATH = Path("lookup_dicts.pkl")          # optional, faster if present
MPD_ZIP_DIR = Path(r"C:\Users\adtra\OneDrive\Documents\Tech X\Data")  # point to MPD zip folder if you need metadata
GPT_MODEL = "gpt-4o-mini"                        # recommended
MAX_CANDIDATES = 200
PER_TRACK_CANDIDATES = 40
TOP_N = 20
MAX_HISTORY_LINES = 40
MAX_PLAYLIST_LINES = 40
TEMPERATURE = 0.2
# --------------------------

def init_openai():
    key = key
    if not key:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=key)

# ---------- Load ALS ----------
def load_als_artifact(path: Path):
    """
    Tries to:
     - load an implicit.AlternatingLeastSquares saved model (npz-like) by checking for 'user_factors'/'item_factors'
     - or load a .npz with arrays user_factors/item_factors
    Returns (item_factors numpy array, metadata note)
    """
    if not path.exists():
        raise FileNotFoundError(f"ALS model not found at: {path}")

    # Try np.load for arrays
    try:
        arr = np.load(str(path), allow_pickle=True)
        if 'item_factors' in arr:
            item_factors = arr['item_factors']
            return item_factors, "npz arrays"
    except Exception:
        pass

    # Try implicit AlternatingLeastSquares.load (the implicit model.save writes a .npz file but must be loaded via implicit)
    try:
        from implicit.als import AlternatingLeastSquares
        model = AlternatingLeastSquares()
        model.load(str(path))
        # model.item_factors is (n_items, factors)
        return model.item_factors, "implicit model"
    except Exception as e:
        raise RuntimeError(f"Could not load ALS model at {path}: {e}")

# ---------- Lookup / metadata ----------
def load_lookups_or_build(lookups_path: Path, mpd_zip_dir: Path):
    """
    Returns a dict with:
      - 'track_uri_to_code', 'code_to_track_uri', 'track_metadata' (pandas-like dict)
    If lookup file exists, load it. Otherwise tries to build minimal metadata by scanning MPD zip files.
    """
    if lookups_path.exists():
        with lookups_path.open("rb") as f:
            lookups = pickle.load(f)
        return lookups

    # Minimal builder: scan MPD zip files for track name / artist name / track uri -> code mapping
    import zipfile
    import json
    import pandas as pd

    files = sorted(mpd_zip_dir.glob("*.zip"))
    if not files:
        raise FileNotFoundError(f"No MPD zip files found under {mpd_zip_dir} and no lookup dict provided.")
    rows = []
    for z in files:
        with zipfile.ZipFile(z, 'r') as archive:
            for name in archive.namelist():
                if name.endswith(".json") and "mpd.slice" in name:
                    with archive.open(name) as fh:
                        data = json.load(fh)
                    pl = pd.json_normalize(data["playlists"])
                    pl = pl.loc[:, ["pid", "tracks"]].explode("tracks", ignore_index=True)
                    track_df = pd.json_normalize(pl["tracks"])
                    track_df["playlist_id"] = pl["pid"].astype("int32")
                    for col in ("artist_uri", "artist_name", "track_uri", "track_name"):
                        if col in track_df.columns:
                            track_df[col] = track_df[col].astype(str)
                    rows.append(track_df[["track_uri", "artist_name", "track_name"]])
        # keep it modest: stop after a few zips to build mapping quick
        if len(rows) >= 10:
            break
    all_tracks = pd.concat(rows, ignore_index=True)
    # dedupe
    all_tracks = all_tracks.drop_duplicates(subset=["track_uri"])
    track_uri_to_code = {uri: i for i, uri in enumerate(all_tracks["track_uri"].tolist())}
    code_to_track_uri = {i: uri for uri, i in track_uri_to_code.items()}
    # build metadata dict-like mapping
    track_metadata = {}
    for _, r in all_tracks.iterrows():
        track_metadata[r["track_uri"]] = {"track_name": r["track_name"], "artist_name": r["artist_name"]}
    lookups = {
        "track_uri_to_code": track_uri_to_code,
        "code_to_track_uri": code_to_track_uri,
        "track_metadata": track_metadata,
        "track_to_artist": {k: v["artist_name"] for k, v in track_metadata.items()}
    }
    # save for future runs
    with lookups_path.open("wb") as f:
        pickle.dump(lookups, f)
    return lookups

# ---------- Playlist enrichment ----------
def enrich_playlist_from_uris(uris, track_metadata):
    out = []
    for uri in uris:
        meta = track_metadata.get(uri, {})
        out.append({
            "uri": uri,
            "track_name": meta.get("track_name") or None,
            "artist_name": meta.get("artist_name") or None
        })
    return out

# ---------- Candidate retrieval ----------
def playlist_vector_from_uris(uris, track_uri_to_code, item_factors):
    indices = [track_uri_to_code.get(u) for u in uris if u in track_uri_to_code]
    if not indices:
        raise ValueError("None of the provided URIs were found in track_uri_to_code mapping.")
    vecs = item_factors[indices]
    # mean pooling, L2-normalize
    mean_vec = np.nanmean(vecs, axis=0)
    if np.all(np.isnan(mean_vec)):
        raise ValueError("Computed empty playlist vector.")
    # safe normalization
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec.reshape(1, -1)

def get_top_candidates_by_cosine(playlist_vec, item_factors, top_k=200, exclude_uris=set(), code_to_track_uri=None):
    # Normalize item factors
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    safe = np.where(norms==0, 1.0, norms)
    item_normed = item_factors / safe
    sims = (item_normed @ playlist_vec.T).reshape(-1)
    # argsort descending
    top_idx = np.argsort(sims)[-top_k:][::-1]
    uris_scores = []
    for idx in top_idx:
        uri = code_to_track_uri[idx] if code_to_track_uri else str(idx)
        if uri in exclude_uris:
            continue
        uris_scores.append((uri, float(sims[idx])))
    return uris_scores

# ---------- GPT helpers ----------
def summarize_playlist_lines(playlist_meta, max_lines=MAX_PLAYLIST_LINES):
    lines = []
    for p in playlist_meta[:max_lines]:
        t = p.get("track_name") or "Unknown Track"
        a = p.get("artist_name") or "Unknown Artist"
        lines.append(f"- {t} — {a}")
    return "\n".join(lines) if lines else "None"

def summarize_history_lines(history, max_lines=MAX_HISTORY_LINES):
    if not history:
        return "None"
    return "\n".join(history[:max_lines])

def generate_personalized_query_gpt(client: OpenAI, user_profile: dict, playlist_meta: list):
    history_text = summarize_history_lines(user_profile.get("history", []))
    prefs = user_profile.get("preferences", {})
    prefs_text = "\n".join(f"{k}: {v}" for k, v in prefs.items()) if isinstance(prefs, dict) and prefs else (str(prefs) if prefs else "None")
    playlist_text = summarize_playlist_lines(playlist_meta)
    system_msg = ("You are a music recommendation assistant. Produce a single concise descriptive query (6-12 words) "
                  "that captures the music the user is most likely to want next. Mention mood/genre/instrument/era if helpful.")
    user_msg = f"""User History:
{history_text}

User Preferences:
{prefs_text}

Current Playlist (sample):
{playlist_text}

Return ONLY a one-line descriptive query, e.g. "dreamy indie pop with lo-fi beats and soft vocals"."""
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"system","content":system_msg},
                  {"role":"user","content":user_msg}],
        temperature=TEMPERATURE,
        max_tokens=80
    )
    text = resp.choices[0].message.content.strip()
    return " ".join(text.splitlines()).strip()

def rerank_with_gpt_indices(client: OpenAI, personalized_query: str, candidates_meta: list):
    """
    Ask GPT to return indices (1-based) in desired order.
    """
    numbered = "\n".join([f"{i+1}. {c['track_name']} — {c['artist_name']}" for i,c in enumerate(candidates_meta)])
    system_msg = ("You are a music ranking assistant. Given a short query and a numbered list of candidate songs, "
                  "return only a comma-separated or newline-separated ordered list of indices (numbers) representing best-to-worst order.")
    user_msg = f"""Query:
"{personalized_query}"

Candidates:
{numbered}

Return only indices in the preferred order (e.g. 3,1,2,4) and nothing else."""
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
        temperature=TEMPERATURE,
        max_tokens=300
    )
    out = resp.choices[0].message.content.strip()
    # parse numbers
    tokens = [t.strip() for t in out.replace(",", "\n").splitlines() if t.strip()]
    indices = []
    for t in tokens:
        try:
            i = int(t)
            indices.append(i-1)
        except:
            # try first token
            try:
                i = int(t.split()[0])
                indices.append(i-1)
            except:
                continue
    return indices

# ---------- Main pipeline ----------
def run_pipeline(playlist_uris, user_profile, rerank_with_gpt=False,
                 als_model_path=ALS_MODEL_PATH, lookups_path=LOOKUPS_PATH):
    print("Loading ALS model ...")
    item_factors, note = load_als_artifact(Path(als_model_path))
    print(f"Loaded ALS item_factors ({item_factors.shape}) via {note}.")

    print("Loading or building lookups ...")
    lookups = load_lookups_or_build(Path(lookups_path), MPD_ZIP_DIR)
    uri2code = lookups["track_uri_to_code"]
    code2uri = lookups["code_to_track_uri"]
    track_metadata = lookups["track_metadata"]
    track_to_artist = lookups.get("track_to_artist", {})

    # --- Convert URLs to URIs and filter unknown tracks ---
    def url_to_uri(track_str):
        if track_str.startswith("spotify:track:"):
            return track_str
        if "open.spotify.com/track/" in track_str:
            return "spotify:track:" + track_str.split("/")[-1].split("?")[0]
        return track_str

    playlist_uris = [url_to_uri(u) for u in playlist_uris]
    matched_uris = [u for u in playlist_uris if u in uri2code]
    skipped_uris = [u for u in playlist_uris if u not in uri2code]

    if skipped_uris:
        print(f"Warning: {len(skipped_uris)} tracks were not found in ALS model and will be skipped:")
        for u in skipped_uris:
            print("  -", u)

    if not matched_uris:
        raise ValueError("None of the playlist tracks were found in the ALS model. Cannot continue.")


    playlist_uris = matched_uris  # overwrite with filtered list

    print("Enriching playlist metadata ...")
    playlist_meta = enrich_playlist_from_uris(playlist_uris, track_metadata)
    print(f"Playlist contains {len(playlist_meta)} items (enriched).")

    # generate personalized query via GPT
    client = init_openai()
    print("Generating personalized query with GPT ...")
    personalized_query = generate_personalized_query_gpt(client, user_profile, playlist_meta)
    print(f"> Personalized query: {personalized_query}")

    # compute playlist vector (requires mapping URIs -> codes)
    print("Computing playlist vector from ALS item_factors ...")
    playlist_vec = playlist_vector_from_uris(playlist_uris, uri2code, item_factors)

    # get top candidate URIs by cosine similarity
    print("Finding top candidate tracks by cosine similarity ...")
    candidates = get_top_candidates_by_cosine(
        playlist_vec, item_factors, top_k=MAX_CANDIDATES, exclude_uris=set(playlist_uris), code_to_track_uri=code2uri
    )
    candidates_meta = []
    for uri, score in candidates:
        meta = track_metadata.get(uri, {"track_name": uri, "artist_name": ""})
        candidates_meta.append({
            "uri": uri,
            "track_name": meta.get("track_name", uri),
            "artist_name": meta.get("artist_name", ""),
            "score": score
        })
    print(f"> Retrieved {len(candidates_meta)} candidates.")

    # optional GPT rerank
    final_order = list(range(len(candidates_meta)))
    if rerank_with_gpt and candidates_meta:
        print("Asking GPT to re-rank candidates (indices) ...")
        indices = rerank_with_gpt_indices(client, personalized_query, candidates_meta)
        if indices and set(indices) <= set(range(len(candidates_meta))):
            final_order = [i for i in indices if 0 <= i < len(candidates_meta)]
        else:
            print("GPT rerank parsing failed or incomplete; falling back to CF order.")

    # build final list
    final = []
    for idx in final_order[:TOP_N]:
        c = candidates_meta[idx]
        spotify_url = f"https://open.spotify.com/track/{c['uri'].split(':')[-1]}"
        final.append({"uri": c["uri"], "label": f"{c['track_name']} - {c['artist_name']}", "url": spotify_url})
    print("\n=== FINAL RECOMMENDATIONS ===")
    for i, item in enumerate(final, start=1):
        print(f"{i}. {item['label']} — {item['url']}")
    return final

    


# ---------- CLI ----------
def load_playlist_uris(json_path):
    """
    Load Spotify track URIs from many common playlist JSON shapes:
    - List of objects with top-level 'track_uri' (MPD-style)
    - Spotify Web API shape: tracks.items[].track.uri or items[].track.uri
    - List of strings (URIs, URLs, or bare 22-char IDs)
    - Top-level dict with tracks/items
    - Objects containing 'external_urls': {'spotify': 'https://open.spotify.com/track/...'}
    """
    import re

    def as_uri(x):
        # string cases
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("spotify:track:"):
                return s
            if "open.spotify.com/track/" in s:
                return "spotify:track:" + s.split("/track/")[-1].split("?")[0].split("/")[-1]
            # bare 22-char Spotify ID
            if re.fullmatch(r"[A-Za-z0-9]{22}", s):
                return f"spotify:track:{s}"
            return None

        # dict cases
        if isinstance(x, dict):
            # nested 'track'
            if "track" in x:
                u = as_uri(x["track"])
                if u: return u
            # common direct fields
            for k in ("track_uri", "uri"):
                v = x.get(k)
                if isinstance(v, str):
                    u = as_uri(v)
                    if u: return u
            # external URL
            ext = x.get("external_urls")
            if isinstance(ext, dict):
                v = ext.get("spotify")
                if isinstance(v, str):
                    u = as_uri(v)
                    if u: return u
            # bare id
            v = x.get("id")
            if isinstance(v, str):
                u = as_uri(v)
                if u: return u
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    uris = []

    if isinstance(data, dict):
        # try Spotify API-like containers
        container = data.get("tracks", data)
        items = container.get("items") if isinstance(container, dict) else None
        if isinstance(items, list):
            for it in items:
                u = as_uri(it)
                if u: uris.append(u)
        else:
            u = as_uri(container)
            if u: uris.append(u)

    elif isinstance(data, list):
        for it in data:
            u = as_uri(it)
            if u: uris.append(u)

    else:
        u = as_uri(data)
        if u: uris.append(u)

    if not uris:
        raise ValueError(
            "Could not find any track URIs in the provided JSON "
            "(looked for 'track_uri', 'uri', nested 'track', external_urls, bare ID, or URL)."
        )

    # de-duplicate, preserve order
    seen, unique = set(), []
    for u in uris:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--playlist", required=True, help="Path to playlist JSON (list of URIs or objects with 'uri')")
    parser.add_argument("--profile", required=False, help="User profile JSON (history list + preferences dict)")
    parser.add_argument("--rerank-with-gpt", action="store_true", help="Ask GPT to rerank candidates (safer: returns indices)")
    parser.add_argument("--als", default=str(ALS_MODEL_PATH))
    parser.add_argument("--lookups", default=str(LOOKUPS_PATH))
    args = parser.parse_args()

    playlist = load_playlist_uris(args.playlist)
    if args.profile:
        profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    else:
        profile = {"history": ["Beach House — Space Song", "Tame Impala — The Less I Know The Better"], "preferences": {"mood":"dreamy, chill"}}

    run_pipeline(playlist, profile, rerank_with_gpt=args.rerank_with_gpt, als_model_path=args.als, lookups_path=args.lookups)
