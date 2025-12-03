# app.py
import streamlit as st
import pandas as pd
import pickle
import re
from difflib import get_close_matches
from typing import List, Tuple, Optional

# ----------------------------
# Helpers: text preprocessing
# ----------------------------
def simple_clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.replace("/", " ").replace("|", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def preprocess_genres(val) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, (list, tuple)):
        return " ".join([str(x).strip().replace(" ", "_") for x in val if x])
    s = str(val)
    parts = re.split(r"[,/|;]+", s)
    parts = [p.strip().replace(" ", "_") for p in parts if p.strip()]
    return " ".join(parts)

def build_combined_text(df: pd.DataFrame, text_cols: List[str]) -> pd.Series:
    def join_row(row):
        parts = []
        for c in text_cols:
            if c not in row or pd.isna(row[c]):
                continue
            if c.lower() in ("genre", "genres"):
                parts.append(preprocess_genres(row[c]))
            else:
                parts.append(simple_clean_text(str(row[c])))
        return " ".join(parts).strip()
    return df.apply(join_row, axis=1)

def detect_text_columns(df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    candidates = ["name","title","english name","anime","anime title","Name"]
    cols_lower = {c.lower(): c for c in df.columns}
    title_col = None
    for cand in candidates:
        if cand in cols_lower:
            title_col = cols_lower[cand]
            break
    if not title_col:
        for c in df.columns:
            if df[c].dtype == object:
                title_col = c
                break
    text_cols = [title_col] if title_col else []
    for c in ["english name","other name","genres","genre","type","synopsis","plot","description","overview"]:
        if c in cols_lower:
            text_cols.append(cols_lower[c])
    if len(text_cols) <= 1:
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # dedupe preserving order
    seen = set(); out = []
    for c in text_cols:
        if c and c not in seen:
            out.append(c); seen.add(c)
    return title_col, out

# ----------------------------
# Load dataset & artifacts (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_dataset(path: str = "anime-dataset-2023.csv") -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_artifacts(path: str = "anime_recommender_artifacts.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    tfidf = data.get("tfidf")
    nn = data.get("nn")
    saved_title_col = data.get("title_col", None)
    return tfidf, nn, saved_title_col

# ----------------------------
# Recommendation helpers
# ----------------------------
def find_best_title_match(query: str, title_to_index: pd.Series) -> Optional[str]:
    if not query:
        return None
    if query in title_to_index.index:
        return query
    for t in title_to_index.index:
        if t.lower() == query.lower():
            return t
    matches = get_close_matches(query, title_to_index.index.tolist(), n=6, cutoff=0.45)
    if matches:
        return matches[0]
    contains = [t for t in title_to_index.index if query.lower() in t.lower()]
    if contains:
        return contains[0]
    return None

def get_recommendations(title: str, df: pd.DataFrame, tfidf_matrix, nn_model, title_to_index: pd.Series, topn: int = 10):
    best_title = find_best_title_match(title, title_to_index)
    if best_title is None:
        return None, None
    idx = int(title_to_index[best_title])
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=topn+1)
    distances = distances.flatten()[1:]
    indices = indices.flatten()[1:]
    sims = 1 - distances
    result = df.iloc[indices].copy().reset_index(drop=True)
    result["similarity"] = sims
    return best_title, result

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Anime Recommender", layout="wide")
st.title("🎬 Anime Recommendation System")
# (hidden) st.write("Content-based engine using TF-IDF + cosine similarity. (Model trained in Jupyter Notebook)")

# ----------------------------
# Hard-coded configuration (no sidebar)
# ----------------------------
csv_path = "anime-dataset-2023.csv"
artifacts_path = "anime_recommender_artifacts.pkl"
topn = 10

# Load dataset and artifacts (silent failures stop without exposing error text)
try:
    df = load_dataset(csv_path)
except Exception:
    # silent fail-stop (no detailed message shown)
    st.stop()

try:
    tfidf, nn_model, saved_title_col = load_artifacts(artifacts_path)
except Exception:
    st.stop()

# Build combined_text if missing (no user-facing info)
if "combined_text" not in df.columns:
    detected_title_col, text_cols = detect_text_columns(df)
    df["combined_text"] = build_combined_text(df, text_cols)
else:
    detected_title_col, _ = detect_text_columns(df)

# Ensure artifacts contain both tfidf and nn (silent stop if not present)
if tfidf is None or nn_model is None:
    st.stop()

# Transform combined_text into TF-IDF matrix using loaded vectorizer
try:
    tfidf_matrix = tfidf.transform(df["combined_text"].fillna("").astype(str).tolist())
except Exception:
    st.stop()

# Determine title->index mapping
title_col = saved_title_col if saved_title_col and saved_title_col in df.columns else detected_title_col
if not title_col or title_col not in df.columns:
    title_list = df.index.astype(str).tolist()
    title_to_index = pd.Series(df.index.values, index=df.index.astype(str)).drop_duplicates()
else:
    title_list = df[title_col].astype(str).tolist()
    title_to_index = pd.Series(df.index.values, index=df[title_col].astype(str)).drop_duplicates()

# Input controls
st.subheader("Search for an anime")
col1, col2 = st.columns([4,1])
with col1:
    select_choice = st.selectbox("Pick a title (quick) — or type below", options=[""] + title_list[:2000], index=0)
    typed_query = st.text_input("Or type title (fuzzy search)", "")
    query = typed_query.strip() if typed_query.strip() else (select_choice if select_choice else "")
with col2:
    recommend_clicked = st.button("Recommend")

# One-line note about similarity
st.caption("Similarity: value ranges 0 → 1.0 — closer to 1.0 means more similar (1.0 = identical).")

# On click: compute and display recommendations
if recommend_clicked:
    if not query:
        st.warning("Please select or type an anime title.")
    else:
        best, recs = get_recommendations(query, df, tfidf_matrix, nn_model, title_to_index, topn=topn)
        if recs is None:
            # Silent: no raw error; provide friendly guidance
            st.error("Anime not found. Try a different spelling or choose from the dropdown.")
            try:
                suggestions = get_close_matches(query, title_list, n=6, cutoff=0.35)
                if suggestions:
                    st.info("Did you mean:")
                    for s in suggestions:
                        st.write(" - " + s)
            except Exception:
                pass
        else:
            st.success(f"Best match: **{best}** — showing top {len(recs)} results")
            # Determine columns to show: Name/title, Genre, Synopsis, Similarity
            possible_title_cols = [title_col, "Name", "name", "title"]
            title_display_col = next((c for c in possible_title_cols if c and c in recs.columns), None)

            possible_genre_cols = ["genres", "genre", "Genres", "Genre"]
            genre_col = next((c for c in possible_genre_cols if c in recs.columns), None)

            possible_syn_cols = ["synopsis", "Synopsis", "description", "Description", "Overview", "overview"]
            syn_col = next((c for c in possible_syn_cols if c in recs.columns), None)

            # Build final display columns
            display_cols = []
            if title_display_col:
                display_cols.append(title_display_col)
            if genre_col:
                display_cols.append(genre_col)
            if syn_col:
                display_cols.append(syn_col)
            # include similarity always
            display_cols.append("similarity")

            # fallback: if none found, use first 3 columns plus similarity
            if not (title_display_col or genre_col or syn_col):
                fallback = list(recs.columns[:3])
                display_cols = fallback + (["similarity"] if "similarity" in recs.columns else [])

            # Ensure columns exist & unique
            seen = set()
            display_cols = [c for c in display_cols if c in recs.columns and not (c in seen or seen.add(c))]

            # Truncate synopsis for tidy table view (adjust length if you want full text)
            if syn_col and syn_col in recs.columns:
                recs = recs.copy()
                recs[syn_col] = recs[syn_col].fillna("").astype(str).apply(lambda s: (s[:350] + "...") if len(s) > 350 else s)

            # Reorder columns so they appear Name | Genre | Synopsis | Similarity
            ordered_cols = []
            if title_display_col and title_display_col in display_cols:
                ordered_cols.append(title_display_col)
            if genre_col and genre_col in display_cols:
                ordered_cols.append(genre_col)
            if syn_col and syn_col in display_cols:
                ordered_cols.append(syn_col)
            if "similarity" in display_cols:
                ordered_cols.append("similarity")
            # If ordered_cols empty (edge case), just use display_cols
            if not ordered_cols:
                ordered_cols = display_cols

            # Display dataframe
            st.dataframe(recs[ordered_cols].reset_index(drop=True).style.format({"similarity": "{:.4f}"}))
