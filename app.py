# -*- coding: utf-8 -*-
"""app"""

import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- PAGE CONFIG ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🎬", layout="wide")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('movies_lite.csv', on_bad_lines='skip')

        # Ensure numeric columns
        df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['year'] = df['release_date'].dt.year.fillna(0).astype(int)

        # Ensure text columns exist
        text_cols = ['genres', 'overview', 'title', 'credits']
        for col in text_cols:
            if col not in df.columns:
                df[col] = ''
            else:
                df[col] = df[col].fillna('')

        # Create content features: Title x2, Genres x1, Credits x2, Overview
        df['content_features'] = (
            (df['title'] + " ") * 2 +
            (df['genres'] + " ") * 1 +
            (df['credits'] + " ") * 2 +
            df['overview']
        )
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

movies = load_data()

# --- 2. TRAIN AI MODEL ---
@st.cache_resource
def train_model(data):
    if data.empty:
        return None
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['content_features'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim = train_model(movies)

# --- 3. HELPERS ---
def make_stars(score):
    count = int(round(score))
    return "⭐" * count + f" ({score:.1f})"

# Ensure 'genres' column exists
if 'genres' not in movies.columns:
    movies['genres'] = ''

if not movies.empty:
    all_genres = sorted(list(set(movies['genres'].str.split(', ').explode().dropna())))
    if '' in all_genres: all_genres.remove('')
else:
    all_genres = []

min_year_data = int(movies['year'].min()) if not movies.empty else 1980
max_year_data = int(movies['year'].max()) if not movies.empty else 2025

# --- 4. CONTENT-BASED RECOMMENDATION ---
def content_based_recommendations(search_query, min_rating=5, selected_genres=None, start_year=1980, end_year=2025, movies=movies, cosine_sim=cosine_sim):
    cols_display = ['title', 'year', 'star_rating', 'Why Shown?', 'genres', 'overview', 'id']
    empty_df = pd.DataFrame(columns=cols_display)
    
    if movies.empty:
        return empty_df, pd.Series(dtype=int)

    # Filter by rating, year, genres
    candidate_pool = movies[
        (movies['vote_average'] >= min_rating) &
        (movies['year'] >= start_year) &
        (movies['year'] <= end_year)
    ]
    if selected_genres:
        pattern = '|'.join(selected_genres)
        candidate_pool = candidate_pool[candidate_pool['genres'].str.contains(pattern, case=False, na=False)]

    if candidate_pool.empty:
        return empty_df, pd.Series(dtype=int)

    results = pd.DataFrame()

    if not search_query:
        results = candidate_pool.sort_values('vote_average', ascending=False).head(20).copy()
        results['Why Shown?'] = "🔥 Top Rated"
    else:
        mask = candidate_pool['content_features'].str.lower().str.contains(search_query.lower())
        keyword_matches = candidate_pool[mask].sort_values('vote_average', ascending=False).copy()

        if not keyword_matches.empty:
            results = keyword_matches.head(20)
            results['Why Shown?'] = f"Found match: '{search_query}'"
        else:
            all_titles = candidate_pool['title'].astype(str).tolist()
            matches = difflib.get_close_matches(search_query, all_titles, n=1, cutoff=0.4)
            if matches:
                exact_title = matches[0]
                idx = movies[movies['title'] == exact_title].index[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]
                movie_indices = [i[0] for i in sim_scores]
                recs = movies.iloc[movie_indices].copy()
                recs = recs[
                    (recs['vote_average'] >= min_rating) &
                    (recs['year'] >= start_year) &
                    (recs['year'] <= end_year)
                ]
                if selected_genres:
                    recs = recs[recs['genres'].str.contains(pattern, case=False, na=False)]
                results = recs.head(20)
                results['Why Shown?'] = f"Similar to: {exact_title}"
            else:
                return empty_df, pd.Series(dtype=int)

    results['star_rating'] = results['vote_average'].apply(make_stars)

    # Genre counts
    all_res_genres = results['genres'].str.split(', ').explode().dropna()
    all_res_genres = all_res_genres[all_res_genres != '']
    genre_counts = pd.Series(dtype=int)
    if not all_res_genres.empty:
        genre_counts = all_res_genres.value_counts().head(5)

    return results[cols_display], genre_counts

# --- 5. UI LAYOUT ---
st.title("🎬 CineMatch Pro")
st.markdown("Search by **Movie** or **Actor** (e.g., *Robert Downey Jr*, *Inception*).")

with st.sidebar:
    st.header("⚙️ Filters")
    min_rating = st.slider("Min Rating", 0.0, 10.0, 5.0, 0.5)
    year_range = st.slider("Year Range", min_year_data, max_year_data, (1980, max_year_data))
    selected_genres = st.multiselect("Genre", all_genres)

col1, col2 = st.columns([4, 1])
search_query = col1.text_input("Search", placeholder="Type an actor or movie...", label_visibility="collapsed")
search_btn = col2.button("🔍 Search", use_container_width=True, type="primary")

# --- 6. RUN CONTENT-BASED SEARCH ---
if search_btn or search_query:
    results, genre_counts = content_based_recommendations(
        search_query=search_query,
        min_rating=min_rating,
        selected_genres=selected_genres,
        start_year=year_range[0],
        end_year=year_range[1],
        movies=movies,
        cosine_sim=cosine_sim
    )

    if results.empty:
        st.warning("No movies found. Try adjusting your filters or search terms.")
    else:
        st.markdown("### 📊 Genre Breakdown in Results")
        if not genre_counts.empty:  # <-- FIXED
            st.bar_chart(genre_counts)

        st.markdown("### 🍿 Results")
        for _, row in results.iterrows():
            with st.container():
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown(f"**{row['year']}**")
                    st.markdown(row['star_rating'])
                    st.caption(row['Why Shown?'])
                with c2:
                    st.subheader(row['title'])
                    st.caption(row['genres'])
                    st.write(row['overview'][:200] + "...")
                    st.link_button("🎬 Watch Intro / Info", f"https://www.themoviedb.org/movie/{row['id']}")
                st.divider()
