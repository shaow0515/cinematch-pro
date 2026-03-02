# -*- coding: utf-8 -*-
"""CineMatch Pro - Full App"""

import streamlit as st
import pandas as pd
import difflib
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- PAGE CONFIG ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🎬", layout="wide")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('TMDB_movie_dataset_v11.csv', on_bad_lines='skip', engine='python')

        df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
        df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
        df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)

        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['year'] = df['release_date'].dt.year.fillna(0).astype(int)

        df = df.sort_values('vote_count', ascending=False).head(5000).copy()  # Lite Mode

        text_cols = ['genres', 'overview', 'title', 'credits']
        for col in text_cols:
            if col not in df.columns: df[col] = ''
            else: df[col] = df[col].fillna('')

        # --- CLEAN CREDITS: extract actor/director names ---
        def extract_names(credit_str):
            try:
                items = ast.literal_eval(credit_str)
                names = [i.get('name','') for i in items if isinstance(i, dict)]
                return ', '.join(names)
            except:
                return ''

        df['credits'] = df['credits'].apply(extract_names)

        # --- CLEAN GENRES ---
        df['genres'] = df['genres'].str.replace('[','').str.replace(']','').str.replace("'", "")

        # --- CREATE CONTENT FEATURES ---
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

# --- 2. TRAIN MODEL ---
@st.cache_resource
def train_model(data):
    if data.empty: return None
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['content_features'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim = train_model(movies)

# --- HELPERS ---
def make_stars(score):
    count = int(round(score))
    return "⭐" * count + f" ({score:.1f})"

all_genres = sorted(list(set(movies['genres'].str.split(', ').explode().dropna())))
if '' in all_genres: all_genres.remove('')

min_year_data = int(movies['year'].min()) if not movies.empty else 1980
max_year_data = int(movies['year'].max()) if not movies.empty else 2025

# --- 3. CONTENT-BASED RECOMMENDER ---
def get_recommendations(search_query, min_rating, selected_genres, start_year, end_year, movies, cosine_sim):
    if movies.empty: return pd.DataFrame(), {}

    candidate_pool = movies[
        (movies['vote_average'] >= min_rating) &
        (movies['year'] >= start_year) &
        (movies['year'] <= end_year)
    ]

    if selected_genres:
        pattern = '|'.join(selected_genres)
        candidate_pool = candidate_pool[candidate_pool['genres'].str.contains(pattern, case=False, na=False)]

    results = pd.DataFrame()

    if not search_query:
        results = candidate_pool.sort_values('vote_average', ascending=False).head(20).copy()
        results['Why Shown?'] = "🔥 Top Rated"
    else:
        # --- search in content features (title, actor, director, overview, genres) ---
        mask = candidate_pool['content_features'].str.lower().str.contains(search_query.lower())
        keyword_matches = candidate_pool[mask].sort_values('vote_average', ascending=False).copy()

        if not keyword_matches.empty:
            results = keyword_matches.head(20)
            results['Why Shown?'] = f"Found match: '{search_query}'"
        else:
            # --- fuzzy title match if no keyword matches ---
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

    if results.empty: return pd.DataFrame(), {}

    all_res_genres = results['genres'].str.split(', ').explode().dropna()
    all_res_genres = all_res_genres[all_res_genres != '']

    genre_counts = {}
    if not all_res_genres.empty:
        genre_counts = all_res_genres.value_counts().head(5)

    return results, genre_counts

# --- 4. UI LAYOUT ---
st.title("🎬 CineMatch Pro")
st.markdown("Search by **Movie**, **Actor**, or **Director** (e.g., *Robert Downey Jr*, *Inception*).")

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.header("⚙️ Filters")
    min_rating = st.slider("Min Rating", 0.0, 10.0, 5.0, 0.5)
    year_range = st.slider("Year Range", min_year_data, max_year_data, (1980, max_year_data))
    selected_genres = st.multiselect("Genre", all_genres)

# --- MAIN SEARCH AREA ---
col1, col2 = st.columns([4, 1])
search_query = col1.text_input("Search", placeholder="Type an actor, director, or movie...", label_visibility="collapsed")
search_btn = col2.button("🔍 Search", use_container_width=True, type="primary")

# --- RUN RECOMMENDATION ---
if search_btn or search_query:
    results, genre_counts = get_recommendations(
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
        # Bar chart for top genres
        st.markdown("### 📊 Genre Breakdown in Results")
        if genre_counts:
            st.bar_chart(genre_counts)

        # Display movie cards
        st.markdown("### 🍿 Results")
        for _, row in results.iterrows():
            with st.container():
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown(f"**{row['year']}**")
                    st.markdown(make_stars(row['vote_average']))
                    st.caption(row['Why Shown?'])
                with c2:
                    st.subheader(row['title'])
                    st.caption(row['genres'])
                    st.write(row['overview'][:200] + "...")
                    st.link_button("🎬 Watch Intro / Info", f"https://www.themoviedb.org/movie/{row['id']}")
                st.divider()
