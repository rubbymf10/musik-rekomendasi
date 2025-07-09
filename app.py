import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Styling dark mode ---
st.markdown("""
    <style>
    .main, .block-container {
        background-color: #121212;
        color: #FFFFFF;
    }
    h1, h2, h3, h4 {
        color: #1DB954;
    }
    button[kind="primary"] {
        background-color: #1DB954;
        color: #FFFFFF;
        border-radius: 20px;
        border: none;
        padding: 8px 20px;
    }
    button[kind="primary"]:hover {
        background-color: #1ed760;
    }
    .stTextInput>div>div>input {
        background-color: #222222;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 8px;
    }
    .music-card {
        background-color: #282828;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .music-card:hover {
        background-color: #333333;
    }
    .music-cover {
        width: 50px;
        height: 50px;
        color: #1DB954;
        font-size: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 6px;
        flex-shrink: 0;
    }
    .music-info {
        flex-grow: 1;
    }
    .music-title {
        font-weight: 600;
        font-size: 16px;
        margin: 0;
    }
    .music-artist {
        color: #b3b3b3;
        margin: 0;
        font-size: 14px;
    }
    .popularity {
        color: #1DB954;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('musik.csv')
    df_clean = df.dropna(subset=['popularity', 'genre', 'subgenre', 'tempo', 'duration_ms', 'energy', 'danceability'])
    low_thresh = df_clean['popularity'].quantile(0.33)
    high_thresh = df_clean['popularity'].quantile(0.66)

    def categorize_popularity(pop):
        if pop <= low_thresh:
            return 'Low'
        elif pop > high_thresh:
            return 'High'
        else:
            return np.nan

    df_clean['pop_category'] = df_clean['popularity'].apply(categorize_popularity)
    df_clean = df_clean.dropna(subset=['pop_category'])
    label_enc = LabelEncoder()
    df_clean['pop_encoded'] = label_enc.fit_transform(df_clean['pop_category'])
    return df, df_clean, label_enc

df, df_clean, label_enc = load_data()

# --- Train Model ---
@st.cache_resource
def train_model(df_clean):
    tfidf_genre = TfidfVectorizer()
    tfidf_subgenre = TfidfVectorizer()
    tfidf_title = TfidfVectorizer()
    tfidf_artist = TfidfVectorizer()
    tfidf_lyrics = TfidfVectorizer(max_features=500)
    tfidf_album = TfidfVectorizer()

    genre_tfidf = tfidf_genre.fit_transform(df_clean['genre'])
    subgenre_tfidf = tfidf_subgenre.fit_transform(df_clean['subgenre'])
    title_tfidf = tfidf_title.fit_transform(df_clean['judul_musik'])
    artist_tfidf = tfidf_artist.fit_transform(df_clean['artist'])
    lyrics_tfidf = tfidf_lyrics.fit_transform(df_clean['lyrics'].fillna(''))
    album_tfidf = tfidf_album.fit_transform(df_clean['album'])

    features_num = ['tempo', 'duration_ms', 'energy', 'danceability']
    scaler = MinMaxScaler()
    df_num_scaled = pd.DataFrame(scaler.fit_transform(df_clean[features_num]), columns=features_num, index=df_clean.index)

    X = pd.concat([df_genre, df_subgenre, df_title, df_artist, df_lyrics, df_album, df_num_scaled], axis=1)
    y = df_clean['pop_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, tfidf_genre, tfidf_subgenre, tfidf_title, tfidf_artist, tfidf_lyrics, tfidf_album, scaler, X_test, y_test, y_pred, title_tfidf

model, tfidf_genre, tfidf_subgenre, tfidf_title, tfidf_artist, tfidf_lyrics, tfidf_album, scaler, X_test, y_test, y_pred, title_tfidf = train_model(df_clean)

# --- Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'recommendation_table' not in st.session_state:
    st.session_state.recommendation_table = pd.DataFrame()

# --- Sidebar ---
with st.sidebar:
    st.markdown('<h2 style="color:#1DB954; margin-bottom: 15px;">\U0001F3B5 Dashboard</h2>', unsafe_allow_html=True)
    halaman = st.radio("", ["Home", "Music Recommendation", "History"], index=0, key="page_select")

# --- Music Card UI ---
def music_card(title, artist, popularity):
    st.markdown(f"""
    <div class="music-card">
        <div class="music-cover">🎵</div>
        <div class="music-info">
            <p class="music-title">{title}</p>
            <p class="music-artist">{artist}</p>
        </div>
        <div class="popularity">{int(popularity)}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Home Page ---
if halaman == "Home":
    st.header("Top 10 Most Popular Music")
    top10 = df.sort_values(by='popularity', ascending=False).head(10)
    for _, row in top10.iterrows():
        music_card(row['judul_musik'], row['artist'], row['popularity'])

    st.markdown("---")
    st.header("Top 5 Popular Songs from Each Genre")
    genre_list = df['genre'].dropna().unique()
    for genre in genre_list:
        st.subheader(f"🎶 Genre: {genre}")
        top5_by_genre = df[df['genre'] == genre].sort_values(by='popularity', ascending=False).head(5)
        for _, row in top5_by_genre.iterrows():
            music_card(row['judul_musik'], row['artist'], row['popularity'])

# --- History Page ---
if halaman == "History":
    st.header("Recommendation Search History")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(f"- **{h['Judul']}** by {h['Artis']} (Genre: {h['Genre']}, Prediction: {h['Prediksi']})")
    else:
        st.info("No search history yet.")

    st.markdown("---")
    st.header("🎧 Recommendation Results")
    if not st.session_state.recommendation_table.empty:
        df_show = st.session_state.recommendation_table.sort_values(by='popularity', ascending=False)
        for _, row in df_show.iterrows():
            music_card(row['judul_musik'], row['artist'], row['popularity'])
    else:
        st.info("No genre-based recommendations shown.")

    if st.button("Reset Search History"):
        st.session_state.history = []
        st.session_state.recommendation_table = pd.DataFrame()
        st.experimental_rerun()
        st.stop()

# --- Music Recommendation Page ---
if halaman == "Music Recommendation":
    st.header("Music Recommendation Based on Title")

    judul_list = df_clean['judul_musik'].dropna().unique()
    pilihan = st.selectbox("Choose from the list of music titles", options=judul_list)
    manual_input = st.text_input("Or type the title manually (optional)")
    judul = manual_input if manual_input.strip() else pilihan

    if st.button("Recommend"):
        if not judul.strip():
            st.warning("Please enter a music title first.")
        else:
            judul_vector = tfidf_title.transform([judul])
            similarities = cosine_similarity(judul_vector, title_tfidf).flatten()
            top_index = similarities.argsort()[::-1][0]
            lagu = df_clean.iloc[[top_index]]

            fitur = lagu.iloc[0]
            genre = fitur['genre']
            subgenre = fitur['subgenre']
            tempo = fitur['tempo']
            duration_ms = fitur['duration_ms']
            energy = fitur['energy']
            danceability = fitur['danceability']
            artist = fitur['artist']
            album = fitur['album']
            lyrics = fitur['lyrics'] if pd.notna(fitur['lyrics']) else ''
            judul_terdekat = fitur['judul_musik']

            X_input = np.hstack([
                tfidf_genre.transform([genre]).toarray(),
                tfidf_subgenre.transform([subgenre]).toarray(),
                tfidf_title.transform([judul_terdekat]).toarray(),
                tfidf_artist.transform([artist]).toarray(),
                tfidf_lyrics.transform([lyrics]).toarray(),
                tfidf_album.transform([album]).toarray(),
                scaler.transform([[tempo, duration_ms, energy, danceability]])
            ])

            pred = model.predict(X_input)[0]
            kategori = label_enc.inverse_transform([pred])[0]

            st.success(f"The input **'{judul}'** is most similar to **'{judul_terdekat}'** by **{artist}**.")
            st.info(f"The genre of this song is **{genre}**.")
            st.success(f"This music is predicted to have popularity level: **{kategori}**.")

            df_rekom_genre = df_clean[df_clean['genre'].str.lower() == genre.lower()].sort_values(by='popularity', ascending=False).head(5)
            st.subheader("🎧 Recommendations Based on Similar Genre")
            for _, row in df_rekom_genre.iterrows():
                music_card(row['judul_musik'], row['artist'], row['popularity'])
                st.caption(f"Genre: {row['genre']}")

            top_indices = similarities.argsort()[::-1][1:6]
            df_rekom_judul = df_clean.iloc[top_indices]
            st.subheader("\U0001F3A7 Recommendations Based on Title Similarity")
            for _, row in df_rekom_judul.sort_values(by='popularity', ascending=False).iterrows():
                music_card(row['judul_musik'], row['artist'], row['popularity'])

            if lyrics.strip():
                lyric_vector = tfidf_lyrics.transform([lyrics])
                lyric_similarities = cosine_similarity(lyric_vector, tfidf_lyrics.transform(df_clean['lyrics'].fillna(''))).flatten()
                top_lyric_indices = lyric_similarities.argsort()[::-1][1:6]
                df_rekom_lyrics = df_clean.iloc[top_lyric_indices]

                st.subheader("\U0001F3A7 Recommendations Based on Lyric Similarity")
                for _, row in df_rekom_lyrics.sort_values(by='popularity', ascending=False).iterrows():
                    music_card(row['judul_musik'], row['artist'], row['popularity'])
            else:
                st.subheader("\U0001F3A7 Recommendations Based on Lyric Similarity")
                st.info("This song has no lyrics to compare.")

            df_rekomendasi = pd.concat([df_rekom_genre, df_rekom_judul, df_rekom_lyrics]).drop_duplicates(subset='judul_musik')
            st.session_state.recommendation_table = df_rekomendasi

            st.session_state.history.append({
                'Judul': judul,
                'Artis': artist,
                'Genre': genre,
                'Subgenre': subgenre,
                'Prediksi': kategori,
                'Rekomendasi': ', '.join(df_rekomendasi['judul_musik'].head(3).tolist())
            })
