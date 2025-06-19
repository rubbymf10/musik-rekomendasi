import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Styling dark mode ---
st.markdown("""
    <style>
    .main, .block-container {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .css-1d391kg {background-color: #181818;}
    .css-1d391kg .css-14xtw13 {color: #b3b3b3;}
    h1, h2, h3, h4 {
        color: #1DB954;
    }
    .dataframe tbody tr th, .dataframe tbody tr td {
        border-color: #333333;
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
    ::-webkit-scrollbar {
      width: 8px;
    }
    ::-webkit-scrollbar-thumb {
      background-color: #1DB954;
      border-radius: 4px;
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

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('musik.csv')
    df_clean = df.dropna(subset=['popularity', 'genre', 'subgenre', 'tempo', 'duration_ms', 'energy', 'danceability'])
    low_thresh = df_clean['popularity'].quantile(0.33)
    high_thresh = df_clean['popularity'].quantile(0.66)

    def categorize_popularity(pop):
        if pop <= low_thresh:
            return 'Rendah'
        elif pop > high_thresh:
            return 'Tinggi'
        else:
            return np.nan

    df_clean['pop_category'] = df_clean['popularity'].apply(categorize_popularity)
    df_clean = df_clean.dropna(subset=['pop_category'])
    label_enc = LabelEncoder()
    df_clean['pop_encoded'] = label_enc.fit_transform(df_clean['pop_category'])
    return df, df_clean, label_enc

df, df_clean, label_enc = load_data()

@st.cache_resource
def train_model(df_clean):
    tfidf_genre = TfidfVectorizer()
    tfidf_subgenre = TfidfVectorizer()
    genre_tfidf = tfidf_genre.fit_transform(df_clean['genre'])
    subgenre_tfidf = tfidf_subgenre.fit_transform(df_clean['subgenre'])

    df_genre_tfidf = pd.DataFrame(genre_tfidf.toarray(), columns=tfidf_genre.get_feature_names_out(), index=df_clean.index)
    df_subgenre_tfidf = pd.DataFrame(subgenre_tfidf.toarray(), columns=tfidf_subgenre.get_feature_names_out(), index=df_clean.index)

    features_num = ['tempo', 'duration_ms', 'energy', 'danceability']
    scaler = MinMaxScaler()
    df_num_scaled = pd.DataFrame(scaler.fit_transform(df_clean[features_num]), columns=features_num, index=df_clean.index)

    X = pd.concat([df_genre_tfidf, df_subgenre_tfidf, df_num_scaled], axis=1)
    y = df_clean['pop_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, tfidf_genre, tfidf_subgenre, scaler, X_test, y_test, y_pred

model, tfidf_genre, tfidf_subgenre, scaler, X_test, y_test, y_pred = train_model(df_clean)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'recommendation_table' not in st.session_state:
    st.session_state.recommendation_table = pd.DataFrame()

with st.sidebar:
    st.markdown('<h2 style="color:#1DB954; margin-bottom: 15px;">🎵 Dashboard</h2>', unsafe_allow_html=True)
    halaman = st.radio("", ["Beranda", "Distribusi Musik", "Rekomendasi Musik"], index=0, key="page_select")

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

if halaman == "Beranda":
    st.header("Top 10 Musik Terpopuler")
    top10 = df.sort_values(by='popularity', ascending=False).head(10)
    for _, row in top10.iterrows():
        music_card(row['judul_musik'], row['artist'], row['popularity'])

    st.markdown("---")
    st.header("5 Musik Terpopuler dari Setiap Genre")
    genre_list = df['genre'].dropna().unique()
    for genre in genre_list:
        st.subheader(f"🎶 Genre: {genre}")
        top5_by_genre = df[df['genre'] == genre].sort_values(by='popularity', ascending=False).head(5)
        for _, row in top5_by_genre.iterrows():
            music_card(row['judul_musik'], row['artist'], row['popularity'])

    st.markdown("---")
    st.header("Riwayat Pencarian Rekomendasi")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(f"- **{h['Judul']}** oleh {h['Artis']} (Genre: {h['Genre']}, Prediksi: {h['Prediksi']})")
    else:
        st.info("Belum ada pencarian.")

    st.markdown("---")
    st.header("🎧 Rekomendasi Genre Terakhir")
    if not st.session_state.recommendation_table.empty:
        df_show = st.session_state.recommendation_table.sort_values(by='popularity', ascending=False)
        for _, row in df_show.iterrows():
            music_card(row['judul_musik'], row['artist'], row['popularity'])
    else:
        st.info("Belum ada rekomendasi genre ditampilkan.")

    if st.button("Reset Riwayat Pencarian"):
        st.session_state.history = []
        st.session_state.recommendation_table = pd.DataFrame()
        st.experimental_rerun()

elif halaman == "Distribusi Musik":
    st.header("Distribusi Musik")
    st.subheader("10 Artis Terpopuler")
    top_artists = df['artist'].value_counts().head(10)
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=top_artists.values, y=top_artists.index, ax=ax1, palette="Greens_d")
    ax1.set_xlabel('Jumlah Lagu')
    ax1.set_ylabel('Artis')
    st.pyplot(fig1)

    st.subheader("10 Genre Terpopuler")
    top_genres = df['genre'].value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax2, palette="Greens_d")
    ax2.set_xlabel('Jumlah Lagu')
    ax2.set_ylabel('Genre')
    st.pyplot(fig2)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    # Uncomment below to show Confusion Matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_enc.classes_)
    # fig_cm, ax_cm = plt.subplots(figsize=(6,6))
    # disp.plot(ax=ax_cm, cmap=plt.cm.Greens, colorbar=False)
    # st.pyplot(fig_cm)

elif halaman == "Rekomendasi Musik":
    st.header("Rekomendasi Musik Berdasarkan Judul")
    judul = st.text_input("Masukkan Judul Musik")

    if st.button("Rekomendasikan"):
        if not judul.strip():
            st.warning("Silakan masukkan judul musik terlebih dahulu.")
        else:
            lagu = df_clean[df_clean['judul_musik'].str.lower() == judul.lower()]
            if lagu.empty:
                st.warning("Judul tidak ditemukan dalam dataset.")
            else:
                fitur = lagu.iloc[0]
                genre = fitur['genre']
                subgenre = fitur['subgenre']
                tempo = fitur['tempo']
                duration_ms = fitur['duration_ms']
                energy = fitur['energy']
                danceability = fitur['danceability']
                artist = fitur['artist']

                genre_tfidf = tfidf_genre.transform([genre])
                subgenre_tfidf = tfidf_subgenre.transform([subgenre])
                features_num = scaler.transform([[tempo, duration_ms, energy, danceability]])

                X_input = np.hstack([
                    genre_tfidf.toarray(),
                    subgenre_tfidf.toarray(),
                    features_num
                ])

                pred = model.predict(X_input)[0]
                kategori = label_enc.inverse_transform([pred])[0]

                st.success(f"Musik '{judul}' oleh {artist} diprediksi memiliki popularitas: **{kategori}**")

                df_rekomendasi = df_clean[df_clean['genre'].str.lower() == genre.lower()]
                df_rekomendasi = df_rekomendasi.sort_values(by='popularity', ascending=False).head(5)

                st.session_state.history.append({
                    'Judul': judul,
                    'Artis': artist,
                    'Genre': genre,
                    'Subgenre': subgenre,
                    'Prediksi': kategori,
                    'Rekomendasi': ', '.join(df_rekomendasi['judul_musik'].head(3).tolist())
                })

                st.session_state.recommendation_table = df_rekomendasi

                st.subheader("🎧 Musik Serupa Berdasarkan Genre")
                for _, row in df_rekomendasi.iterrows():
                    music_card(row['judul_musik'], row['artist'], row['popularity'])
