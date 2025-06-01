import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

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

# Train model once
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

    return model, tfidf_genre, tfidf_subgenre, scaler

model, tfidf_genre, tfidf_subgenre, scaler = train_model(df_clean)

# Session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Navigasi
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih Halaman", ["Beranda", "Distribusi Musik", "Rekomendasi Musik"])

# Halaman Beranda
if halaman == "Beranda":
    st.title("🎵 Beranda Musik Populer")
    st.subheader("10 Musik Terpopuler")
    top10 = df.sort_values(by='popularity', ascending=False).head(10)[['judul_musik', 'artist', 'popularity']]
    st.dataframe(top10)

    st.subheader("Riwayat Pencarian Rekomendasi")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
    else:
        st.info("Belum ada pencarian.")

# Halaman Distribusi Musik
elif halaman == "Distribusi Musik":
    st.title("📊 Distribusi Musik")

    st.subheader("10 Artis Terpopuler")
    top_artists = df['artist'].value_counts().head(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_artists.values, y=top_artists.index, ax=ax1)
    st.pyplot(fig1)

    st.subheader("10 Genre Terpopuler")
    top_genres = df['genre'].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax2)
    st.pyplot(fig2)

# Halaman Rekomendasi Musik
elif halaman == "Rekomendasi Musik":
    st.title("🔍 Rekomendasi Musik Berdasarkan Judul")

    judul = st.text_input("Masukkan Judul Musik")

    if st.button("Rekomendasikan"):
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

            # Tambahkan ke riwayat
            st.session_state.history.append({
                'Judul': judul,
                'Artis': artist,
                'Genre': genre,
                'Subgenre': subgenre,
                'Prediksi': kategori
            })

            # Rekomendasi serupa berdasarkan genre dan urut popularitas
            df_rekomendasi = df_clean[df_clean['genre'].str.lower() == genre.lower()]
            df_rekomendasi = df_rekomendasi.sort_values(by='popularity', ascending=False).head(5)

            st.subheader("🎧 Musik Serupa Berdasarkan Genre")
            if not df_rekomendasi.empty:
                st.dataframe(df_rekomendasi[['judul_musik', 'artist', 'genre', 'popularity']])
            else:
                st.info("Tidak ditemukan musik serupa untuk genre tersebut.")
