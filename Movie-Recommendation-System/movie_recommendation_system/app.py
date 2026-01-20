import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "tmdb_5000_movies.csv")

df = pd.read_csv(csv_path)
df = df[['title', 'overview']]
df['overview'] = df['overview'].fillna('')

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(movie_title, num_recommendations=5):
    idx = df[df['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_movies = similarity_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in top_movies]
    return df['title'].iloc[movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_name = st.selectbox("Select a movie:", df['title'].values)

if st.button("Recommend"):
    recommendations = recommend_movies(movie_name)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)
