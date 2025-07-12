import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies_data = pd.read_csv('movies.csv')

# Prepare data
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Vectorize
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Similarity matrix
similarity = cosine_similarity(feature_vectors)

# Function to get recommendations
def recommend_movies(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    finding_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if not finding_close_match:
        return []
    close_match = finding_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index.values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies):
        index = movie[0]
        title = movies_data.iloc[index]['title']
        if i < 30:
            recommended_movies.append(title)
    return recommended_movies



# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter your favorite movie:")

if st.button("Recommend"):
    if movie_name:
        recommendations = recommend_movies(movie_name)
        if recommendations:
            st.subheader("Recommended Movies:")
            for i, movie in enumerate(recommendations[1:30], start=1):
                st.write(f"{i}. {movie}")
        else:
            st.warning("No close match found! Try a different movie name.")
    else:
        st.error("Please enter a movie name to get recommendations.")

