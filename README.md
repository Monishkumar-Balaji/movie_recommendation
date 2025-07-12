# Movie Recommendation System using Content-Based Filtering (Streamlit Web App)
<img width="1919" height="1117" alt="sample image" src="https://github.com/user-attachments/assets/12845ceb-c783-4f69-89ca-c52f87fc9590" />


This project is a content-based movie recommendation system that suggests movies similar to a user’s favorite film based on their content attributes. It uses TF-IDF vectorization to convert metadata like genres, keywords, tagline, cast, and director into numerical feature vectors, and computes cosine similarity to find closely related movies.

The application is built with Python and features a simple, interactive Streamlit web interface. Users can enter a movie name, and the system recommends the top 29 most similar titles from the dataset in real time.

The backend loads a movie dataset, combines selected content features, and transforms them using Scikit-learn’s TfidfVectorizer. The system then calculates pairwise similarity scores between movies. When a user inputs a title, the app finds the closest match in the dataset and displays a ranked list of similar movies.

This project demonstrates how to build an effective recommendation system using content-based filtering techniques and deploy it as a lightweight, user-friendly web application.

