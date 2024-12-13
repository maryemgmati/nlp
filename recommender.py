import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger et préparer les données
df = pd.read_csv("imdb_movies_cleaned.csv")
df = df.dropna(subset=["Overview", "Series_Title"])
df['Overview'] = df['Overview'].str.lower()

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Overview'])

# Fonction pour recommander des films basés sur un résumé
def recommend_by_summary(user_summary, num_recommendations=5):
    # Vectoriser le résumé de l'utilisateur
    user_vector = vectorizer.transform([user_summary.lower()])

    # Calculer la similarité cosinus entre l'entrée utilisateur et les résumés des films
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Trouver les indices des films les plus similaires
    similar_indices = cosine_sim.argsort()[-num_recommendations:][::-1]

    # Obtenir les titres des films correspondants
    similar_movies = df.iloc[similar_indices][['Series_Title', 'Overview']]

    return similar_movies
