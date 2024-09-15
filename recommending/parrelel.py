import pandas as pd
import warnings
import numpy as np
import re
import os


warnings.filterwarnings("ignore")
final_df = pd.read_csv("final_df_parallel.csv")
books_df = pd.read_csv("books_df_parallel.csv")
filepath = "user_ids_part_1.csv"
users_df = pd.read_csv(filepath)

corpus = (
    (
        books_df["Book-Title"].astype(str)
        + " "
        + books_df["Book-Author"].astype(str)
        + " "
        + books_df["Description"].astype(str)
        + " "
        + books_df["Categories"].astype(str)
        + " "
        + books_df["openlibrary_Description"].astype(str)
        + " "
        + books_df["openlibrary_Categories"].astype(str)
    )
    .apply(str.split)
    .tolist()
)

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Train a Word2Vec model
word2vec_model_recommender = Word2Vec(
    sentences=corpus, vector_size=500, window=5, min_count=5, sg=2
)

def precompute_book_embeddings(books_df, word2vec_model):
    book_embeddings = {}
    for idx, row in books_df.iterrows():
        text = " ".join(
            [
                str(row[col])
                for col in [
                    "Book-Title",
                    "Book-Author",
                    "Categories",
                    "Description",
                    "openlibrary_Description",
                    "openlibrary_Categories",
                ]
            ]
        )
        tokens = text.split()
        vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
        if len(vectors) > 0:
            embedding = np.mean(vectors, axis=0)
            book_embeddings[row["ISBN"]] = {'embedding': embedding, 'row_data': row.to_dict()}
    return book_embeddings

book_embeddings = precompute_book_embeddings(books_df, word2vec_model_recommender)
np.save('book_embeddings.npy', book_embeddings)

# Load precomputed book embeddings
book_embeddings = np.load('book_embeddings.npy', allow_pickle=True).item()

def recommend(user_id, data, word2vec_model, book_embeddings):
    # Get user preferences
    user_preferences = data[data["User-ID"] == user_id]
    if user_preferences.empty:
        return None

    # Get information about the user's favorite books
    liked_books = user_preferences["Book-Title"].tolist()
    liked_ISBN = user_preferences["ISBN"].tolist()
    liked_authors = user_preferences["Book-Author"].tolist()
    liked_genres = user_preferences["Categories"].tolist()
    liked_description = user_preferences["Description"].tolist()
    liked_openlibrary_description = user_preferences["openlibrary_Description"].tolist()
    liked_openlibrary_categories = user_preferences["openlibrary_Categories"].tolist()

    # Merge user preference information
    text = " ".join(
        liked_ISBN
        + liked_books
        + liked_authors
        + liked_genres
        + liked_description
        + liked_openlibrary_description
        + liked_openlibrary_categories
    )
    tokens = text.split()

    # Get text vector
    vectors = [
        word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv
    ]
    if len(vectors) == 0:
        return None
    avg_vector = sum(vectors) / len(vectors)

    # Calculate the similarity to each book
    similarities = []
    recommended_titles = set()
    for isbn, book_vector in book_embeddings.items():
        if isbn not in liked_ISBN:
            similarity = cosine_similarity([avg_vector], [book_vector['embedding']])[0][0]
            similarities.append((book_vector['row_data'], similarity))


    # Rank the similarity and select the top 10 recommendations
    similarities.sort(key=lambda x: x[1], reverse=True)
    recommendations = []
    for book, sim in similarities:
        recommendations.append(book)
        recommended_titles.add(book["Book-Title"])
        if len(recommendations) >= 10:
            break

    # Convert the recommendation result to a DataFrame
    recommendations_df = pd.DataFrame(recommendations)
    return recommendations_df


# Example of using the function
#user_id = 279858
#recommendations = recommend(user_id, final_df, word2vec_model_recommender)

from sklearn.metrics import precision_score, recall_score

# Evaluate recommendations and save to CSV in batches of 10 users
def evaluate_recommendations(data, word2vec_model, users_df, book_embeddings, top_n=10, batch_size=1):

    user_ids = users_df[users_df['Processed'] == False]['User-ID'].unique()
    #user_ids = [279858]

    all_recommendations = []
    
    # Load existing batch recommendations if any
    if os.path.exists('recommendations_batch_combined.csv'):
        batch_recommendations = pd.read_csv('recommendations_batch_combined.csv')
    else:
        batch_recommendations = pd.DataFrame()

    for idx, user_id in enumerate(user_ids):
        recommendations = recommend(user_id, data, word2vec_model, book_embeddings)
        
        if recommendations is not None:
            recommendations['User-ID'] = user_id
            all_recommendations.append(recommendations)
        
        # Mark the user as processed
        users_df.loc[users_df['User-ID'] == user_id, 'Processed'] = True

        # Save recommendations to a CSV file for every batch_size users
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(user_ids):
            batch_number = (idx + 1) // batch_size
            batch_recommendations = pd.concat([batch_recommendations] + all_recommendations, ignore_index=True)
            batch_recommendations.to_csv('recommendations_batch_combined.csv', index=False)
            all_recommendations = []

            # Save the processed users to CSV
            users_df.to_csv(filepath, index=False)

# Example usage of the evaluation function
evaluate_recommendations(final_df, word2vec_model_recommender, users_df, book_embeddings)
