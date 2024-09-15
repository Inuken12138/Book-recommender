import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load precomputed book embeddings
book_embeddings = np.load('book_embeddings.npy', allow_pickle=True).item()

# Load precomputed recommendations
batch_recommendations = pd.read_csv('recommendations_batch_combined.csv')

final_df = pd.read_csv("final_df_parallel.csv")




def classify_all_books(user_id, data, book_embeddings):
    user_preferences = data[data["User-ID"] == user_id]
    liked_books = user_preferences[user_preferences["Book-Rating"] > 5]["ISBN"].tolist()
    disliked_books = user_preferences[user_preferences["Book-Rating"] <= 5]["ISBN"].tolist()

    liked_embeddings = [book_embeddings[isbn]['embedding'] for isbn in liked_books if isbn in book_embeddings]
    disliked_embeddings = [book_embeddings[isbn]['embedding'] for isbn in disliked_books if isbn in book_embeddings]

    liked_vector = np.mean(liked_embeddings, axis=0).reshape(1, -1) if liked_embeddings else None
    disliked_vector = np.mean(disliked_embeddings, axis=0).reshape(1, -1) if disliked_embeddings else None

    classified_books = {
        'liked_books': [],
        'disliked_books': []
    }

    for isbn, book_data in book_embeddings.items():
        book_embedding = book_data['embedding'].reshape(1, -1)
        if liked_vector is not None:
            liked_similarity = cosine_similarity(liked_vector, book_embedding)[0][0]
            
        else:
            liked_similarity = -1
        
        if disliked_vector is not None:
            disliked_similarity = cosine_similarity(disliked_vector, book_embedding)[0][0]
        else:
            disliked_similarity = -1
        
        if liked_similarity > disliked_similarity:
            classified_books['liked_books'].append(isbn)
        else:
            classified_books['disliked_books'].append(isbn)

    return classified_books

def calculate_tp_fp(batch_recommendations, final_df, book_embeddings):
    
    # Check if the metrics file already exists
    metrics_file = 'relevant_items.csv'
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
    else:
        metrics_df = pd.DataFrame(columns=['User-ID', 'Relevant-ISBNs'])


    for user_id in batch_recommendations['User-ID'].unique():
        # Skip user if already processed
        if user_id in metrics_df['User-ID'].values:
            continue



        classified_all_books = classify_all_books(user_id, final_df, book_embeddings)
        if classified_all_books is None:
            print("error: classified_all_books is none")
            continue
        
        total_relevant = set(classified_all_books['liked_books'])
        

        # Save metrics for the user
        relevant_item = pd.DataFrame([{
            'User-ID': user_id,
            'Relevant-ISBNs': ",".join(total_relevant)  # Join ISBNs into a single string
        }])

        # Append to the DataFrame
        metrics_df = pd.concat([metrics_df, relevant_item], ignore_index=True)

        # Save the updated DataFrame to CSV after each user is processed
        metrics_df.to_csv(metrics_file, index=False)
        

calculate_tp_fp(batch_recommendations, final_df, book_embeddings)
