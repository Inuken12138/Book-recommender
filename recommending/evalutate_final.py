import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load precomputed book embeddings
book_embeddings = np.load('book_embeddings.npy', allow_pickle=True).item()

# Load precomputed recommendations
batch_recommendations = pd.read_csv('recommendations_batch_combined.csv')

relevant_item = pd.read_csv('relevant_items_test.csv')

final_df = pd.read_csv("final_df_parallel.csv")


def classify_recommendations(user_id, data, batch_recommendations, book_embeddings):
    # Get the recommendations for the user
    user_recommendations = batch_recommendations[batch_recommendations['User-ID'] == user_id]
    if user_recommendations.empty:
        return None
    
    recommended_books = user_recommendations['ISBN'].tolist()
    
    user_preferences = data[data["User-ID"] == user_id]
    liked_books = user_preferences[user_preferences["Book-Rating"] > 5]["ISBN"].tolist()
    disliked_books = user_preferences[user_preferences["Book-Rating"] <= 5]["ISBN"].tolist()

    liked_embeddings = [book_embeddings[isbn]['embedding'] for isbn in liked_books if isbn in book_embeddings]
    disliked_embeddings = [book_embeddings[isbn]['embedding'] for isbn in disliked_books if isbn in book_embeddings]

    liked_vector = np.mean(liked_embeddings, axis=0).reshape(1, -1) if liked_embeddings else None
    disliked_vector = np.mean(disliked_embeddings, axis=0).reshape(1, -1) if disliked_embeddings else None

    classified_recommendations = {
        'liked_recommendations': [],
        'disliked_recommendations': []
    }

    for isbn in recommended_books:
        if isbn in book_embeddings:
            book_embedding = book_embeddings[isbn]['embedding'].reshape(1, -1)
            if liked_vector is not None:
                liked_similarity = cosine_similarity(liked_vector, book_embedding)[0][0]
            else:
                liked_similarity = -1
            
            if disliked_vector is not None:
                disliked_similarity = cosine_similarity(disliked_vector, book_embedding)[0][0]

            else:
                disliked_similarity = -1
            
            if liked_similarity > disliked_similarity:
                classified_recommendations['liked_recommendations'].append(isbn)
            else:
                classified_recommendations['disliked_recommendations'].append(isbn)

    return classified_recommendations


def calculate_tp_fp(batch_recommendations, final_df, book_embeddings):
    
    total_books = len(book_embeddings)

    # Check if the metrics file already exists
    metrics_file = 'user_metrics_test.csv'
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
    else:
        metrics_df = pd.DataFrame(columns=['User-ID', 'Precision', 'Hit rate', 'Recall', 'Accuracy', 'F1'])


    for user_id in batch_recommendations['User-ID'].unique():
        # Skip user if already processed
        if user_id in metrics_df['User-ID'].values:
            continue

        total_recommended = 10
        #total_relevant = len(final_df[(final_df['User-ID'] == user_id) & (final_df['Book-Rating'] > 5)])

        classified_recommendations = classify_recommendations(user_id, final_df, batch_recommendations, book_embeddings)
        if classified_recommendations is None:
            print("error: classified_recommendations is none")
            continue

        tp = set(classified_recommendations['liked_recommendations'])
        fp = set(classified_recommendations['disliked_recommendations'])



       
        # Retrieve relevant books for the user from the relevant_items.csv file
        relevant_books_row = relevant_item[relevant_item['User-ID'] == user_id]
        relevant_isbns_str = relevant_books_row['Relevant-ISBNs'].values[0]
        
        if pd.isna(relevant_isbns_str):  # Check if the string is empty
            total_relevant = set()
        else:
            total_relevant = set(relevant_isbns_str.split(','))

        
        fn = total_relevant - tp
        tn = total_books - (len(tp) + len(fp) + len(fn))

        # precision@N and recall@N
        precision = len(tp) / total_recommended
        recall = len(tp) / len(total_relevant) if len(total_relevant) > 0 else 0
        accuracy = (len(tp) + tn)/total_books
        f1 = (2*precision*recall)/(precision + recall) if (precision + recall) > 0 else 0

        # Calculate hit rate (whether there is at least one correct recommendation)
        if len(tp) > 0:
            hit_rate = 1
        else:
            hit_rate = 0

        # Save metrics for the user
        user_metrics = pd.DataFrame([{
            'User-ID': user_id,
            'Precision': precision,
            'Hit rate': hit_rate,
            'Recall': recall,
            'Accuracy': accuracy,
            'F1': f1
        }])

        # Append to the DataFrame
        metrics_df = pd.concat([metrics_df, user_metrics], ignore_index=True)

        # Save the updated DataFrame to CSV after each user is processed
        metrics_df.to_csv(metrics_file, index=False)
        

calculate_tp_fp(batch_recommendations, final_df, book_embeddings)
