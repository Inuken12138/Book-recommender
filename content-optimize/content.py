import pandas as pd
import warnings
import numpy as np
import re

warnings.filterwarnings("ignore")
books_df = pd.read_csv("combined_books.csv")
ratings_df = pd.read_csv("content_Ratings.csv")
users_df = pd.read_csv("content_Users.csv")

# 填充 Description 和 openlibrary_Description 列
books_df["Description"] = books_df["Description"].fillna(
    books_df["openlibrary_Description"]
)
books_df["openlibrary_Description"] = books_df["openlibrary_Description"].fillna(
    books_df["Description"]
)

# 删除同时 Description 和 openlibrary_Description 都为空的行
books_df = books_df.dropna(subset=["Description", "openlibrary_Description"], how="all")

# 填充 Description 和 openlibrary_Description 列
books_df["Categories"] = books_df["Categories"].fillna(
    books_df["openlibrary_Categories"]
)
books_df["openlibrary_Categories"] = books_df["openlibrary_Categories"].fillna(
    books_df["Categories"]
)

# 删除同时 Description 和 openlibrary_Description 都为空的行
books_df = books_df.dropna(subset=["Categories", "openlibrary_Categories"], how="all")

merged_df = pd.merge(books_df, ratings_df, on="ISBN")
final_df = pd.merge(merged_df, users_df, on="User-ID")

final_df["Book-Author"].fillna(final_df["Book-Author"].mode()[0], inplace=True)
final_df["Publisher"].fillna(final_df["Publisher"].mode()[0], inplace=True)

# Handle missing values in 'Age' column with median
final_df["Age"].fillna(final_df["Age"].median(), inplace=True)

# Converts the data types of the three fields in the final_df data box to integers
final_df["Age"] = final_df["Age"].astype(int)
final_df["Book-Rating"] = final_df["Book-Rating"].astype(int)
final_df["User-ID"] = final_df["User-ID"].astype(int)

# Remove the data whose rating is 0
zero_rating_books_df = final_df[final_df["Book-Rating"] == 0]

final_df = final_df[final_df["Book-Rating"] != 0]

# Handle outliers in the 'Year-Of-Publication' column
# 将出版年份异常值处理为NaN并使用平均值填充
import numpy as np

final_df.loc[
    (final_df["Year-Of-Publication"] > 2022) | (final_df["Year-Of-Publication"] == 0),
    "Year-Of-Publication",
] = np.nan
final_df.loc[:, "Year-Of-Publication"] = (
    final_df["Year-Of-Publication"]
    .fillna(round(books_df["Year-Of-Publication"].mean()))
    .astype(np.int32)
)

final_df.loc[(final_df.Age > 90) | (final_df.Age < 5), "Age"] = np.nan

# replacing NaNs with mean
final_df.Age = final_df.Age.fillna(users_df.Age.mean())

# setting the data type as int
final_df.Age = final_df.Age.astype(np.int32)
final_df["Year-Of-Publication"] = final_df["Year-Of-Publication"].astype(np.int32)

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
from sklearn.metrics import precision_score, recall_score

# Train a Word2Vec model
word2vec_model_recommender = Word2Vec(
    sentences=corpus, vector_size=500, window=5, min_count=5, sg=2
)


def recommend(user_id, data, word2vec_model):
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
    for idx, row in data.iterrows():
        # Skip books that the user has already rated
        if row["ISBN"] in liked_ISBN:
            continue

        row_text = " ".join(
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
        row_tokens = row_text.split()
        row_vectors = [
            word2vec_model.wv[token]
            for token in row_tokens
            if token in word2vec_model.wv
        ]
        if len(row_vectors) > 0:
            row_avg_vector = sum(row_vectors) / len(row_vectors)
            similarity = cosine_similarity([avg_vector], [row_avg_vector])[0][0]
            similarities.append((row, similarity))

    # Rank the similarity and select the top 10 recommendations
    similarities.sort(key=lambda x: x[1], reverse=True)
    recommendations = []
    for book, sim in similarities:
        if book["Book-Title"] not in recommended_titles and book["Book-Rating"] > 5:
            recommendations.append(book.to_dict())
            recommended_titles.add(book["Book-Title"])
        if len(recommendations) >= 10:
            break

    # Convert the recommendation result to a DataFrame
    recommendations_df = pd.DataFrame(recommendations)
    return recommendations_df

def evaluate_recommendations(data, word2vec_model, top_n=10):
    user_ids = data['User-ID'].unique()
    
    all_recall = []
    all_precision = []
    hit_rate = []
    
    for user_id in user_ids:
        recommendations = recommend(user_id, data, word2vec_model)
        if recommendations is None or recommendations.empty:
            continue
        
        user_preferences = data[data['User-ID'] == user_id]
        liked_books = user_preferences['Book-Title'].tolist()
        
        recommended_books = recommendations['Book-Title'].tolist()
        true_positives = [1 if book in liked_books else 0 for book in recommended_books[:top_n]]
        
        # Calculate precision and recall
        precision = sum(true_positives) / top_n
        recall = sum(true_positives) / len(liked_books)
        
        all_precision.append(precision)
        all_recall.append(recall)
        
        # Calculate hit rate (whether there is at least one correct recommendation)
        if sum(true_positives) > 0:
            hit_rate.append(1)
        else:
            hit_rate.append(0)
    
    # Calculate the average recall, precision, and hit rate across all users
    average_recall = np.mean(all_recall)
    average_precision = np.mean(all_precision)
    hit_rate = np.mean(hit_rate)
    
    return average_recall, average_precision, hit_rate

# Example of using the evaluation function
average_recall, average_precision, hit_rate = evaluate_recommendations(final_df, word2vec_model_recommender)
print(f'Average Recall: {average_recall:.2f}')
print(f'Average Precision: {average_precision:.2f}')
print(f'Hit Rate: {hit_rate:.2f}')