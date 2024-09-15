import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the original datasets
books = pd.read_csv('books_subset.csv')
users = pd.read_csv('users_subset.csv')
ratings = pd.read_csv('ratings_subset.csv')

ratings_with_titles = ratings.merge(books[['ISBN', 'Book-Title']], on='ISBN')

""" duplicates = ratings_with_titles[ratings_with_titles.duplicated(subset=['User-ID', 'Book-Title'], keep=False)]

# Print all duplicate entries
print("Duplicate entries:")
print(duplicates) """

""" # Create a pivot table for the ratings
ratings_pivot = ratings_with_titles.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# Save the pivot table to a CSV file
ratings_pivot.to_csv('user_book_matrix.csv')

print("User-book matrix has been saved to 'user_book_matrix.csv'.")
 """

# Calculate total number of ratings per user
user_ratings_count = ratings_with_titles.groupby('User-ID')['Book-Rating'].count().reset_index()
user_ratings_count.columns = ['User-ID', 'Total Ratings']

# Calculate average rating per user
user_ratings_avg = ratings_with_titles.groupby('User-ID')['Book-Rating'].mean().reset_index()
user_ratings_avg.columns = ['User-ID', 'Average Rating']

# Calculate number of unique books rated by each user
user_unique_books = ratings_with_titles.groupby('User-ID')['ISBN'].nunique().reset_index()
user_unique_books.columns = ['User-ID', 'Unique Books Rated']

# Merge the data into a single DataFrame
user_analytics = pd.merge(user_ratings_count, user_ratings_avg, on='User-ID')
user_analytics = pd.merge(user_analytics, user_unique_books, on='User-ID')

user_analytics.to_csv('user_analytics.csv', index=False)
