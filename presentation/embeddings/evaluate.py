import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load the data
ratings = pd.read_csv('./Ratings.csv')
users = pd.read_csv('./Users.csv')
dtype_spec = {
    'ISBN': str,
    'Book-Title': str,
    'Book-Author': str,
    'Year-Of-Publication': str,
    'Publisher': str,
    'Image-URL-S': str,
    'Image-URL-M': str,
    'Image-URL-L': str,
    'Description': str,
    'Categories': str
}
books = pd.read_csv('./updated_books_progress.csv', dtype=dtype_spec, low_memory=False)

# Preprocess the data
def preprocess_users(users):
    users['Age'].fillna(users['Age'].median(), inplace=True)
    users['Age'] = users['Age'].astype(int)
    users['Location'] = users['Location'].astype('category').cat.codes
    return users

def preprocess_books(books):
    books.fillna('', inplace=True)
    books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
    books['Year-Of-Publication'].fillna(books['Year-Of-Publication'].median(), inplace=True)
    books['Publisher'] = books['Publisher'].astype('category').cat.codes
    books['Book-Author'] = books['Book-Author'].astype('category').cat.codes
    return books

users = preprocess_users(users)
books = preprocess_books(books)

user_id_mapping = {id: idx for idx, id in enumerate(ratings['User-ID'].unique())}
book_id_mapping = {id: idx for idx, id in enumerate(ratings['ISBN'].unique())}

ratings['User-ID'] = ratings['User-ID'].map(user_id_mapping)
ratings['ISBN'] = ratings['ISBN'].astype(str)
books['ISBN'] = books['ISBN'].astype(str)
ratings['ISBN'] = ratings['ISBN'].map(book_id_mapping)
books['ISBN'] = books['ISBN'].map(book_id_mapping)
ratings.dropna(subset=['Book-Rating'], inplace=True)

# Split data into training and test sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
users[['Age']] = scaler.fit_transform(users[['Age']])
books[['Year-Of-Publication']] = scaler.fit_transform(books[['Year-Of-Publication']])

# Define the EmbeddingNet model
class EmbeddingNet(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim, num_locations, num_authors, num_publishers):
        super(EmbeddingNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        self.location_embedding = nn.Embedding(num_locations, embedding_dim)
        self.author_embedding = nn.Embedding(num_authors, embedding_dim)
        self.publisher_embedding = nn.Embedding(num_publishers, embedding_dim)
        self.user_age = nn.Linear(1, embedding_dim)
        self.book_year = nn.Linear(1, embedding_dim)
    
    def forward(self, user_id, book_id, location_id, age, author_id, year, publisher_id):
        user_embed = self.user_embedding(user_id).squeeze()
        book_embed = self.book_embedding(book_id).squeeze()
        location_embed = self.location_embedding(location_id).squeeze()
        author_embed = self.author_embedding(author_id).squeeze()
        publisher_embed = self.publisher_embedding(publisher_id).squeeze()
        age_embed = self.user_age(age).squeeze()
        year_embed = self.book_year(year).squeeze()
        return torch.cat([user_embed, book_embed, location_embed, age_embed, author_embed, year_embed, publisher_embed], dim=-1)

# Preprocess test data
test_data = test_data.merge(users[['User-ID', 'Age', 'Location']], on='User-ID')
test_data = test_data.merge(books[['ISBN', 'Book-Author', 'Year-Of-Publication', 'Publisher']], on='ISBN')

test_data['Age'] = test_data['Age'].apply(lambda x: torch.tensor(x).float().unsqueeze(0))
test_data['Book-Author'] = test_data['Book-Author'].apply(lambda x: torch.tensor(x).long())
test_data['Year-Of-Publication'] = test_data['Year-Of-Publication'].apply(lambda x: torch.tensor(x).float().unsqueeze(0))
test_data['Location'] = test_data['Location'].apply(lambda x: torch.tensor(x).long())
test_data['Publisher'] = test_data['Publisher'].apply(lambda x: torch.tensor(x).long())

user_ids_test = torch.tensor(test_data['User-ID'].values).long()
book_ids_test = torch.tensor(test_data['ISBN'].values).long()
ratings_test = torch.tensor(test_data['Book-Rating'].values).float()
ages_test = torch.stack(list(test_data['Age'].values))
locations_test = torch.tensor(test_data['Location'].values.tolist()).long()
authors_test = torch.tensor(test_data['Book-Author'].values.tolist()).long()
years_test = torch.stack(list(test_data['Year-Of-Publication'].values))
publishers_test = torch.tensor(test_data['Publisher'].values.tolist()).long()

test_dataset = TensorDataset(user_ids_test, book_ids_test, ratings_test, ages_test, locations_test, authors_test, years_test, publishers_test)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the trained model
embedding_dim = 10
num_users = len(user_id_mapping)
num_books = len(book_id_mapping)
num_locations = users['Location'].nunique()
num_authors = books['Book-Author'].nunique()
num_publishers = books['Publisher'].nunique()

model = EmbeddingNet(num_users, num_books, embedding_dim, num_locations, num_authors, num_publishers)
model.load_state_dict(torch.load('embedding_model.pth'))
model.eval()

book_embeddings = {}
for book_id in book_id_mapping.values():
    if book_id in books.index:
        book_idx = torch.tensor([book_id]).long()
        author_id = torch.tensor([books.loc[book_id, 'Book-Author']]).long()
        year = torch.tensor([books.loc[book_id, 'Year-Of-Publication']]).float().unsqueeze(0)
        publisher_id = torch.tensor([books.loc[book_id, 'Publisher']]).long()
        
        with torch.no_grad():
            embedding = model.book_embedding(book_idx).squeeze().numpy()
            book_embeddings[book_id] = embedding

# Evaluation function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            user_id, book_id, rating, age, location_id, author_id, year, publisher_id = batch

            user_id = user_id.long()
            book_id = book_id.long()
            rating = rating.float()

            embedding = model(user_id, book_id, location_id, age, author_id, year, publisher_id)
            output = torch.sum(embedding, dim=1)

            loss = criterion(output, rating)
            total_loss += loss.item()
    return np.sqrt(total_loss / len(dataloader.dataset))

# Initialize the criterion
criterion = nn.MSELoss()

# Evaluate the model
rmse = evaluate_model(model, test_dataloader, criterion)
print(f'Test RMSE: {rmse}')

def precision_at_k(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]
    return len(set(predicted) & set(actual)) / float(k)

def recall_at_k(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]
    return len(set(predicted) & set(actual)) / float(len(actual))

def recommend_books(user_index, user_location, user_age, model, num_recommendations=15):
    scores = {}
    user_embed = model.user_embedding(user_index).squeeze().detach().numpy()
    
    for book_id, book_embed in book_embeddings.items():
        score = cosine_similarity([user_embed], [book_embed])[0][0]
        scores[book_id] = score

    recommended_books = sorted(scores, key=scores.get, reverse=True)[:num_recommendations]
    return recommended_books


def evaluate_model2(test_dataloader, model, k=15):
    precisions = []
    recalls = []
    
    for batch in test_dataloader:
        user_ids, book_ids, ratings, ages, locations, authors, years, publishers = batch
        for i, user_id in enumerate(user_ids):
            user_index = user_ids[i].unsqueeze(0)
            user_location = locations[i].unsqueeze(0)
            user_age = ages[i].unsqueeze(0)
            
            actual_books = test_data[test_data['User-ID'] == user_id.item()]['ISBN'].values
            if len(actual_books) == 0:
                continue
            
            recommended_books = recommend_books(user_index, user_location, user_age, model, num_recommendations=k)
            
            # Calculate similarity with actual books
            actual_book_embeddings = np.array([book_embeddings[book_id_mapping[book]] for book in actual_books if book in book_id_mapping])
            recommended_book_embeddings = np.array([book_embeddings[book_id] for book_id in recommended_books])
            
            if actual_book_embeddings.size == 0 or recommended_book_embeddings.size == 0:
                continue
            
            similarities = cosine_similarity(recommended_book_embeddings, actual_book_embeddings)
            
            liked_books = (similarities > 0.5).sum(axis=1) > 0  # Define liked books based on similarity threshold
            
            precision = liked_books.sum() / len(recommended_books)
            recall = liked_books.sum() / len(actual_books)
            
            precisions.append(precision)
            recalls.append(recall)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    return avg_precision, avg_recall


def test_one_user(user_id, test_dataloader, model, k=10):
    user_data = None
    for batch in test_dataloader:
        user_ids, book_ids, ratings, ages, locations, authors, years, publishers = batch
        for i, uid in enumerate(user_ids):
            if uid.item() == user_id:
                user_data = {
                    'user_index': uid.unsqueeze(0),
                    'user_location': locations[i].unsqueeze(0),
                    'user_age': ages[i].unsqueeze(0),
                    'actual_books': test_data[test_data['User-ID'] == uid.item()]['ISBN'].values
                }
                break
        if user_data is not None:
            break
    
    if user_data is None:
        print(f"User {user_id} not found in test data.")
        return
    print(user_data)
    
    user_index = user_data['user_index']
    user_location = user_data['user_location']
    user_age = user_data['user_age']
    actual_books = user_data['actual_books']
    
    recommended_books = recommend_books(user_index, user_location, user_age, model, num_recommendations=k)
    
    precision = precision_at_k(actual_books, recommended_books, k)
    recall = recall_at_k(actual_books, recommended_books, k)
    
    print(f"User ID: {user_id}")
    print(f"Actual Books: {actual_books}")
    print(f"Recommended Books: {recommended_books}")
    print(f"Precision@{k}: {precision}")
    print(f"Recall@{k}: {recall}")

avg_precision, avg_recall = evaluate_model2(test_dataloader, model, k=15)
print(f'Average Precision@15: {avg_precision}')
print(f'Average Recall@15: {avg_recall}')


