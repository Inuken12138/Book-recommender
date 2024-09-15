import numpy as np
from scipy.sparse.linalg import svds
import pandas as pd
import warnings
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

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
# ratings = ratings[ratings['ISBN'].isin(books['ISBN']) & ratings['User-ID'].isin(users['User-ID'])]

all_user_ids = ratings['User-ID'].unique()
all_book_ids = ratings['ISBN'].unique()

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

train_user_ids = set(train_data['User-ID'])
train_book_ids = set(train_data['ISBN'])
missing_users = set(all_user_ids) - train_user_ids
missing_books = set(all_book_ids) - train_book_ids

missing_data = ratings[ratings['User-ID'].isin(missing_users) | ratings['ISBN'].isin(missing_books)]
train_data = pd.concat([train_data, missing_data]).drop_duplicates()

n_users = ratings['User-ID'].nunique()
n_items = ratings['ISBN'].nunique()
train_matrix = csr_matrix((train_data['Book-Rating'], (train_data['User-ID'], train_data['ISBN'])), shape=(n_users, n_items))
test_matrix = csr_matrix((test_data['Book-Rating'], (test_data['User-ID'], test_data['ISBN'])), shape=(n_users, n_items))
print("-------matrix finished---------")

if np.any(np.isnan(train_matrix.data)):
    print("NaN values found in training matrix")
else:
    print("No NaN values in training matrix")

scaler = StandardScaler()
users[['Age']] = scaler.fit_transform(users[['Age']])
books[['Year-Of-Publication']] = scaler.fit_transform(books[['Year-Of-Publication']])

# als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20, use_gpu=False, calculate_training_loss=True)

# print("Training ALS model...")
# als_model.fit(train_matrix.T, show_progress=True)
# print("Model training completed.")

class EmbeddingNet(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim, num_locations, num_authors, num_publishers):
    # def __init__(self, num_users, num_books, embedding_dim, num_authors):
        super(EmbeddingNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        self.location_embedding = nn.Embedding(num_locations, embedding_dim)
        self.author_embedding = nn.Embedding(num_authors, embedding_dim)
        self.publisher_embedding = nn.Embedding(num_publishers, embedding_dim)
        self.user_age = nn.Linear(1, embedding_dim)
        self.book_year = nn.Linear(1, embedding_dim)
    
    def forward(self, user_id, book_id, location_id, age, author_id, year, publisher_id):
    # def forward(self, user_id, book_id, age, author_id, year):
        user_embed = self.user_embedding(user_id).squeeze()
        book_embed = self.book_embedding(book_id).squeeze()
        location_embed = self.location_embedding(location_id).squeeze()
        author_embed = self.author_embedding(author_id).squeeze()
        publisher_embed = self.publisher_embedding(publisher_id).squeeze()
        age_embed = self.user_age(age).squeeze()
        year_embed = self.book_year(year).squeeze()
        return torch.cat([user_embed, book_embed, location_embed, age_embed, author_embed, year_embed, publisher_embed], dim=-1)
        # return torch.cat([user_embed, book_embed, author_embed, age_embed, year_embed], dim=-1)

# train_data = train_data.merge(users[['User-ID', 'Age']], on='User-ID')
# train_data = train_data.merge(books[['ISBN', 'Book-Author', 'Year-Of-Publication']], on='ISBN')
train_data = train_data.merge(users[['User-ID', 'Age', 'Location']], on='User-ID')
train_data = train_data.merge(books[['ISBN', 'Book-Author', 'Year-Of-Publication', 'Publisher']], on='ISBN')

train_data['Age'] = train_data['Age'].apply(lambda x: torch.tensor(x).float().unsqueeze(0))
train_data['Book-Author'] = train_data['Book-Author'].apply(lambda x: torch.tensor(x).long())
train_data['Year-Of-Publication'] = train_data['Year-Of-Publication'].apply(lambda x: torch.tensor(x).float().unsqueeze(0))
train_data['Location'] = train_data['Location'].apply(lambda x: torch.tensor(x).long())
train_data['Publisher'] = train_data['Publisher'].apply(lambda x: torch.tensor(x).long())

user_ids = torch.tensor(train_data['User-ID'].values).long()
book_ids = torch.tensor(train_data['ISBN'].values).long()
ratings = torch.tensor(train_data['Book-Rating'].values).float()
ages = torch.stack(list(train_data['Age'].values))
locations = torch.tensor(train_data['Location'].values.tolist()).long()
authors = torch.tensor(train_data['Book-Author'].values.tolist()).long()
years = torch.stack(list(train_data['Year-Of-Publication'].values))
publishers = torch.tensor(train_data['Publisher'].values.tolist()).long()

# dataset = TensorDataset(user_ids, book_ids, ratings, ages, authors, years)
dataset = TensorDataset(user_ids, book_ids, ratings, ages, locations, authors, years, publishers)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

embedding_dim = 10
num_users = len(user_id_mapping)
num_books = len(book_id_mapping)
num_locations = users['Location'].nunique()
num_authors = books['Book-Author'].nunique()
num_publishers = books['Publisher'].nunique()

print(f"{num_users} users, {num_books} books")
print(f"dataloader: {len(dataloader)}")

model = EmbeddingNet(num_users, num_books, embedding_dim, num_locations, num_authors, num_publishers)
# model = EmbeddingNet(num_users, num_books, embedding_dim, num_authors)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 20
for epoch in range(num_epochs):
    print(f'epoch{epoch} starts:')
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        user_id, book_id, rating, age, location_id, author_id, year, publisher_id = batch
        
        user_id = user_id.long()
        book_id = book_id.long()
        rating = rating.float()
        
        embedding = model(user_id, book_id, location_id, age, author_id, year, publisher_id)
        # embedding = model(user_id, book_id, age, author_id, year)

        output = torch.sum(embedding, dim=1)
        # print(f'Rating: {rating}, Output: {output}')
        
        optimizer.zero_grad()
        # output = embedding.dot(embedding)
        loss = criterion(output, rating)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # print(f"{i}/{len(dataloader)} in epoch {epoch}")
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_data)}')

torch.save(model.state_dict(), 'embedding_model.pth')

user_embeddings = model.user_embedding.weight.data.numpy()
book_embeddings = model.book_embedding.weight.data.numpy()

np.save('user_embeddings.npy', user_embeddings)
np.save('book_embeddings.npy', book_embeddings)