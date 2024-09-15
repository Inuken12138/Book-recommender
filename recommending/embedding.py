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
from sklearn.metrics.pairwise import cosine_similarity
import csv

warnings.filterwarnings("ignore")


data = pd.read_csv("./final_df_parallel.csv", low_memory=False)


def preprocess_data1(data):
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["Age"] = data["Age"].astype(int)
    data["Location"] = data["Location"].astype("category").cat.codes
    return data


def preprocess_data(data):
    data.fillna("", inplace=True)
    data["Year-Of-Publication"] = pd.to_numeric(
        data["Year-Of-Publication"], errors="coerce"
    )
    data["Year-Of-Publication"].fillna(
        data["Year-Of-Publication"].median(), inplace=True
    )
    data["Publisher"] = data["Publisher"].astype("category").cat.codes
    data["Book-Author"] = data["Book-Author"].astype("category").cat.codes
    return data


data = preprocess_data1(data)
data = preprocess_data(data)
data["ISBN"] = data["ISBN"].astype(str)

user_id_mapping = {id: idx for idx, id in enumerate(data["User-ID"].unique())}
book_id_mapping = {id: idx for idx, id in enumerate(data["ISBN"].unique())}


data["User-ID"] = data["User-ID"].map(user_id_mapping)
# data['ISBN'] = data['ISBN'].astype(str)
# data['ISBN'] = data['ISBN'].astype(str)
# data['ISBN'] = data['ISBN'].map(book_id_mapping)
# data['ISBN'] = data['ISBN'].map(book_id_mapping)
data["book_id"] = data["ISBN"].map(book_id_mapping)
data.dropna(subset=["Book-Rating"], inplace=True)
# data = data[data['ISBN'].isin(data['ISBN']) & data['User-ID'].isin(data['User-ID'])]

all_user_ids = data["User-ID"].unique()
all_book_ids = data["book_id"].unique()

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_user_ids = set(train_data["User-ID"])
train_book_ids = set(train_data["book_id"])
# missing_data = set(all_user_ids) - train_user_ids
# missing_data = set(all_book_ids) - train_book_ids

# missing_data = data[data['User-ID'].isin(missing_data) | data['ISBN'].isin(missing_data)]
# train_data = pd.concat([train_data, missing_data]).drop_duplicates()
print(len(book_id_mapping))
print(data["Location"].nunique())
print(data["Location"][:20])

n_users = data["User-ID"].nunique()
n_items = data["book_id"].nunique()
print(f"{n_users} users and {n_items} items")
train_matrix = csr_matrix(
    (train_data["Book-Rating"], (train_data["User-ID"], train_data["book_id"])),
    shape=(n_users, n_items),
)
test_matrix = csr_matrix(
    (test_data["Book-Rating"], (test_data["User-ID"], test_data["book_id"])),
    shape=(n_users, n_items),
)
print("-------matrix finished---------")

if np.any(np.isnan(train_matrix.data)):
    print("NaN values found in training matrix")
else:
    print("No NaN values in training matrix")

scaler = StandardScaler()
data[["Age"]] = scaler.fit_transform(data[["Age"]])
data[["Year-Of-Publication"]] = scaler.fit_transform(data[["Year-Of-Publication"]])

# als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20, use_gpu=False, calculate_training_loss=True)

# print("Training ALS model...")
# als_model.fit(train_matrix.T, show_progress=True)
# print("Model training completed.")


class EmbeddingNet(nn.Module):
    def __init__(
        self,
        num_users,
        num_books,
        embedding_dim,
        num_locations,
        num_authors,
        num_publishers,
    ):
        # def __init__(self, num_data, num_data, embedding_dim, num_authors):
        super(EmbeddingNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        self.location_embedding = nn.Embedding(num_locations, embedding_dim)
        self.author_embedding = nn.Embedding(num_authors, embedding_dim)
        self.publisher_embedding = nn.Embedding(num_publishers, embedding_dim)
        self.user_age = nn.Linear(1, embedding_dim)
        self.book_year = nn.Linear(1, embedding_dim)

    def forward(
        self, user_id, book_id, location_id, age, author_id, year, publisher_id
    ):
        # def forward(self, user_id, book_id, age, author_id, year):
        user_embed = self.user_embedding(user_id).squeeze()
        book_embed = self.book_embedding(book_id).squeeze()
        location_embed = self.location_embedding(location_id).squeeze()
        author_embed = self.author_embedding(author_id).squeeze()
        publisher_embed = self.publisher_embedding(publisher_id).squeeze()
        age_embed = self.user_age(age).squeeze()
        year_embed = self.book_year(year).squeeze()
        return torch.cat(
            [
                user_embed,
                book_embed,
                location_embed,
                age_embed,
                author_embed,
                year_embed,
                publisher_embed,
            ],
            dim=-1,
        )
        # return torch.cat([user_embed, book_embed, author_embed, age_embed, year_embed], dim=-1)


# train_data = train_data.merge(data[['User-ID', 'Age']], on='User-ID')
# train_data = train_data.merge(data[['ISBN', 'Book-Author', 'Year-Of-Publication']], on='ISBN')
# train_data = train_data.merge(data[['User-ID', 'Age', 'Location']], on='User-ID')
# train_data = train_data.merge(data[['ISBN', 'Book-Author', 'Year-Of-Publication', 'Publisher']], on='ISBN')

train_data["Age"] = train_data["Age"].apply(
    lambda x: torch.tensor(x).float().unsqueeze(0)
)
train_data["Book-Author"] = train_data["Book-Author"].apply(
    lambda x: torch.tensor(x).long()
)
train_data["Year-Of-Publication"] = train_data["Year-Of-Publication"].apply(
    lambda x: torch.tensor(x).float().unsqueeze(0)
)
train_data["Location"] = train_data["Location"].apply(lambda x: torch.tensor(x).long())
train_data["Publisher"] = train_data["Publisher"].apply(
    lambda x: torch.tensor(x).long()
)

user_ids = torch.tensor(train_data["User-ID"].values).long()
book_ids = torch.tensor(train_data["book_id"].values).long()
rates = torch.tensor(train_data["Book-Rating"].values).float()
ages = torch.stack(list(train_data["Age"].values))
locations = torch.tensor(train_data["Location"].values.tolist()).long()
authors = torch.tensor(train_data["Book-Author"].values.tolist()).long()
years = torch.stack(list(train_data["Year-Of-Publication"].values))
publishers = torch.tensor(train_data["Publisher"].values.tolist()).long()

dataset = TensorDataset(user_ids, book_ids, data, ages, authors, years)
dataset = TensorDataset(
    user_ids, book_ids, rates, ages, locations, authors, years, publishers
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# Ensure that your tensors are correctly created
print(f"user_ids shape: {user_ids.shape}")
print(f"book_ids shape: {book_ids.shape}")
print(f"rates shape: {rates.shape}")
print(f"ages shape: {ages.shape}")
print(f"locations shape: {locations.shape}")
print(f"authors shape: {authors.shape}")
print(f"years shape: {years.shape}")
print(f"publishers shape: {publishers.shape}")

# Create TensorDataset
dataset = TensorDataset(
    user_ids, book_ids, rates, ages, locations, authors, years, publishers
)
print(f"Total dataset size: {len(dataset)}")

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"Total batches in dataloader: {len(dataloader)}")

embedding_dim = 10
num_users = len(user_id_mapping)
num_books = len(book_id_mapping)
num_locations = data["Location"].nunique()
num_authors = data["Book-Author"].nunique()
num_publishers = data["Publisher"].nunique()

print(f"{num_users} users, {num_books} books")
print(f"dataloader: {len(dataloader)}")

test_data["Age"] = test_data["Age"].apply(
    lambda x: torch.tensor(x).float().unsqueeze(0)
)
test_data["Book-Author"] = test_data["Book-Author"].apply(
    lambda x: torch.tensor(x).long()
)
test_data["Year-Of-Publication"] = test_data["Year-Of-Publication"].apply(
    lambda x: torch.tensor(x).float().unsqueeze(0)
)
test_data["Location"] = test_data["Location"].apply(lambda x: torch.tensor(x).long())
test_data["Publisher"] = test_data["Publisher"].apply(lambda x: torch.tensor(x).long())

user_ids_test = torch.tensor(test_data["User-ID"].values).long()
book_ids_test = torch.tensor(test_data["book_id"].values).long()
ratings_test = torch.tensor(test_data["Book-Rating"].values).float()
ages_test = torch.stack(list(test_data["Age"].values))
locations_test = torch.tensor(test_data["Location"].values.tolist()).long()
authors_test = torch.tensor(test_data["Book-Author"].values.tolist()).long()
years_test = torch.stack(list(test_data["Year-Of-Publication"].values))
publishers_test = torch.tensor(test_data["Publisher"].values.tolist()).long()

test_dataset = TensorDataset(
    user_ids_test,
    book_ids_test,
    ratings_test,
    ages_test,
    locations_test,
    authors_test,
    years_test,
    publishers_test,
)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"test dataloader: {len(test_dataloader)}")

model = EmbeddingNet(
    num_users, num_books, embedding_dim, num_locations, num_authors, num_publishers
)
# model = EmbeddingNet(num_data, num_data, embedding_dim, num_authors)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# model = EmbeddingNet(
#     num_users, num_books, embedding_dim, num_locations, num_authors, num_publishers
# )
# model.load_state_dict(torch.load("embedding_model.pth"))
# model.eval()


# book_embeddings = np.load("book_embeddings.npy", allow_pickle=True)

# # Check if embeddings are loaded correctly
# print(f"Loaded book embeddings for {len(book_embeddings)} books.")

# inverse_book_id_mapping = {v: k for k, v in book_id_mapping.items()}


def get_book_embedding(isbn):
    """Retrieve the embedding for a given ISBN."""
    book_id = book_id_mapping.get(isbn)
    if book_id is not None:
        return book_embeddings[book_id]
    else:
        return None


print(get_book_embedding(data["ISBN"][0]))

num_epochs = 20
for epoch in range(num_epochs):
    print(f"epoch{epoch} starts:")
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        user_id, book_id, rating, age, location_id, author_id, year, publisher_id = (
            batch
        )

        user_id = user_id.long()
        book_id = book_id.long()
        rating = rating.float()

        embedding = model(
            user_id, book_id, location_id, age, author_id, year, publisher_id
        )
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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_data)}")

torch.save(model.state_dict(), "embedding_model.pth")

user_embeddings = model.user_embedding.weight.data.numpy()
book_embeddings = model.book_embedding.weight.data.numpy()

np.save("user_embeddings.npy", user_embeddings)
np.save("book_embeddings.npy", book_embeddings)


# def evaluate_model(model, dataloader, criterion):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for batch in dataloader:
#             (
#                 user_id,
#                 book_id,
#                 rating,
#                 age,
#                 location_id,
#                 author_id,
#                 year,
#                 publisher_id,
#             ) = batch

#             user_id = user_id.long()
#             book_id = book_id.long()
#             rating = rating.float()

#             embedding = model(
#                 user_id, book_id, location_id, age, author_id, year, publisher_id
#             )
#             output = torch.sum(embedding, dim=1)

#             loss = criterion(output, rating)
#             total_loss += loss.item()
#     return np.sqrt(total_loss / len(dataloader.dataset))


# # Initialize the criterion
# criterion = nn.MSELoss()

# # Evaluate the model
# rmse = evaluate_model(model, test_dataloader, criterion)
# print(f"Test RMSE: {rmse}")


# def precision_at_k(actual, predicted, k):
#     if len(predicted) > k:
#         predicted = predicted[:k]
#     return len(set(predicted) & set(actual)) / float(k)


# def recall_at_k(actual, predicted, k):
#     if len(predicted) > k:
#         predicted = predicted[:k]
#     return len(set(predicted) & set(actual)) / float(len(actual))


# def recommend_books(user_id, model, num_recommendations=15):
#     user_index = torch.tensor(
#         [user_id_mapping[user_id]]
#     ).long()  # Ensure user_index is a Tensor

#     with torch.no_grad():
#         user_embed = model.user_embedding(user_index).squeeze().numpy()

#     scores = {}
#     for book_id in range(book_embeddings.shape[0]):
#         book_embed = book_embeddings[book_id]
#         score = cosine_similarity([user_embed], [book_embed])[0][0]
#         scores[book_id] = score

#     recommended_book_indices = sorted(scores, key=scores.get, reverse=True)[
#         :num_recommendations
#     ]
#     recommended_books = [
#         inverse_book_id_mapping[idx] for idx in recommended_book_indices
#     ]
#     return recommended_books


# def evaluate_model2(test_dataloader, model, k=15):
#     precisions = []
#     recalls = []

#     for batch in test_dataloader:
#         user_ids, book_ids, ratings, ages, locations, authors, years, publishers = batch
#         for i, user_id in enumerate(user_ids):
#             user_index = torch.tensor(
#                 [user_id_mapping[user_id.item()]]
#             ).long()  # Ensure user_index is a Tensor
#             user_location = locations[i].unsqueeze(0)
#             user_age = ages[i].unsqueeze(0)

#             actual_books = test_data[test_data["User-ID"] == user_id.item()][
#                 "ISBN"
#             ].values
#             if len(actual_books) == 0:
#                 continue

#             recommended_books = recommend_books(
#                 user_index, user_location, user_age, model, num_recommendations=k
#             )

#             # Calculate similarity with actual books
#             actual_book_embeddings = np.array(
#                 [
#                     book_embeddings[book_id_mapping[book]]
#                     for book in actual_books
#                     if book in book_id_mapping
#                 ]
#             )
#             recommended_book_embeddings = np.array(
#                 [book_embeddings[book_id_mapping[book]] for book in recommended_books]
#             )

#             if (
#                 actual_book_embeddings.size == 0
#                 or recommended_book_embeddings.size == 0
#             ):
#                 continue

#             similarities = cosine_similarity(
#                 recommended_book_embeddings, actual_book_embeddings
#             )

#             liked_books = (similarities > 0.5).sum(
#                 axis=1
#             ) > 0  # Define liked books based on similarity threshold

#             precision = liked_books.sum() / len(recommended_books)
#             recall = liked_books.sum() / len(actual_books)

#             precisions.append(precision)
#             recalls.append(recall)

#     avg_precision = np.mean(precisions)
#     avg_recall = np.mean(recalls)

#     return avg_precision, avg_recall


# def test_one_user(user_id, test_dataloader, model, k=10):
#     user_data = None
#     for batch in test_dataloader:
#         user_ids, book_ids, ratings, ages, locations, authors, years, publishers = batch
#         for i, uid in enumerate(user_ids):
#             if uid.item() == user_id:
#                 user_data = {
#                     "user_index": torch.tensor(
#                         [user_id_mapping[uid.item()]]
#                     ).long(),  # Ensure user_index is a Tensor
#                     "user_location": locations[i].unsqueeze(0),
#                     "user_age": ages[i].unsqueeze(0),
#                     "actual_books": test_data[test_data["User-ID"] == uid.item()][
#                         "ISBN"
#                     ].values,
#                 }
#                 break
#         if user_data is not None:
#             break

#     if user_data is None:
#         print(f"User {user_id} not found in test data.")
#         return
#     print(user_data)

#     user_index = user_data["user_index"]
#     user_location = user_data["user_location"]
#     user_age = user_data["user_age"]
#     actual_books = user_data["actual_books"]

#     recommended_books = recommend_books(
#         user_index, user_location, user_age, model, num_recommendations=k
#     )

#     precision = precision_at_k(actual_books, recommended_books, k)
#     recall = recall_at_k(actual_books, recommended_books, k)

#     print(f"User ID: {user_id}")
#     print(f"Actual Books: {actual_books}")
#     print(f"Recommended Books: {recommended_books}")
#     print(f"Precision@{k}: {precision}")
#     print(f"Recall@{k}: {recall}")


# # avg_precision, avg_recall = evaluate_model2(test_dataloader, model, k=15)
# # print(f'Average Precision@15: {avg_precision}')
# # print(f'Average Recall@15: {avg_recall}')

# # print(data['User-ID'][1])
# # print(recommend_books(2,model))
# # print(recommend_books(8,model))
# # print(user_id_mapping[2])
# # print(user_id_mapping[11400])

# output_file = "user_recommendations.csv"
# new_data = pd.read_csv("./final_df_parallel.csv")

# # Open the file in write mode
# with open(output_file, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     # Write the header
#     writer.writerow(["User-ID", "Recommended Books"])
#     total = new_data["User-ID"].nunique()
#     cnt = 1
#     # Iterate over all unique user IDs
#     for user_id in new_data["User-ID"].unique():
#         try:
#             # Generate recommendations for the user
#             recommended_books = recommend_books(user_id, model, num_recommendations=15)
#             # Write the user ID and recommended books to the file
#             writer.writerow([user_id] + recommended_books)
#             print(f"{cnt} / {total} record saved")
#             cnt += 1
#             if cnt > 200:
#                 break
#         except Exception as e:
#             print(f"Error generating recommendations for user {user_id}: {e}")

# print(f"Recommendations have been saved to {output_file}")
