import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load datasets
users = pd.read_csv('users.csv')  # replace with the actual path
books = pd.read_csv('books.csv')  # replace with the actual path
ratings = pd.read_csv('ratings.csv')  # replace with the actual path

# Map user and book IDs to continuous indices
user_id_mapping = {id: idx for idx, id in enumerate(ratings['User-ID'].unique())}
book_id_mapping = {id: idx for idx, id in enumerate(ratings['ISBN'].unique())}

ratings['User-ID'] = ratings['User-ID'].map(user_id_mapping)
ratings['ISBN'] = ratings['ISBN'].map(book_id_mapping)

# Create user-item rating matrix in sparse format
n_users = ratings['User-ID'].nunique()
n_items = ratings['ISBN'].nunique()
R = coo_matrix((ratings['Book-Rating'], (ratings['User-ID'], ratings['ISBN'])), shape=(n_users, n_items))

# Function to implement NMF using gradient descent
def nmf(R, k, alpha=0.01, lambda_=0.1, n_iterations=1000):
    R = R.tocsr()
    n_users, n_items = R.shape
    P = np.random.rand(n_users, k)
    Q = np.random.rand(n_items, k)

    for iteration in range(n_iterations):
        for u, i, r_ui in zip(R.row, R.col, R.data):
            error = r_ui - np.dot(P[u, :], Q[i, :])
            P[u, :] += alpha * (2 * error * Q[i, :] - 2 * lambda_ * P[u, :])
            Q[i, :] += alpha * (2 * error * P[u, :] - 2 * lambda_ * Q[i, :])

    return P, Q

def evaluate(R, P, Q):
    R_pred = P.dot(Q.T)
    mask = (R > 0)
    return np.sqrt(mean_squared_error(R[mask], R_pred[mask]))

# Split data into training and validation sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

R_train = coo_matrix((train_data['Book-Rating'], (train_data['User-ID'], train_data['ISBN'])), shape=(n_users, n_items))
R_test = coo_matrix((test_data['Book-Rating'], (test_data['User-ID'], test_data['ISBN'])), shape=(n_users, n_items))

# Range of k values to test
k_values = [2, 5, 10, 15, 20]
errors = []

for k in k_values:
    P, Q = nmf(R_train, k, alpha=0.01, lambda_=0.1, n_iterations=1000)
    error = evaluate(R_test, P, Q)
    errors.append(error)
    print(f'k={k}, RMSE={error}')

# Plot the errors
plt.plot(k_values, errors, marker='o')
plt.xlabel('Number of Latent Factors (k)')
plt.ylabel('RMSE')
plt.title('Selecting k using Cross-Validation')
plt.show()

