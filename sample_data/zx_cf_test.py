import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt


# Load the original datasets
""" books = pd.read_csv('combined_books.csv')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv') """

books = pd.read_csv('sample_books.csv')
users = pd.read_csv('sample_users.csv')
ratings = pd.read_csv('sample_ratings.csv')


# potential problem: some users don't have a ratings and some books don't have ratings
""" user_id_mapping = {id: idx for idx, id in enumerate(users['User-ID'])}
book_id_mapping = {id: idx for idx, id in enumerate(books['ISBN'])} """

""" users['User-ID'] = users['User-ID'].map(user_id_mapping)
books['ISBN'] = books['ISBN'].map(book_id_mapping) """

# data cleaning: check for duplicated users or books
# Create mappings from books.csv and users.csv
user_id_mapping = {id: idx for idx, id in enumerate(ratings['User-ID'].unique())}
book_id_mapping = {id: idx for idx, id in enumerate(ratings['ISBN'].unique())}


ratings['User-ID'] = ratings['User-ID'].map(user_id_mapping)
ratings['ISBN'] = ratings['ISBN'].map(book_id_mapping)



# Create user-item rating matrix
""" n_users = ratings['User-ID'].nunique() # optimization: just calculate the length of the mappings since they are already unique
n_items = ratings['ISBN'].nunique() """
n_users = len(user_id_mapping)
n_items = len(book_id_mapping)

row = ratings['User-ID'].values
col = ratings['ISBN'].values
data = ratings['Book-Rating'].values
#R = np.zeros((n_users, n_items))
R = csr_matrix((data, (row, col)), shape=(n_users, n_items))


# data cleaning: need data cleaning potentially. there may be duplicates 
""" for row in ratings.itertuples():
    R[row[1], row[2]] = row[3]  """


def evaluate(R, P, Q):
    R_pred = np.dot(P, Q.T)
    

    actual_ratings = R[~np.isnan(R)]
    predicted_ratings = R_pred[~np.isnan(R)]

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    return rmse

# Split indices into training and validation sets
""" indices = np.arange(R.shape[0])
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create training and validation sets based on the indices
R_train = R[train_indices, :]
R_val = R[val_indices, :] """


# Split data into training and validation sets
#R_train, R_val = train_test_split(R, test_size=0.2, random_state=42)

# Range of k values to test
#k_values = [2, 5, 10, 15, 20]
k_values = 2
#errors = []
error = 0

""" for k in k_values:
    P, Q = nmf(R_train, k, alpha=0.01, lambda_=0.1, n_iterations=1000)
    error = evaluate(R_val, P, Q)
    errors.append(error)
    print(f'k={k}, RMSE={error}') """



# Plot the errors
#plt.plot(k_values, errors, marker='o')
""" plt.plot(k_values, error, marker='o')
plt.xlabel('Number of Latent Factors (k)')
plt.ylabel('RMSE')
plt.title('Selecting k using Cross-Validation')
plt.show()
 """






# Evaluate the model
""" rmse = np.sqrt(mean_squared_error(R[R > 0], R_pred[R_pred > 0]))
print(f'k={k_values}, RMSE={rmse}') """



from implicit.als import AlternatingLeastSquares
als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=1000, use_gpu=False, calculate_training_loss=True)
als_model.fit(R.T)

# Compute the predicted rating matrix
predicted_ratings = np.dot(als_model.user_factors, als_model.item_factors.T)

print(predicted_ratings)