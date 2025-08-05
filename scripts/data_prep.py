import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler

# Load processed data (tables created in extract_tables.py)
user_ratings = pd.read_csv('data/processed/user_ratings.csv')
product_ratings = pd.read_csv('data/processed/product_ratings2.csv')
transactions = pd.read_csv('data/processed/transactions.csv')
product_ratings = product_ratings.drop(columns=['userId', 'rating'], errors='ignore')
# Step 1: Handle missing values (if any)
# For example, fill missing ratings with 0 or the mean of the column, depending on the context.
user_ratings['rating'] = user_ratings['rating'].fillna(user_ratings['rating'].mean())
# product_ratings['rating'] = product_ratings['rating'].fillna(product_ratings['rating'].mean())
transactions['rating'] = transactions['rating'].fillna(transactions['rating'].mean())

# Step 2: Normalize the data (e.g., ratings and price)
scaler = MinMaxScaler()

# Normalize ratings in the range of 0 to 1
user_ratings['normalized_rating'] = scaler.fit_transform(user_ratings[['rating']])
# product_ratings['normalized_rating'] = scaler.fit_transform(product_ratings[['rating']])

# Normalize prices if necessary (e.g., for transaction data)
transactions['normalized_price'] = scaler.fit_transform(transactions[['price']])

# Step 3: Create a sparse user-product interaction matrix for collaborative filtering
# Convert 'userId' and 'productId' to categorical values to save memory
user_ratings['userId'] = user_ratings['userId'].astype('category')
user_ratings['productId'] = user_ratings['productId'].astype('category')

# Create a sparse matrix of user-product interactions (ratings)
interaction_matrix_sparse = sp.coo_matrix(
    (user_ratings['rating'], 
     (user_ratings['userId'].cat.codes, user_ratings['productId'].cat.codes))
)

# Save the sparse matrix in .npz format (Compressed Sparse Row format)
sp.save_npz('data/processed/interaction_matrix_sparse.npz', interaction_matrix_sparse)

# Step 4: Save the cleaned and normalized data (optional)
user_ratings.to_csv('data/processed/cleaned_user_ratings.csv', index=False)
product_ratings.to_csv('data/processed/cleaned_product_ratings2.csv', index=False)
transactions.to_csv('data/processed/cleaned_transactions.csv', index=False)

print("Data preparation completed:")
print("- Normalized ratings and prices")
print("- Created sparse user-product interaction matrix")
