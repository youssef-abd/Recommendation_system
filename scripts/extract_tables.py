import pandas as pd

# Load the converted CSV
df = pd.read_csv('data/processed/Electronics.csv')

# Drop rows where any of the essential fields are missing (use correct column names)
df = df.dropna(subset=['userId', 'productId', 'score'])

# Table 1: user_ratings
user_ratings = df[['userId', 'productId', 'score']]
user_ratings.columns = ['userId', 'productId', 'rating']
user_ratings.to_csv('data/processed/user_ratings.csv', index=False)

# Table 2: product_ratings (Including title as product description)
product_ratings = df[['productId', 'title', 'price']]  # Adding title as description
product_ratings.columns = ['productId', 'description', 'price']

# CORRECTED: Remove userId and price columns (using correct column names)
# product_ratings = product_ratings.drop(columns=['userId', 'price'], errors='ignore')

product_ratings.to_csv('data/processed/product_ratings2.csv', index=False)

# Table 3: transactions
df['price'] = df['price'].fillna(0.0)  # Fill missing price with 0.0
transactions = df[['userId', 'productId', 'score', 'helpfulness', 'price', 'time']]
transactions.columns = ['userId', 'productId', 'rating', 'helpfulness', 'price', 'time']
transactions.to_csv('data/processed/transactions.csv', index=False)

print("Tables created:")
print("- user_ratings.csv")
print("- product_ratings.csv")
print("- transactions.csv")
