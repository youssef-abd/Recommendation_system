import os
import re
import pandas as pd

input_path = 'data/raw/Electronics.txt'
output_csv = 'data/processed/Electronics.csv'

def parse_reviews(file_path):
    reviews = []
    current_review = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current_review:
                    reviews.append(current_review)
                    current_review = {}
                continue

            if line.startswith("product/productId:"):
                current_review["productId"] = line.split("product/productId:")[1].strip()
            elif line.startswith("product/title:"):
                current_review["title"] = line.split("product/title:")[1].strip()
            elif line.startswith("product/price:"):
                price = line.split("product/price:")[1].strip()
                try:
                    current_review["price"] = float(price)
                except:
                    current_review["price"] = None
            elif line.startswith("review/userId:"):
                current_review["userId"] = line.split("review/userId:")[1].strip()
            elif line.startswith("review/profileName:"):
                current_review["profileName"] = line.split("review/profileName:")[1].strip()
            elif line.startswith("review/helpfulness:"):
                helpfulness = line.split("review/helpfulness:")[1].strip()
                try:
                    num, den = helpfulness.split('/')
                    ratio = float(num) / float(den) if int(den) > 0 else 0.0
                except:
                    ratio = 0.0
                current_review["helpfulness"] = ratio
            elif line.startswith("review/score:"):
                current_review["score"] = float(line.split("review/score:")[1].strip())
            elif line.startswith("review/time:"):
                current_review["time"] = line.split("review/time:")[1].strip()
            elif line.startswith("review/summary:"):
                current_review["summary"] = line.split("review/summary:")[1].strip()
            elif line.startswith("review/text:"):
                current_review["text"] = line.split("review/text:")[1].strip()

    return pd.DataFrame(reviews)

df = parse_reviews(input_path)

# Save full dataset
df.to_csv(output_csv, index=False)
print(f"Converted {len(df)} reviews to CSV: {output_csv}")

# Aggregate data for Streamlit tables
user_ratings = df[['userId', 'productId', 'score', 'helpfulness']]

user_ratings.to_csv("data/processed/user_ratings.csv", index=False)

product_ratings = df.groupby('productId').agg({
    'userId': lambda x: list(x),
    'price': 'mean'
}).reset_index()
product_ratings.rename(columns={'price': 'avg_price'}, inplace=True)
product_ratings.to_csv("data/processed/product_ratings.csv", index=False)
