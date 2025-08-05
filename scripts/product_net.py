import pandas as pd
import numpy as np

def clean_product_table(input_file, output_file):
    """
    Clean product table by keeping only rows with prices when duplicates exist.
    
    Logic:
    - If a product has multiple rows and some have prices, keep only rows with prices
    - If a product has multiple rows and none have prices, keep only one row
    - If a product has only one row, keep it regardless of price
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        print(f"Original unique products: {df['productId'].nunique()}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Clean whitespace in string columns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Convert price column to numeric, handling empty strings and 'nan'
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Create a boolean mask for rows with valid prices
    df['has_price'] = df['price'].notna() & (df['price'] != '') & (df['price'] > 0)
    
    # Group by productId and description to handle duplicates
    grouped = df.groupby(['productId', 'description'])
    
    cleaned_rows = []
    
    for (product_id, description), group in grouped:
        # Check if any row in this group has a price
        rows_with_price = group[group['has_price']]
        rows_without_price = group[~group['has_price']]
        
        if len(rows_with_price) > 0:
            # If there are rows with prices, keep only those
            # If multiple rows have prices, keep the first one (you can modify this logic)
            cleaned_rows.append(rows_with_price.iloc[0])
            if len(rows_with_price) > 1:
                print(f"Product {product_id} has multiple rows with prices. Keeping first one with price: {rows_with_price.iloc[0]['price']}")
        else:
            # If no rows have prices, keep just one row
            cleaned_rows.append(group.iloc[0])
            if len(group) > 1:
                print(f"Product {product_id} has {len(group)} duplicate rows with no prices. Keeping one row.")
    
    # Create cleaned dataframe
    df_cleaned = pd.DataFrame(cleaned_rows)
    
    # Remove the helper column
    df_cleaned = df_cleaned.drop('has_price', axis=1)
    
    # Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    # Save cleaned data
    try:
        df_cleaned.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Cleaned unique products: {df_cleaned['productId'].nunique()}")
        print(f"Rows with prices: {df_cleaned['price'].notna().sum()}")
        print(f"Rows removed: {len(df) - len(df_cleaned)}")
    except Exception as e:
        print(f"Error saving file: {e}")
    
    return df_cleaned

def preview_data(file_path, n_rows=10):
    """Preview the first n rows of the data"""
    try:
        df = pd.read_csv(file_path)
        print(f"\nPreview of {file_path}:")
        print(df.head(n_rows))
        print(f"\nData info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Unique products: {df['productId'].nunique()}")
    except Exception as e:
        print(f"Error previewing file: {e}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    input_file = "data/processed/product_ratings2.csv"  # Your input CSV file
    output_file = "products_net.csv"  # Output file name
    
    print("Starting product table cleaning...")
    print("="*50)
    
    # Preview original data
    preview_data(input_file, 5)
    
    # Clean the data
    cleaned_df = clean_product_table(input_file, output_file)
    
    # Preview cleaned data
    if cleaned_df is not None:
        print("\n" + "="*50)
        preview_data(output_file, 5)
        
        # Show some statistics
        print(f"\nCleaning Summary:")
        print(f"- Products with prices: {cleaned_df['price'].notna().sum()}")
        print(f"- Products without prices: {cleaned_df['price'].isna().sum()}")
        print(f"- Average price (where available): ${cleaned_df['price'].mean():.2f}")