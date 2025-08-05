import pandas as pd
import numpy as np

def clean_user_ratings_csv(input_file, output_file):
    """
    Clean user ratings CSV file by removing unknown users and applying data quality checks.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
    """
    
    print("Loading CSV file...")
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Step 1: Handle whitespace in headers and data
    print("\nStep 1: Cleaning whitespace...")
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from string columns
    if 'userId' in df.columns:
        df['userId'] = df['userId'].astype(str).str.strip()
    if 'productId' in df.columns:
        df['productId'] = df['productId'].astype(str).str.strip()
    
    # Step 2: Remove "unknown" users
    print("\nStep 2: Removing 'unknown' users...")
    unknown_count = len(df[df['userId'] == 'unknown'])
    print(f"Found {unknown_count} rows with 'unknown' users")
    
    df_clean = df[df['userId'] != 'unknown'].copy()
    print(f"Removed {unknown_count} rows with unknown users")
    
    # Step 3: Remove rows with empty or null user/product IDs
    print("\nStep 3: Removing rows with missing user/product IDs...")
    initial_count = len(df_clean)
    
    # Remove rows with null values in critical columns
    df_clean = df_clean.dropna(subset=['userId', 'productId'])
    
    # Remove rows with empty strings
    df_clean = df_clean[(df_clean['userId'] != '') & (df_clean['productId'] != '')]
    
    removed_empty = initial_count - len(df_clean)
    print(f"Removed {removed_empty} rows with empty/null IDs")
    
    # Step 4: Remove duplicate ratings
    print("\nStep 4: Removing duplicate user-product combinations...")
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['userId', 'productId'], keep='first')
    duplicates_removed = initial_count - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate user-product combinations")
    
    # Step 5: Validate rating values
    print("\nStep 5: Validating rating values...")
    initial_count = len(df_clean)
    
    # Remove any rows with missing ratings
    df_clean = df_clean.dropna(subset=['rating'])
    
    # Convert rating to numeric if it's not already
    df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
    
    # Remove rows where rating conversion failed
    df_clean = df_clean.dropna(subset=['rating'])
    
    # Ensure ratings are within expected range (1-5)
    df_clean = df_clean[(df_clean['rating'] >= 1.0) & (df_clean['rating'] <= 5.0)]
    
    invalid_ratings_removed = initial_count - len(df_clean)
    print(f"Removed {invalid_ratings_removed} rows with invalid ratings")
    
    # Step 6: Optimize data types
    print("\nStep 6: Optimizing data types...")
    df_clean['userId'] = df_clean['userId'].astype('string')
    df_clean['productId'] = df_clean['productId'].astype('string')
    df_clean['rating'] = df_clean['rating'].astype('float32')
    
    # Display summary statistics
    print("\n" + "="*50)
    print("CLEANING SUMMARY")
    print("="*50)
    print(f"Original rows: {df.shape[0]}")
    print(f"Cleaned rows: {df_clean.shape[0]}")
    print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")
    print(f"Percentage retained: {(len(df_clean)/len(df))*100:.2f}%")
    
    print(f"\nUnique users: {df_clean['userId'].nunique()}")
    print(f"Unique products: {df_clean['productId'].nunique()}")
    
    print(f"\nRating distribution:")
    rating_dist = df_clean['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        print(f"  {rating}: {count} ({count/len(df_clean)*100:.1f}%)")
    
    # Save cleaned data
    print(f"\nSaving cleaned data to: {output_file}")
    df_clean.to_csv(output_file, index=False)
    
    return df_clean

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    input_file = "data/processed/user_ratings.csv"
    output_file = "user_ratings_net.csv"
    
    try:
        cleaned_df = clean_user_ratings_csv(input_file, output_file)
        print("\nCleaning completed successfully!")
        
        # Display first few rows of cleaned data
        print("\nFirst 5 rows of cleaned data:")
        print(cleaned_df.head())
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")