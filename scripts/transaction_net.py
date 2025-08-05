import pandas as pd
import numpy as np

def clean_review_data(input_file, output_file=None):
    """
    Clean review data by:
    1. Removing rows with 'unknown' userId
    2. Rounding helpfulness values to 4 decimal places
    3. Validating rating and helpfulness ranges
    """
    
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Step 1: Remove rows with 'unknown' userId
    print("\n1. Removing 'unknown' users...")
    original_count = len(df)
    df = df[df['userId'] != 'unknown']
    removed_count = original_count - len(df)
    print(f"Removed {removed_count} rows with 'unknown' userId")
    print(f"Remaining rows: {len(df)}")
    
    # Step 2: Round helpfulness values to 4 decimal places
    print("\n2. Rounding helpfulness values...")
    df['helpfulness'] = df['helpfulness'].round(4)
    print("Helpfulness values rounded to 4 decimal places")
    
    # Step 3: Remove duplicates
    print("\n3. Removing duplicates...")
    original_count = len(df)
    
    # Check for duplicates based on userId + productId combination
    duplicates = df.duplicated(subset=['userId', 'productId'], keep='first')
    duplicate_count = duplicates.sum()
    
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate user-product combinations")
        print("Keeping the first occurrence of each duplicate...")
        df = df.drop_duplicates(subset=['userId', 'productId'], keep='first')
        print(f"Removed {duplicate_count} duplicate rows")
    else:
        print("No duplicates found based on userId + productId")
    
    print(f"Remaining rows after duplicate removal: {len(df)}")
    
    # Step 4: Validate data ranges
    print("\n4. Validating data ranges...")
    
    # Check rating range (should be 1.0-5.0)
    invalid_ratings = df[(df['rating'] < 1.0) | (df['rating'] > 5.0)]
    if len(invalid_ratings) > 0:
        print(f"WARNING: Found {len(invalid_ratings)} rows with invalid ratings (outside 1.0-5.0 range)")
        print("Invalid rating values:", invalid_ratings['rating'].unique())
    else:
        print("✓ All ratings are within valid range (1.0-5.0)")
    
    # Check helpfulness range (should be 0.0-1.0)
    invalid_helpfulness = df[(df['helpfulness'] < 0.0) | (df['helpfulness'] > 1.0)]
    if len(invalid_helpfulness) > 0:
        print(f"WARNING: Found {len(invalid_helpfulness)} rows with invalid helpfulness (outside 0.0-1.0 range)")
        print("Invalid helpfulness values:", invalid_helpfulness['helpfulness'].unique())
    else:
        print("✓ All helpfulness values are within valid range (0.0-1.0)")
    
    # Check for negative prices (might be data entry errors)
    negative_prices = df[df['price'] < 0.0]
    if len(negative_prices) > 0:
        print(f"WARNING: Found {len(negative_prices)} rows with negative prices")
    else:
        print("✓ No negative prices found")
    
    # Display summary statistics
    print("\n5. Summary statistics after cleaning:")
    print(f"Final data shape: {df.shape}")
    print("\nRating distribution:")
    print(df['rating'].value_counts().sort_index())
    print(f"\nHelpfulness range: {df['helpfulness'].min()} - {df['helpfulness'].max()}")
    print(f"Price range: {df['price'].min()} - {df['price'].max()}")
    print(f"Unique users: {df['userId'].nunique()}")
    print(f"Unique products: {df['productId'].nunique()}")
    
    # Save cleaned data
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")
    
    return df

# Example usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual file path
    input_filename = 'data/processed/transactions.csv'
    output_filename = 'transactions_net.csv'
    
    try:
        cleaned_df = clean_review_data(input_filename, output_filename)
        print("\n✓ Data cleaning completed successfully!")
        
        # Display first few rows of cleaned data
        print("\nFirst 5 rows of cleaned data:")
        print(cleaned_df.head())
        
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
        print("Please update the input_filename variable with your actual file path.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

