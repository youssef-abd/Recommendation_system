from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, first
import os

def clean_product_data(input_path: str, output_path: str):
    spark = SparkSession.builder \
        .appName("CleanProductData") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    try:
        print("📥 Reading raw product data...")
        df = spark.read.csv(input_path, header=True, inferSchema=True)

        print(f"🔍 Raw count: {df.count()}")

        # Drop rows with null or empty descriptions, and keep only descriptions longer than 10 chars
        df_clean = df.dropna(subset=["description"]) \
                     .filter(col("description") != "") \
                     .filter(length(col("description")) > 10)

        # Remove exact duplicates
        df_clean = df_clean.dropDuplicates()

        # Keep one entry per productId (the first description)
        df_unique = df_clean.groupBy("productId").agg(first("description").alias("description"))

        print(f"✅ Cleaned count (unique products): {df_unique.count()}")

        # Save cleaned data locally as CSV
        print(f"💾 Saving cleaned data to {output_path}...")
        # Coalesce to 1 file for easier usage
        df_unique.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

        print("🎉 Done! Cleaned file is ready.")

    except Exception as e:
        print(f"❌ Error during cleaning: {str(e)}")
    finally:
        spark.stop()
        print("🛑 Spark session stopped.")

if __name__ == "__main__":
    # Use your actual local paths here
    INPUT_CSV = "data/processed/cleaned_product_ratings2.csv"
    OUTPUT_DIR = "data/processed/cleaned_dupes_product.csv"

    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clean_product_data(INPUT_CSV, OUTPUT_DIR)
