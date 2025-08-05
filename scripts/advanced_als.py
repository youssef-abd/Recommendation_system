import os
import json
import numpy as np
from itertools import product
import pickle
from pyspark.ml.recommendation import ALSModel
from pyspark.ml import PipelineModel

# Memory settings
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 6g --executor-memory 6g --executor-cores 2 --num-executors 2 --conf spark.driver.extraJavaOptions='-Xss4m' --conf spark.executor.extraJavaOptions='-Xss4m' pyspark-shell"

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, row_number, concat, lit, avg, stddev, 
    abs as spark_abs, count, min as spark_min, max as spark_max,
    percentile_approx, variance, skewness, kurtosis
)
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

# Enhanced Spark session
spark = SparkSession.builder \
    .appName("AdvancedALS") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

print("[INFO] Loading ratings data...")

try:
    ratings = spark.read.csv(
        "data/processed/clean/cleaned_user_ratings.csv", 
        header=True, 
        inferSchema=True
    )
    
    total_rows = ratings.count()
    print(f"[DATA] Loaded {total_rows:,} ratings")
    
    if total_rows == 0:
        print("[ERROR] No data found. Exiting.")
        spark.stop()
        exit()
        
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    spark.stop()
    exit()

# Advanced feature engineering
print("[PROCESS] Advanced feature engineering...")

# Handle unknown users
unknown_count = ratings.filter(col("userId") == "unknown").count()
if unknown_count > 0:
    print(f"[INFO] Found {unknown_count:,} unknown userIds - assigning unique IDs")
    window = Window.partitionBy(lit(1)).orderBy(col("userId"))
    ratings = ratings.withColumn(
        "row_id",
        when(col("userId") == "unknown", row_number().over(window)).otherwise(lit(0))
    ).withColumn(
        "userId",
        when(col("userId") == "unknown", concat(lit("unknown_"), col("row_id")))
        .otherwise(col("userId"))
    ).drop("row_id")

# Clean data
ratings_clean = ratings.filter(
    col("userId").isNotNull() & 
    col("productId").isNotNull() & 
    col("rating").isNotNull() &
    col("rating").between(1, 5)
)

print("[FEATURE] Creating advanced user and item features...")

# User behavior features
user_features = ratings_clean.groupBy("userId").agg(
    count("rating").alias("user_rating_count"),
    avg("rating").alias("user_avg_rating"),
    stddev("rating").alias("user_rating_std"),
    spark_min("rating").alias("user_min_rating"),
    spark_max("rating").alias("user_max_rating"),
    variance("rating").alias("user_rating_variance"),
    skewness("rating").alias("user_rating_skewness"),
    kurtosis("rating").alias("user_rating_kurtosis")
).fillna(0)  # Fill NaN values for users with single ratings

# Item popularity and quality features
item_features = ratings_clean.groupBy("productId").agg(
    count("rating").alias("item_rating_count"),
    avg("rating").alias("item_avg_rating"),
    stddev("rating").alias("item_rating_std"),
    spark_min("rating").alias("item_min_rating"),
    spark_max("rating").alias("item_max_rating"),
    variance("rating").alias("item_rating_variance"),
    skewness("rating").alias("item_rating_skewness"),
    kurtosis("rating").alias("item_rating_kurtosis"),
    percentile_approx("rating", 0.25).alias("item_rating_q1"),
    percentile_approx("rating", 0.75).alias("item_rating_q3")
).fillna(0)

print("[PROCESS] Enhanced filtering with feature-based selection...")

# More sophisticated filtering based on statistical properties
# Keep users with diverse rating patterns (higher std) and sufficient data
qualified_users = user_features.filter(
    (col("user_rating_count") >= 5) &
    (col("user_rating_std").isNotNull()) &
    (col("user_rating_std") > 0.5)  # Users who use rating scale diversity
)

# Keep items with good statistical properties
qualified_items = item_features.filter(
    (col("item_rating_count") >= 5) &
    (col("item_avg_rating").between(2.0, 5.0))  # Remove items with only extreme ratings
)

# Join with features
ratings_with_features = ratings_clean \
    .join(qualified_users.select("userId"), "userId", "inner") \
    .join(qualified_items.select("productId"), "productId", "inner") \
    .join(user_features, "userId", "left") \
    .join(item_features, "productId", "left")

# Create interaction features
ratings_enhanced = ratings_with_features.withColumn(
    "user_item_rating_diff", 
    col("rating") - col("user_avg_rating")
).withColumn(
    "rating_vs_item_avg_diff",
    col("rating") - col("item_avg_rating")
).withColumn(
    "user_popularity_score",
    col("user_rating_count") / col("item_rating_count")
)

filtered_count = ratings_enhanced.count()
print(f"[INFO] After advanced filtering: {filtered_count:,} ratings")

# Create indices
print("[PROCESS] Creating indices...")
user_indexer = StringIndexer(inputCol="userId", outputCol="userIndex", handleInvalid="skip")
product_indexer = StringIndexer(inputCol="productId", outputCol="productIndex", handleInvalid="skip")

user_model = user_indexer.fit(ratings_enhanced)
product_model = product_indexer.fit(ratings_enhanced)

ratings_indexed = user_model.transform(ratings_enhanced)
ratings_indexed = product_model.transform(ratings_indexed)

# Prepare data for hybrid model
base_features = ["userIndex", "productIndex", "rating"]
additional_features = [
    "user_avg_rating", "user_rating_std", "user_rating_count",
    "item_avg_rating", "item_rating_std", "item_rating_count",
    "user_item_rating_diff", "rating_vs_item_avg_diff", "user_popularity_score"
]

final_data = ratings_indexed.select(base_features + additional_features)
final_data = final_data.repartition(10).cache()
final_count = final_data.count()
print(f"[INFO] Final enhanced dataset: {final_count:,} ratings")

# Train/test split
print("[PROCESS] Creating train/test split...")
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

train_data.cache()
test_data.cache()
train_count = train_data.count()
test_count = test_data.count()

print(f"[INFO] Training: {train_count:,} ratings")
print(f"[INFO] Testing: {test_count:,} ratings")

# Train multiple models and ensemble
print("[MODEL] Training ensemble of models...")

# 1. Optimized ALS model
print("[MODEL] Training optimized ALS...")
best_als = ALS(
    maxIter=25,
    regParam=0.05,  # Lower regularization for better fit
    rank=75,        # Higher rank for more complex patterns
    userCol="userIndex",
    itemCol="productIndex",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    alpha=1.0,      # Confidence parameter
    seed=42
)

als_model = best_als.fit(train_data)
als_predictions = als_model.transform(test_data).select("userIndex", "productIndex", "rating", col("prediction").alias("als_prediction"))

# 2. Feature-based Random Forest model
print("[MODEL] Training Random Forest on features...")

# Prepare feature vector
feature_cols = [col for col in additional_features if col != "rating"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

rf = RandomForestRegressor(
    featuresCol="scaled_features",
    labelCol="rating",
    predictionCol="rf_prediction",
    numTrees=50,
    maxDepth=10,
    seed=42
)

# Create pipeline
rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
rf_model = rf_pipeline.fit(train_data)
rf_predictions = rf_model.transform(test_data).select("userIndex", "productIndex", "rating", "rf_prediction")

# 3. Ensemble predictions
print("[MODEL] Creating ensemble predictions...")
ensemble_predictions = als_predictions.join(
    rf_predictions, 
    ["userIndex", "productIndex", "rating"], 
    "inner"
).withColumn(
    "ensemble_prediction",
    (col("als_prediction") * 0.7 + col("rf_prediction") * 0.3)  # Weight ALS more heavily
)

# Evaluate all models
print("[EVAL] Evaluating models...")

rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating")
mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating")
r2_evaluator = RegressionEvaluator(metricName="r2", labelCol="rating")

# ALS evaluation
als_rmse = rmse_evaluator.setPredictionCol("als_prediction").evaluate(ensemble_predictions)
als_mae = mae_evaluator.setPredictionCol("als_prediction").evaluate(ensemble_predictions)
als_r2 = r2_evaluator.setPredictionCol("als_prediction").evaluate(ensemble_predictions)

# RF evaluation
rf_rmse = rmse_evaluator.setPredictionCol("rf_prediction").evaluate(ensemble_predictions)
rf_mae = mae_evaluator.setPredictionCol("rf_prediction").evaluate(ensemble_predictions)
rf_r2 = r2_evaluator.setPredictionCol("rf_prediction").evaluate(ensemble_predictions)

# Ensemble evaluation
ensemble_rmse = rmse_evaluator.setPredictionCol("ensemble_prediction").evaluate(ensemble_predictions)
ensemble_mae = mae_evaluator.setPredictionCol("ensemble_prediction").evaluate(ensemble_predictions)
ensemble_r2 = r2_evaluator.setPredictionCol("ensemble_prediction").evaluate(ensemble_predictions)

print(f"[METRICS] ALS - RMSE: {als_rmse:.4f}, MAE: {als_mae:.4f}, R²: {als_r2:.4f}")
print(f"[METRICS] RF - RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")
print(f"[METRICS] Ensemble - RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}, R²: {ensemble_r2:.4f}")

# Calculate accuracy within 1 point for ensemble
ensemble_with_error = ensemble_predictions.withColumn(
    "abs_error", 
    spark_abs(col("rating") - col("ensemble_prediction"))
)

within_1 = ensemble_with_error.filter(col("abs_error") <= 1.0).count()
total_predictions = ensemble_with_error.count()
accuracy_within_1 = (within_1 / total_predictions) * 100

within_05 = ensemble_with_error.filter(col("abs_error") <= 0.5).count()
accuracy_within_05 = (within_05 / total_predictions) * 100

print(f"[METRICS] Ensemble Accuracy within 1 point: {accuracy_within_1:.2f}%")
print(f"[METRICS] Ensemble Accuracy within 0.5 points: {accuracy_within_05:.2f}%")

# Generate enhanced recommendations
print("[RECS] Generating enhanced recommendations...")
all_users = final_data.select("userIndex").distinct()
recommendations = als_model.recommendForUserSubset(all_users, 15)  # Get more recommendations

# Create mappings
user_mapping = ratings_indexed.select("userIndex", "userId").distinct()
product_mapping = ratings_indexed.select("productIndex", "productId").distinct()

# Flatten and enhance recommendations
from pyspark.sql.functions import explode
recs_flat = recommendations.select(
    col("userIndex"),
    explode(col("recommendations")).alias("rec")
).select(
    col("userIndex"),
    col("rec.productIndex").alias("productIndex"),
    col("rec.rating").alias("predictedRating")
)

# Add diversity scoring (prefer items with different characteristics)
final_recs = recs_flat \
    .join(user_mapping, "userIndex") \
    .join(product_mapping, "productIndex") \
    .join(item_features.select("productId", "item_avg_rating", "item_rating_count"), "productId", "left") \
    .withColumn(
        "diversity_score",
        col("predictedRating") + (col("item_rating_count") / 1000.0)  # Slight boost for popular items
    ) \
    .select("userId", "productId", "predictedRating", "diversity_score") \
    .orderBy("userId", col("diversity_score").desc())

# Save results
print("[SAVE] Saving advanced results...")
try:
    base_dir = "data/processed/recommendations"
    os.makedirs(base_dir, exist_ok=True)
    
    final_recs.select("userId", "productId", "predictedRating") \
        .coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(f"{base_dir}/advanced_als_recommendations")
    
    rec_count = final_recs.count()
    print(f"[SUCCESS] Saved {rec_count:,} enhanced recommendations")
    
    # Comprehensive metrics
    metrics = {
        "model_type": "Advanced_Ensemble_ALS",
        "total_ratings": int(final_count),
        "train_ratings": int(train_count),
        "test_ratings": int(test_count),
        "als_rmse": float(als_rmse),
        "als_mae": float(als_mae),
        "als_r2": float(als_r2),
        "rf_rmse": float(rf_rmse),
        "rf_mae": float(rf_mae),
        "rf_r2": float(rf_r2),
        "ensemble_rmse": float(ensemble_rmse),
        "ensemble_mae": float(ensemble_mae),
        "ensemble_r2": float(ensemble_r2),
        "accuracy_within_1_point": float(accuracy_within_1),
        "accuracy_within_05_point": float(accuracy_within_05),
        "recommendations_generated": int(rec_count),
        "feature_engineering": "advanced",
        "model_approach": "ensemble"
    }
    
    with open(f"{base_dir}/advanced_als_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("[SUCCESS] Advanced metrics saved")
    print("\n[SUMMARY] Advanced improvements:")
    print(f"  - Feature engineering with user/item statistics")
    print(f"  - Ensemble of ALS + Random Forest")
    print(f"  - Enhanced filtering and diversity scoring")
    print(f"  - Target: RMSE < 1.2, Accuracy > 70%")
    
except Exception as e:
    print(f"[ERROR] Error saving: {e}")
print("[SAVE] Saving all models for Kafka integration...")

# Create models directory
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

try:
    # 1. Save Spark ALS Model (Spark native format)
    als_model_path = f"{models_dir}/als_model"
    als_model.write().overwrite().save(als_model_path)
    print(f"[SUCCESS] ALS model saved to {als_model_path}")
    
    # 2. Save Random Forest Pipeline (Spark native format)
    rf_model_path = f"{models_dir}/rf_pipeline_model"
    rf_model.write().overwrite().save(rf_model_path)
    print(f"[SUCCESS] RF Pipeline model saved to {rf_model_path}")
    
    # 3. Save Index Models (for user/product mapping)
    user_model_path = f"{models_dir}/user_indexer_model"
    product_model_path = f"{models_dir}/product_indexer_model"
    
    user_model.write().overwrite().save(user_model_path)
    product_model.write().overwrite().save(product_model_path)
    print(f"[SUCCESS] Indexer models saved")
    
    # 4. Save Model Metadata and Ensemble Configuration
    model_metadata = {
        "model_type": "spark_ensemble",
        "als_weight": 0.7,
        "rf_weight": 0.3,
        "als_params": {
            "maxIter": 25,
            "regParam": 0.05,
            "rank": 75,
            "alpha": 1.0
        },
        "rf_params": {
            "numTrees": 50,
            "maxDepth": 10
        },
        "feature_columns": [
            "user_avg_rating", "user_rating_std", "user_rating_count",
            "item_avg_rating", "item_rating_std", "item_rating_count",
            "user_item_rating_diff", "rating_vs_item_avg_diff", "user_popularity_score"
        ],
        "performance": {
            "ensemble_rmse": float(ensemble_rmse),
            "ensemble_mae": float(ensemble_mae),
            "ensemble_r2": float(ensemble_r2),
            "accuracy_within_1": float(accuracy_within_1)
        }
    }
    
    with open(f"{models_dir}/model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=4)
    print(f"[SUCCESS] Model metadata saved")
    
    # 5. Save User and Item Features for Real-time Inference
    print("[SAVE] Saving feature datasets...")
    
    # Save user features
    user_features.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"models/user_features")
    
    # Save item features  
    item_features.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"models/item_features")
    
    # Save mappings
    user_mapping.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"models/user_mapping")
    product_mapping.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"models/product_mapping")
    
    print("[SUCCESS] All models and features saved successfully!")
    print("\nSaved components:")
    print("  [OK] ALS Model (Spark format)")
    print("  [OK] RF Pipeline Model (Spark format)")
    print("  [OK] User/Product Indexers")
    print("  [OK] Model metadata and ensemble config")
    print("  [OK] User/Item features for real-time inference")
    print("  [OK] User/Product ID mappings")
    
except Exception as e:
    print(f"[ERROR] Error saving models: {e}")

# Model Loading Class for Kafka Consumer
model_loader_code = '''
import os
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel

class SparkModelLoader:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.spark = None
        self.als_model = None
        self.rf_model = None
        self.user_indexer = None
        self.product_indexer = None
        self.metadata = None
        self.user_features = None
        self.item_features = None
        self.user_mapping = None
        self.product_mapping = None
        
    def initialize_spark(self):
        """Initialize Spark session for model loading"""
        self.spark = SparkSession.builder \\
            .appName("RecommendationInference") \\
            .config("spark.sql.adaptive.enabled", "true") \\
            .getOrCreate()
    
    def load_models(self):
        """Load all saved models"""
        try:
            if self.spark is None:
                self.initialize_spark()
            
            # Load ALS model
            self.als_model = ALSModel.load(f"{self.models_dir}/als_model")
            print("[OK] ALS model loaded")
            
            # Load RF pipeline
            self.rf_model = PipelineModel.load(f"{self.models_dir}/rf_pipeline_model")
            print("[OK] RF pipeline loaded")
            
            # Load indexers
            self.user_indexer = StringIndexerModel.load(f"{self.models_dir}/user_indexer_model")
            self.product_indexer = StringIndexerModel.load(f"{self.models_dir}/product_indexer_model")
            print("[OK] Indexers loaded")
            
            # Load metadata
            with open(f"{self.models_dir}/model_metadata.json", "r") as f:
                self.metadata = json.load(f)
            print("[OK] Metadata loaded")
            
            # Load feature datasets
            self.user_features = self.spark.read.csv(f"{self.models_dir}/user_features", header=True, inferSchema=True)
            self.item_features = self.spark.read.csv(f"{self.models_dir}/item_features", header=True, inferSchema=True)
            self.user_mapping = self.spark.read.csv(f"{self.models_dir}/user_mapping", header=True, inferSchema=True)
            self.product_mapping = self.spark.read.csv(f"{self.models_dir}/product_mapping", header=True, inferSchema=True)
            print("[OK] Feature datasets loaded")
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_recommendations(self, user_id, num_recommendations=10):
        """Generate recommendations for a user"""
        try:
            # Convert user_id to index
            user_df = self.spark.createDataFrame([(user_id,)], ["userId"])
            user_indexed = self.user_indexer.transform(user_df)
            user_index = user_indexed.collect()[0]["userIndex"]
            
            # Generate ALS recommendations
            user_subset = self.spark.createDataFrame([(user_index,)], ["userIndex"])
            als_recs = self.als_model.recommendForUserSubset(user_subset, num_recommendations)
            
            # Convert back to original IDs and return
            recommendations = []
            for row in als_recs.collect():
                for rec in row["recommendations"]:
                    product_index = rec["productIndex"]
                    rating = rec["rating"]
                    
                    # Get original product ID
                    product_df = self.spark.createDataFrame([(product_index,)], ["productIndex"])
                    product_original = product_df.join(self.product_mapping, "productIndex").collect()[0]["productId"]
                    
                    recommendations.append({
                        "product_id": product_original,
                        "predicted_rating": float(rating)
                    })
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []
    
    def cleanup(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
'''

# Save the model loader class
with open(f"{models_dir}/spark_model_loader.py", "w", encoding='utf-8') as f:
    f.write(model_loader_code)

print(f"[SUCCESS] Model loader class saved to {models_dir}/spark_model_loader.py")
spark.stop()