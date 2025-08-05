import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
import os
import pickle
import json
import warnings
import gc
import pickle
from tqdm import tqdm
warnings.filterwarnings('ignore')

class MemoryEfficientContentBasedRecommender:
    def __init__(self, input_path, output_dir, chunk_size=1000):
        """
        Initialize the Memory-Efficient Content-Based Recommender
        
        Args:
            input_path: Path to the input CSV file
            output_dir: Directory to save recommendations
            chunk_size: Number of products to process at once for similarity computation
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.df = None
        self.product_features = None
        self.product_indices = None
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        print("Loading full dataset...")
        self.df = pd.read_csv(self.input_path)
        
        # Display basic info about the dataset
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Handle missing values
        self.df['title'] = self.df['title'].fillna('')
        self.df['summary'] = self.df['summary'].fillna('')
        self.df['text'] = self.df['text'].fillna('')
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df['score'] = pd.to_numeric(self.df['score'], errors='coerce').fillna(0)
        
        # Create combined text features
        self.df['combined_text'] = (
            self.df['title'] + ' ' + 
            self.df['summary'] + ' ' + 
            self.df['text']
        ).str.lower()
        
        # Extract product categories from title (simple heuristic)
        self.df['category'] = self.df['title'].str.extract(r'(\w+)')
        self.df['category'] = self.df['category'].fillna('Unknown')
        
        print("Data preprocessing completed!")
        return self.df
    
    def create_product_profiles(self):
        """Create aggregated product profiles from user reviews"""
        print("Creating product profiles...")
        
        # Group by product and aggregate features
        product_profiles = self.df.groupby('productId').agg({
            'title': 'first',
            'price': 'first',
            'score': 'mean',  # Average rating
            'helpfulness': 'mean',
            'combined_text': lambda x: ' '.join(x),  # Combine all review texts
            'category': 'first'
        }).reset_index()
        
        # Calculate review count for each product
        review_counts = self.df.groupby('productId').size().reset_index(name='review_count')
        product_profiles = product_profiles.merge(review_counts, on='productId')
        
        print(f"Created profiles for {len(product_profiles)} unique products")
        return product_profiles
    
    def extract_text_features(self, product_profiles):
        """Extract TF-IDF features from combined text"""
        print("Extracting text features using TF-IDF...")
        
        # Initialize TF-IDF vectorizer (reduced features for memory efficiency)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,  # Reduced to save memory
            stop_words='english',
            ngram_range=(1, 1),  # Only unigrams for memory efficiency
            min_df=2,
            max_df=0.8
        )
        
        # Fit and transform the combined text
        text_features = self.tfidf_vectorizer.fit_transform(product_profiles['combined_text'])
        
        # Reduce dimensionality for better performance and memory usage
        svd = TruncatedSVD(n_components=150, random_state=42)  # Reduced components
        text_features_reduced = svd.fit_transform(text_features)
        
        print(f"Text features shape: {text_features_reduced.shape}")
        print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
        return text_features_reduced
    
    def extract_numerical_features(self, product_profiles):
        """Extract and normalize numerical features"""
        print("Extracting numerical features...")
        
        numerical_features = []
        
        # Price feature (handle missing values)
        price_feature = product_profiles['price'].fillna(product_profiles['price'].median())
        numerical_features.append(price_feature.values.reshape(-1, 1))
        
        # Rating feature
        rating_feature = product_profiles['score'].values.reshape(-1, 1)
        numerical_features.append(rating_feature)
        
        # Helpfulness feature
        helpfulness_feature = product_profiles['helpfulness'].values.reshape(-1, 1)
        numerical_features.append(helpfulness_feature)
        
        # Review count feature (log-transformed)
        review_count_feature = np.log1p(product_profiles['review_count'].values).reshape(-1, 1)
        numerical_features.append(review_count_feature)
        
        # Combine all numerical features
        numerical_matrix = np.hstack(numerical_features)
        
        # Normalize features
        numerical_matrix_scaled = self.scaler.fit_transform(numerical_matrix)
        
        print(f"Numerical features shape: {numerical_matrix_scaled.shape}")
        return numerical_matrix_scaled
    
    def extract_categorical_features(self, product_profiles):
        """Extract categorical features"""
        print("Extracting categorical features...")
        
        # One-hot encode categories (limit to top categories to save memory)
        top_categories = product_profiles['category'].value_counts().head(50).index
        product_profiles['category_filtered'] = product_profiles['category'].apply(
            lambda x: x if x in top_categories else 'Other'
        )
        
        categories = pd.get_dummies(product_profiles['category_filtered'], prefix='category')
        
        print(f"Categorical features shape: {categories.shape}")
        return categories.values
    
    def build_product_features(self):
        """Build the product feature matrix"""
        print("Building product feature matrix...")
        
        # Load and preprocess data
        self.load_and_preprocess_data()
          
        # Create product profiles
        product_profiles = self.create_product_profiles()
        
        # Extract different types of features
        text_features = self.extract_text_features(product_profiles)
        numerical_features = self.extract_numerical_features(product_profiles)
        categorical_features = self.extract_categorical_features(product_profiles)
        
        # Combine all features with different weights
        text_weight = 0.6
        numerical_weight = 0.3
        categorical_weight = 0.1
        
        # Normalize feature matrices to have similar scales
        text_features_norm = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        numerical_features_norm = numerical_features / np.linalg.norm(numerical_features, axis=1, keepdims=True)
        categorical_features_norm = categorical_features / np.linalg.norm(categorical_features, axis=1, keepdims=True)
        
        # Handle any NaN values from normalization
        text_features_norm = np.nan_to_num(text_features_norm)
        numerical_features_norm = np.nan_to_num(numerical_features_norm)
        categorical_features_norm = np.nan_to_num(categorical_features_norm)
        
        # Combine features with weights
        self.product_features = np.hstack([
            text_features_norm * text_weight,
            numerical_features_norm * numerical_weight,
            categorical_features_norm * categorical_weight
        ])
        
        # Create product index mapping
        self.product_indices = pd.Series(
            index=product_profiles['productId'],
            data=range(len(product_profiles))
        )
        
        # Save product profiles and features for reference
        product_profiles.to_csv(
            os.path.join(self.output_dir, 'product_profiles.csv'),
            index=False
        )
        
        # Save product features to disk to avoid recomputation
        with open(os.path.join(self.output_dir, 'product_features.pkl'), 'wb') as f:
            pickle.dump(self.product_features, f)
        
        with open(os.path.join(self.output_dir, 'product_indices.pkl'), 'wb') as f:
            pickle.dump(self.product_indices, f)
        
        print(f"Product features shape: {self.product_features.shape}")
        print("Product features built and saved successfully!")
    
    def get_recommendations_for_product(self, product_idx, n_recommendations=10):
        """Get recommendations for a single product using chunked similarity computation"""
        similarities = []
        
        # Get the feature vector for the target product
        target_feature = self.product_features[product_idx:product_idx+1]
        
        # Compute similarities in chunks to save memory
        num_products = self.product_features.shape[0]
        
        for start_idx in range(0, num_products, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_products)
            chunk_features = self.product_features[start_idx:end_idx]
            
            # Compute cosine similarity for this chunk
            chunk_similarities = cosine_similarity(target_feature, chunk_features).flatten()
            similarities.extend([(start_idx + i, sim) for i, sim in enumerate(chunk_similarities)])
        
        # Sort by similarity (excluding the product itself)
        similarities = [(idx, sim) for idx, sim in similarities if idx != product_idx]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recommendations = similarities[:n_recommendations]
        
        # Convert indices back to product IDs
        product_ids = list(self.product_indices.index)
        recommendations = [
            {
                'product_id': product_ids[idx],
                'similarity_score': float(sim)
            }
            for idx, sim in top_recommendations
        ]
        
        return recommendations
    
    def generate_all_recommendations(self, n_recommendations=10):
        """Generate recommendations for all products using memory-efficient approach"""
        print(f"Generating recommendations for all {len(self.product_indices)} products...")
        print(f"Using chunk size: {self.chunk_size} for memory efficiency")
        
        recommendations_list = []
        product_ids = list(self.product_indices.index)
        total_products = len(product_ids)
        
        # Use tqdm for better progress tracking
        for i, product_id in enumerate(tqdm(product_ids, desc="Processing products")):
            product_idx = self.product_indices[product_id]
            recommendations = self.get_recommendations_for_product(product_idx, n_recommendations)
            
            # Convert to list format for CSV
            for rank, rec in enumerate(recommendations, 1):
                recommendations_list.append({
                    'source_product_id': product_id,
                    'recommended_product_id': rec['product_id'],
                    'similarity_score': rec['similarity_score'],
                    'rank': rank
                })
            
            # Periodic garbage collection to free memory
            if i % 1000 == 0:
                gc.collect()
        
        # Convert to DataFrame and save as CSV
        recommendations_df = pd.DataFrame(recommendations_list)
        output_file = os.path.join(self.output_dir, 'content_based_recommendations.csv')
        recommendations_df.to_csv(output_file, index=False)
        
        print(f"Recommendations saved to {output_file}")
        
        # Create detailed summary
        self.create_recommendations_summary(recommendations_df)
        
        return recommendations_df
    
    def create_recommendations_summary(self, recommendations_df):
        """Create a detailed summary CSV with product information"""
        print("Creating detailed recommendations summary...")
        
        # Load product profiles
        product_profiles = pd.read_csv(os.path.join(self.output_dir, 'product_profiles.csv'))
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        summary_chunks = []
        
        for i in range(0, len(recommendations_df), chunk_size):
            chunk = recommendations_df.iloc[i:i+chunk_size]
            
            # Merge with source product details
            chunk_summary = chunk.merge(
                product_profiles, 
                left_on='source_product_id', 
                right_on='productId', 
                how='left',
                suffixes=('', '_source')
            )
            
            # Merge with recommended product details
            chunk_summary = chunk_summary.merge(
                product_profiles, 
                left_on='recommended_product_id', 
                right_on='productId', 
                how='left',
                suffixes=('_source', '_recommended')
            )
            
            summary_chunks.append(chunk_summary)
        
        # Combine all chunks
        summary_df = pd.concat(summary_chunks, ignore_index=True)
        
        # Select relevant columns for the summary
        summary_columns = [
            'source_product_id',
            'title_source',
            'category_source',
            'score_source',
            'price_source',
            'recommended_product_id',
            'title_recommended',
            'category_recommended',
            'score_recommended',
            'price_recommended',
            'similarity_score',
            'rank'
        ]
        
        # Create final summary
        final_summary = summary_df[summary_columns].copy()
        final_summary.columns = [
            'source_product_id',
            'source_title',
            'source_category',
            'source_rating',
            'source_price',
            'recommended_product_id',
            'recommended_title',
            'recommended_category',
            'recommended_rating',
            'recommended_price',
            'similarity_score',
            'rank'
        ]
        
        # Save detailed summary
        summary_file = os.path.join(self.output_dir, 'recommendations_detailed.csv')
        final_summary.to_csv(summary_file, index=False)
        
        print(f"Detailed recommendations saved to {summary_file}")
        return final_summary
    
    def evaluate_recommendations(self, sample_size=200):
        """Evaluate the quality of recommendations"""
        print(f"Evaluating recommendation quality on {sample_size} products...")
        
        product_ids = list(self.product_indices.index)
        sample_products = np.random.choice(product_ids, min(sample_size, len(product_ids)), replace=False)
        
        # Load product profiles for evaluation
        product_profiles = pd.read_csv(os.path.join(self.output_dir, 'product_profiles.csv'))
        
        evaluation_results = {
            'diversity_scores': [],
            'avg_similarity_scores': [],
            'category_coverage': []
        }
        
        for i, product_id in enumerate(tqdm(sample_products, desc="Evaluating")):
            product_idx = self.product_indices[product_id]
            recommendations = self.get_recommendations_for_product(product_idx, 10)
            
            if not recommendations:
                continue
            
            # Calculate average similarity score
            avg_similarity = np.mean([rec['similarity_score'] for rec in recommendations])
            evaluation_results['avg_similarity_scores'].append(avg_similarity)
            
            # Calculate category coverage
            rec_product_ids = [rec['product_id'] for rec in recommendations]
            rec_products = product_profiles[product_profiles['productId'].isin(rec_product_ids)]
            unique_categories = rec_products['category'].nunique()
            evaluation_results['category_coverage'].append(unique_categories)
        
        # Calculate final metrics
        metrics = {
            'avg_similarity': np.mean(evaluation_results['avg_similarity_scores']),
            'avg_category_coverage': np.mean(evaluation_results['category_coverage']),
            'total_products_evaluated': len(sample_products)
        }
        
        # Save evaluation results as CSV
        eval_df = pd.DataFrame([metrics])
        eval_file = os.path.join(self.output_dir, 'evaluation_metrics.csv')
        eval_df.to_csv(eval_file, index=False)
        
        print("Evaluation Results:")
        print(f"Average Similarity Score: {metrics['avg_similarity']:.4f}")
        print(f"Average Category Coverage: {metrics['avg_category_coverage']:.2f}")
        
        return metrics
    
    def display_sample_recommendations(self, n_samples=3):
        """Display sample recommendations for inspection"""
        print("\nSample Recommendations:")
        print("=" * 50)
        
        # Load product profiles
        product_profiles = pd.read_csv(os.path.join(self.output_dir, 'product_profiles.csv'))
        
        # Get random sample of products
        sample_products = product_profiles.sample(n_samples)
        
        for _, product in sample_products.iterrows():
            print(f"\nProduct: {product['title']}")
            print(f"Category: {product['category']}")
            print(f"Average Rating: {product['score']:.2f}")
            print(f"Review Count: {product['review_count']}")
            
            product_idx = self.product_indices[product['productId']]
            recommendations = self.get_recommendations_for_product(product_idx, 5)
            
            print("Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                rec_product = product_profiles[product_profiles['productId'] == rec['product_id']].iloc[0]
                print(f"  {i}. {rec_product['title']} (Similarity: {rec['similarity_score']:.3f})")
            
            print("-" * 50)

def main():
    """Main function to run the memory-efficient content-based recommender"""
    import sys
    
    # Configuration
    input_path = "data/processed/Electronics.csv"
    
    # Check if output directory is provided as command line argument
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        print(f"Using custom output directory: {output_dir}")
    else:
        # Default output directory
        output_dir = "data/processed/content_based_recommendations"
        print(f"Using default output directory: {output_dir}")
    
    # Check if chunk size is provided as second argument
    chunk_size = 1000  # Default chunk size
    if len(sys.argv) > 2:
        try:
            chunk_size = int(sys.argv[2])
            print(f"Using custom chunk size: {chunk_size}")
        except ValueError:
            print("Invalid chunk size provided, using default: 1000")
    
    print("Memory-Efficient Content-Based Filtering Recommendation System")
    print("=" * 65)
    print(f"Processing entire dataset with chunk size: {chunk_size}")
    print("This approach uses less memory but takes longer to compute")
    
    # Initialize recommender
    recommender = MemoryEfficientContentBasedRecommender(input_path, output_dir, chunk_size)
    
    try:
        # Build product features (no full similarity matrix)
        recommender.build_product_features()
        
        # Generate recommendations for all products
        recommendations_df = recommender.generate_all_recommendations(n_recommendations=10)
        
        # Evaluate recommendations
        metrics = recommender.evaluate_recommendations(sample_size=200)
        
        # Display sample recommendations
        recommender.display_sample_recommendations(n_samples=3)
        
        print("\n" + "=" * 65)
        print("Memory-efficient content-based filtering completed successfully!")
        print(f"Total recommendation pairs generated: {len(recommendations_df)}")
        print(f"Recommendations saved to: {output_dir}")
        print("\nOutput files created:")
        print("- content_based_recommendations.csv (main recommendations)")
        print("- recommendations_detailed.csv (with product details)")
        print("- product_profiles.csv (aggregated product data)")
        print("- product_features.pkl (feature matrix)")
        print("- product_indices.pkl (product index mapping)")
        print("- evaluation_metrics.csv (quality metrics)")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file at {input_path}")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data and try again.")

    with open(os.path.join(output_dir, 'models/cb_recommender_model.pkl'), 'wb') as f:
        pickle.dump(recommender, f)


if __name__ == "__main__":
    main()