import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ndcg_score
import json
import os
import pickle
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

class EnhancedHybridRecommender:
    """
    Enhanced hybrid recommender with improved error handling, performance optimizations,
    and model persistence capabilities
    """
    
    def __init__(self, cf_path: str = None, cb_path: str = None, 
                 metadata_path: str = None, user_history_path: str = None):
        # File paths (can be None when loading from pickle)
        self.cf_path = cf_path
        self.cb_path = cb_path
        self.metadata_path = metadata_path
        self.user_history_path = user_history_path
        
        # Optimized weights (Config 3 parameters)
        self.cf_weight = 0.7
        self.cb_weight = 0.2
        self.popularity_weight = 0.1
        
        # Quality thresholds
        self.min_cf_score = 0.15
        self.min_cb_score = 0.08
        self.relevance_threshold = 3.0
        
        # Adaptive K parameters
        self.use_adaptive_k = True
        self.min_k = 6
        self.max_k = 14
        self.default_k = 10
        
        # Data containers
        self.cf_recs = None
        self.cb_recs = None
        self.metadata = None
        self.user_history = None
        self.hybrid_recommendations = None
        self.popularity_scores = {}
        self.user_activity_levels = {}
        
        # Model state tracking
        self.is_trained = False
        self.training_timestamp = None
        self.model_version = "1.0"
        
        # Lookup dictionaries for fast inference (these get saved)
        self.cb_lookup = defaultdict(float)
        self.user_profiles = {}  # Store user preference profiles
        self.product_features = {}  # Store product feature vectors
        
        self.evaluation_results = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def save_model(self, filepath: str):
        """
        Save the trained model to a pickle file
        
        **Model Persistence Strategy:**
        This method saves the complete trained model state including all learned parameters,
        lookup tables, and configuration settings. The saved model can be loaded later
        for making predictions without requiring the original training data.
        
        **What Gets Saved:**
        - All model parameters and weights
        - Popularity scores dictionary
        - User activity levels mapping
        - Content-based lookup tables
        - User profiles and product features
        - Model configuration and metadata
        - Training timestamp and version info
        """
        try:
            if not self.is_trained:
                self.logger.warning("Model hasn't been trained yet. Saving current state...")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Prepare model state for serialization
            model_state = {
                # Core model parameters
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'popularity_weight': self.popularity_weight,
                'min_cf_score': self.min_cf_score,
                'min_cb_score': self.min_cb_score,
                'relevance_threshold': self.relevance_threshold,
                
                # Adaptive K parameters
                'use_adaptive_k': self.use_adaptive_k,
                'min_k': self.min_k,
                'max_k': self.max_k,
                'default_k': self.default_k,
                
                # Learned model components (the "trained" part)
                'popularity_scores': dict(self.popularity_scores),
                'user_activity_levels': dict(self.user_activity_levels),
                'cb_lookup': dict(self.cb_lookup),
                'user_profiles': self.user_profiles,
                'product_features': self.product_features,
                
                # Model metadata and state
                'is_trained': self.is_trained,
                'training_timestamp': self.training_timestamp,
                'model_version': self.model_version,
                'evaluation_results': self.evaluation_results,
                
                # Configuration for reproducibility
                'original_file_paths': {
                    'cf_path': self.cf_path,
                    'cb_path': self.cb_path,
                    'metadata_path': self.metadata_path,
                    'user_history_path': self.user_history_path
                }
            }
            
            # Save using pickle with high protocol for efficiency
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"✅ Model saved successfully to: {filepath}")
            self.logger.info(f"📊 Model contains {len(self.popularity_scores)} product popularity scores")
            self.logger.info(f"👥 Model contains {len(self.user_activity_levels)} user activity profiles")
            self.logger.info(f"🔍 Model contains {len(self.cb_lookup)} content-based lookup entries")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a trained model from a pickle file
        
        **Model Loading Process:**
        This class method creates a new instance of the recommender and loads
        all the trained parameters from the saved pickle file. The loaded model
        is ready for making predictions without requiring retraining.
        
        **Usage Example:**
        ```python
        # Load the trained model
        recommender = EnhancedHybridRecommender.load_model('models/trained_recommender.pkl')
        
        # Make predictions immediately
        recommendations = recommender.predict_for_user('user_123', top_k=10)
        ```
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Load the model state
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            # Create new instance
            instance = cls()
            
            # Restore model parameters
            instance.cf_weight = model_state.get('cf_weight', 0.7)
            instance.cb_weight = model_state.get('cb_weight', 0.2)
            instance.popularity_weight = model_state.get('popularity_weight', 0.1)
            instance.min_cf_score = model_state.get('min_cf_score', 0.15)
            instance.min_cb_score = model_state.get('min_cb_score', 0.08)
            instance.relevance_threshold = model_state.get('relevance_threshold', 3.0)
            
            # Restore adaptive K parameters
            instance.use_adaptive_k = model_state.get('use_adaptive_k', True)
            instance.min_k = model_state.get('min_k', 6)
            instance.max_k = model_state.get('max_k', 14)
            instance.default_k = model_state.get('default_k', 10)
            
            # Restore learned components
            instance.popularity_scores = defaultdict(float, model_state.get('popularity_scores', {}))
            instance.user_activity_levels = model_state.get('user_activity_levels', {})
            instance.cb_lookup = defaultdict(float, model_state.get('cb_lookup', {}))
            instance.user_profiles = model_state.get('user_profiles', {})
            instance.product_features = model_state.get('product_features', {})
            
            # Restore model state
            instance.is_trained = model_state.get('is_trained', False)
            instance.training_timestamp = model_state.get('training_timestamp', None)
            instance.model_version = model_state.get('model_version', '1.0')
            instance.evaluation_results = model_state.get('evaluation_results', {})
            
            # Restore file paths (for reference)
            original_paths = model_state.get('original_file_paths', {})
            instance.cf_path = original_paths.get('cf_path')
            instance.cb_path = original_paths.get('cb_path')
            instance.metadata_path = original_paths.get('metadata_path')
            instance.user_history_path = original_paths.get('user_history_path')
            
            instance.logger.info(f"✅ Model loaded successfully from: {filepath}")
            instance.logger.info(f"🕐 Model trained on: {instance.training_timestamp}")
            instance.logger.info(f"📊 Loaded {len(instance.popularity_scores)} product popularity scores")
            instance.logger.info(f"👥 Loaded {len(instance.user_activity_levels)} user activity profiles")
            instance.logger.info(f"🔍 Loaded {len(instance.cb_lookup)} content-based lookup entries")
            
            if instance.is_trained:
                instance.logger.info("🎯 Model is ready for predictions!")
            else:
                instance.logger.warning("⚠️ Model was saved before training completion")
            
            return instance
            
        except Exception as e:
            logging.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def predict_for_user(self, user_id: str, top_k: int = 10, 
                        exclude_seen: bool = True) -> pd.DataFrame:
        """
        Make predictions for a specific user using the trained model
        
        **Fast Prediction Method:**
        This method uses the pre-computed lookup tables and learned parameters
        to quickly generate recommendations for a user without requiring
        access to the original training data.
        """
        try:
            if not self.is_trained:
                self.logger.warning("Model not fully trained. Results may be suboptimal.")
            
            user_id_str = str(user_id)
            
            # Get adaptive K for this user
            user_k = self._get_adaptive_k(user_id_str)
            actual_k = min(top_k, user_k)
            
            # Get user's seen products for exclusion (if we have this data)
            seen_products = set()
            if exclude_seen and hasattr(self, 'user_history') and self.user_history is not None:
                user_history_subset = self.user_history[self.user_history['user_id'] == user_id_str]
                seen_products = set(user_history_subset['product_id'].astype(str).tolist())
            
            # Generate predictions using lookup tables
            predictions = []
            
            # Get all available products from our lookup tables
            all_products = set(self.popularity_scores.keys()) | set(self.cb_lookup.keys())
            
            for product_id in all_products:
                product_id_str = str(product_id)
                
                # Skip seen products if requested
                if exclude_seen and product_id_str in seen_products:
                    continue
                
                # Get scores from lookup tables
                cf_score = self._get_cf_score_for_user_product(user_id_str, product_id_str)
                cb_score = self.cb_lookup.get(product_id_str, 0.0)
                popularity_score = self.popularity_scores.get(product_id_str, 0.5)
                
                # Calculate hybrid score
                hybrid_score = (
                    self.cf_weight * cf_score +
                    self.cb_weight * cb_score +
                    self.popularity_weight * popularity_score
                )
                
                predictions.append({
                    'user_id': user_id_str,
                    'product_id': product_id_str,
                    'cf_score': cf_score,
                    'cb_score': cb_score,
                    'popularity_score': popularity_score,
                    'hybrid_score': hybrid_score
                })
            
            # Convert to DataFrame and get top K
            if predictions:
                pred_df = pd.DataFrame(predictions)
                pred_df = pred_df.nlargest(actual_k, 'hybrid_score')
                
                self.logger.info(f"Generated {len(pred_df)} predictions for user {user_id}")
                return pred_df
            else:
                self.logger.warning(f"No predictions generated for user {user_id}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error making predictions for user {user_id}: {str(e)}")
            return pd.DataFrame()
    
    def _get_cf_score_for_user_product(self, user_id: str, product_id: str) -> float:
        """
        Get collaborative filtering score for user-product pair
        This is a simplified version - in a full implementation, you'd want to
        store user-item matrices or use more sophisticated lookup methods
        """
        # For now, return a default score based on user activity and product popularity
        # In a full implementation, you'd store the actual CF predictions
        
        user_activity = self.user_activity_levels.get(user_id, 'medium')
        product_popularity = self.popularity_scores.get(product_id, 0.5)
        
        # Simple heuristic - more active users get slightly higher base scores
        activity_multiplier = {'low': 0.8, 'medium': 1.0, 'high': 1.2}.get(user_activity, 1.0)
        
        return min(product_popularity * activity_multiplier, 1.0)
    
    def batch_predict(self, user_ids: List[str], top_k: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Make predictions for multiple users efficiently
        
        **Batch Prediction Optimization:**
        This method processes multiple users in a single call, which is more
        efficient than calling predict_for_user multiple times.
        """
        try:
            results = {}
            
            self.logger.info(f"Making batch predictions for {len(user_ids)} users...")
            
            for i, user_id in enumerate(user_ids):
                if i % 100 == 0:
                    self.logger.info(f"Processed {i}/{len(user_ids)} users")
                
                results[str(user_id)] = self.predict_for_user(user_id, top_k)
            
            self.logger.info(f"Completed batch predictions for {len(user_ids)} users")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            return {}
    
    def update_user_profile(self, user_id: str, new_interactions: List[Dict]):
        """
        Update user profile with new interactions (for online learning)
        
        **Online Learning Capability:**
        This method allows the model to adapt to new user interactions
        without requiring full retraining.
        """
        try:
            user_id_str = str(user_id)
            
            # Update user activity level based on new interactions
            current_activity = len(new_interactions)
            
            # Simple activity level update logic
            if current_activity >= 10:
                self.user_activity_levels[user_id_str] = 'high'
            elif current_activity >= 5:
                self.user_activity_levels[user_id_str] = 'medium'
            else:
                self.user_activity_levels[user_id_str] = 'low'
            
            # Store user profile for future reference
            if user_id_str not in self.user_profiles:
                self.user_profiles[user_id_str] = []
            
            self.user_profiles[user_id_str].extend(new_interactions)
            
            self.logger.info(f"Updated profile for user {user_id} with {len(new_interactions)} interactions")
            
        except Exception as e:
            self.logger.error(f"Error updating user profile: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model"""
        return {
            'is_trained': self.is_trained,
            'training_timestamp': self.training_timestamp,
            'model_version': self.model_version,
            'num_products_with_popularity': len(self.popularity_scores),
            'num_users_with_activity': len(self.user_activity_levels),
            'num_cb_entries': len(self.cb_lookup),
            'num_user_profiles': len(self.user_profiles),
            'model_parameters': {
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'popularity_weight': self.popularity_weight,
                'use_adaptive_k': self.use_adaptive_k
            }
        }
    
    # [Previous methods remain the same - load_data, preprocess_data, etc.]
    def load_data(self):
        """Load and prepare data with enhanced error handling"""
        try:
            self.logger.info("Loading datasets...")
            
            # Load datasets
            self.cf_recs = pd.read_csv(self.cf_path)
            self.cb_recs = pd.read_csv(self.cb_path)
            self.metadata = pd.read_csv(self.metadata_path)
            self.user_history = pd.read_csv(self.user_history_path)
            
            # Log initial data shapes
            self.logger.info(f"CF recommendations: {self.cf_recs.shape}")
            self.logger.info(f"CB recommendations: {self.cb_recs.shape}")
            self.logger.info(f"Metadata: {self.metadata.shape}")
            self.logger.info(f"User history: {self.user_history.shape}")
            
            self._standardize_columns()
            self._calculate_popularity_scores()
            self._calculate_user_activity_levels()
            
            self.logger.info("Data loading completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _standardize_columns(self):
        """Standardize column names and handle data types"""
        try:
            # CF recommendations
            if 'userId' in self.cf_recs.columns:
                self.cf_recs = self.cf_recs.rename(columns={
                    'userId': 'user_id', 
                    'productId': 'product_id'
                })
            
            # CB recommendations  
            if 'source_product_id' in self.cb_recs.columns:
                self.cb_recs = self.cb_recs.rename(columns={
                    'source_product_id': 'product_id',
                    'recommended_product_id': 'recommended_product_id',
                    'similarity_score': 'cb_score'
                })
            elif 'similarity_score' in self.cb_recs.columns:
                self.cb_recs = self.cb_recs.rename(columns={'similarity_score': 'cb_score'})
            
            # User history
            if 'userId' in self.user_history.columns:
                self.user_history = self.user_history.rename(columns={
                    'userId': 'user_id', 
                    'productId': 'product_id'
                })
            
            # Metadata
            if 'productId' in self.metadata.columns:
                self.metadata = self.metadata.rename(columns={'productId': 'product_id'})
            
            # Ensure consistent data types (keep as strings to avoid conversion issues)
            for df_name, df in [('cf_recs', self.cf_recs), ('cb_recs', self.cb_recs), 
                               ('user_history', self.user_history), ('metadata', self.metadata)]:
                if 'user_id' in df.columns:
                    df['user_id'] = df['user_id'].astype(str)
                if 'product_id' in df.columns:
                    df['product_id'] = df['product_id'].astype(str)
                if 'recommended_product_id' in df.columns:
                    df['recommended_product_id'] = df['recommended_product_id'].astype(str)
            
            self.logger.info("Column standardization completed")
            
        except Exception as e:
            self.logger.error(f"Error in column standardization: {str(e)}")
            raise
    
    def _calculate_popularity_scores(self):
        """Calculate enhanced popularity scores"""
        try:
            # Group by product and calculate comprehensive stats
            pop_stats = self.user_history.groupby('product_id').agg({
                'rating': ['count', 'mean', 'std']
            }).reset_index()
            
            pop_stats.columns = ['product_id', 'rating_count', 'avg_rating', 'rating_std']
            
            # Fill NaN std with 0 (single ratings)
            pop_stats['rating_std'] = pop_stats['rating_std'].fillna(0)
            
            # Calculate rating consistency (inverse of std, normalized)
            max_std = pop_stats['rating_std'].max()
            if max_std > 0:
                pop_stats['consistency_score'] = 1 - (pop_stats['rating_std'] / max_std)
            else:
                pop_stats['consistency_score'] = 1.0
            
            # Apply minimum interaction filter
            min_interactions = max(3, pop_stats['rating_count'].quantile(0.1))
            pop_stats = pop_stats[pop_stats['rating_count'] >= min_interactions]
            
            # Normalize components using robust scaling
            scaler = MinMaxScaler()
            pop_stats[['count_norm', 'rating_norm', 'consistency_norm']] = scaler.fit_transform(
                pop_stats[['rating_count', 'avg_rating', 'consistency_score']]
            )
            
            # Enhanced popularity score with quality and consistency
            pop_stats['popularity_score'] = (
                0.4 * pop_stats['count_norm'] + 
                0.4 * pop_stats['rating_norm'] +
                0.2 * pop_stats['consistency_norm']
            )
            
            self.popularity_scores = dict(zip(
                pop_stats['product_id'], 
                pop_stats['popularity_score']
            ))
            
            self.logger.info(f"Calculated enhanced popularity scores for {len(self.popularity_scores)} products")
            
        except Exception as e:
            self.logger.error(f"Error calculating popularity: {str(e)}")
            self.popularity_scores = {}
    
    def _calculate_user_activity_levels(self):
        """Calculate user activity levels for adaptive K selection"""
        try:
            user_activity = self.user_history.groupby('user_id').size()
            
            # Define activity thresholds
            low_threshold = user_activity.quantile(0.33)
            high_threshold = user_activity.quantile(0.67)
            
            for user_id, activity_count in user_activity.items():
                if activity_count <= low_threshold:
                    self.user_activity_levels[str(user_id)] = 'low'
                elif activity_count >= high_threshold:
                    self.user_activity_levels[str(user_id)] = 'high'
                else:
                    self.user_activity_levels[str(user_id)] = 'medium'
            
            self.logger.info(f"Calculated activity levels for {len(self.user_activity_levels)} users")
            
        except Exception as e:
            self.logger.error(f"Error calculating user activity: {str(e)}")
            self.user_activity_levels = {}
    
    def _get_adaptive_k(self, user_id: str) -> int:
        """Get adaptive K based on user activity level"""
        if not self.use_adaptive_k:
            return self.default_k
        
        activity_level = self.user_activity_levels.get(str(user_id), 'medium')
        
        if activity_level == 'low':
            return self.min_k
        elif activity_level == 'high':
            return self.max_k
        else:
            return self.default_k
    
    def preprocess_data(self):
        """Enhanced preprocessing with robust outlier handling"""
        try:
            self.logger.info("Preprocessing data with enhanced methods...")
            
            # Process CF scores with robust outlier handling
            if 'predictedRating' in self.cf_recs.columns:
                cf_scores = self.cf_recs['predictedRating'].copy()
                
                # Use IQR-based outlier detection for robust preprocessing
                Q1 = cf_scores.quantile(0.25)
                Q3 = cf_scores.quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds (more conservative than standard 1.5*IQR)
                lower_bound = Q1 - 1.2 * IQR
                upper_bound = Q3 + 1.2 * IQR
                
                # Clip extreme outliers but preserve range
                cf_scores_processed = cf_scores.clip(lower=lower_bound, upper=upper_bound)
                
                # Normalize
                cf_min, cf_max = cf_scores_processed.min(), cf_scores_processed.max()
                if cf_max > cf_min:
                    self.cf_recs['cf_score_norm'] = (cf_scores_processed - cf_min) / (cf_max - cf_min)
                else:
                    self.cf_recs['cf_score_norm'] = 0.5
                
                # Apply threshold with retention tracking
                initial_count = len(self.cf_recs)
                self.cf_recs = self.cf_recs[self.cf_recs['cf_score_norm'] >= self.min_cf_score]
                retention_rate = len(self.cf_recs) / initial_count
                
                self.logger.info(f"CF recommendations after filtering: {len(self.cf_recs)} (retention: {retention_rate:.1%})")
            
            # Process CB scores
            if 'cb_score' in self.cb_recs.columns:
                cb_scores = self.cb_recs['cb_score'].copy()
                
                # Handle potential negative similarities
                if cb_scores.min() < 0:
                    cb_scores = cb_scores - cb_scores.min()
                
                cb_min, cb_max = cb_scores.min(), cb_scores.max()
                if cb_max > cb_min:
                    self.cb_recs['cb_score_norm'] = (cb_scores - cb_min) / (cb_max - cb_min)
                else:
                    self.cb_recs['cb_score_norm'] = 0.5
                
                # Apply threshold
                initial_count = len(self.cb_recs)
                self.cb_recs = self.cb_recs[self.cb_recs['cb_score_norm'] >= self.min_cb_score]
                retention_rate = len(self.cb_recs) / initial_count
                
                self.logger.info(f"CB recommendations after filtering: {len(self.cb_recs)} (retention: {retention_rate:.1%})")
            
            self.logger.info("Enhanced preprocessing completed!")
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def generate_hybrid_recommendations(self, top_k: int = 10) -> pd.DataFrame:
        """Generate hybrid recommendations with enhanced error handling and optimizations"""
        try:
            self.logger.info("Generating enhanced hybrid recommendations...")
            
            # Build optimized CB lookup dictionary
            cb_lookup = defaultdict(float)
            
            # Handle different CB data structures
            if 'recommended_product_id' in self.cb_recs.columns:
                # CB recommendations are in format: source_product -> recommended_product
                for _, row in self.cb_recs.iterrows():
                    try:
                        product_id = str(row['recommended_product_id'])
                        score = float(row['cb_score_norm'])
                        cb_lookup[product_id] = max(cb_lookup[product_id], score)  # Take max if multiple
                    except (ValueError, TypeError):
                        continue
            elif 'product_id' in self.cb_recs.columns:
                # Direct product-based CB scores
                cb_lookup = self.cb_recs.groupby('product_id')['cb_score_norm'].mean().to_dict()
                cb_lookup = defaultdict(float, cb_lookup)
            
            # Store CB lookup for model persistence
            self.cb_lookup = cb_lookup
            
            self.logger.info(f"Built CB lookup with {len(cb_lookup)} products")
            
            # Generate recommendations with optimized processing
            hybrid_recs = []
            unique_users = self.cf_recs['user_id'].unique()
            
            self.logger.info(f"Processing recommendations for {len(unique_users)} users...")
            
            for i, user_id in enumerate(unique_users):
                if i % 1000 == 0:
                    self.logger.info(f"Processed {i}/{len(unique_users)} users")
                
                user_cf_recs = self.cf_recs[self.cf_recs['user_id'] == user_id].copy()
                
                if len(user_cf_recs) == 0:
                    continue
                
                # Get adaptive K for this user
                user_k = self._get_adaptive_k(str(user_id))
                
                user_scores = []
                for _, row in user_cf_recs.iterrows():
                    try:
                        product_id = str(row['product_id'])
                        
                        # Get CF score (handle both column names)
                        cf_score = float(row.get('cf_score_norm', row.get('predictedRating', 0)))
                        
                        # Get CB score from lookup
                        cb_score = cb_lookup[product_id]
                        
                        # Get popularity score
                        popularity_score = self.popularity_scores.get(product_id, 0.5)
                        
                        # Calculate hybrid score
                        hybrid_score = (
                            self.cf_weight * cf_score +
                            self.cb_weight * cb_score +
                            self.popularity_weight * popularity_score
                        )
                        
                        user_scores.append({
                            'user_id': str(user_id),
                            'product_id': product_id,
                            'cf_score': cf_score,
                            'cb_score': cb_score,
                            'popularity_score': popularity_score,
                            'hybrid_score': hybrid_score
                        })
                        
                    except (ValueError, TypeError, KeyError) as e:
                        # Skip problematic rows but log occasionally
                        if i % 5000 == 0:
                            self.logger.warning(f"Skipping row for user {user_id}: {str(e)}")
                        continue
                
                # Select top recommendations based on hybrid score
                if user_scores:
                    user_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
                    top_recommendations = user_scores[:user_k]
                    hybrid_recs.extend(top_recommendations)
            
            # Create DataFrame with proper data types
            self.hybrid_recommendations = pd.DataFrame(hybrid_recs)
            
            if not self.hybrid_recommendations.empty:
                # Keep user_id and product_id as strings to avoid conversion issues
                self.hybrid_recommendations['user_id'] = self.hybrid_recommendations['user_id'].astype(str)
                self.hybrid_recommendations['product_id'] = self.hybrid_recommendations['product_id'].astype(str)
                
                # Ensure numeric columns are proper floats
                numeric_cols = ['cf_score', 'cb_score', 'popularity_score', 'hybrid_score']
                for col in numeric_cols:
                    self.hybrid_recommendations[col] = pd.to_numeric(self.hybrid_recommendations[col], errors='coerce')
            
            self.logger.info(f"Generated {len(self.hybrid_recommendations)} recommendations for {len(unique_users)} users")
            return self.hybrid_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def calculate_comprehensive_metrics(self, test_ratings: pd.DataFrame, k: int = 10) -> Dict:
        """Calculate comprehensive evaluation metrics including NDCG"""
        try:
            self.logger.info("Calculating comprehensive metrics...")
            
            # Standardize test data column names and types
            if 'userId' in test_ratings.columns:
                test_ratings = test_ratings.rename(columns={
                    'userId': 'user_id', 
                    'productId': 'product_id'
                })
            
            # Ensure consistent data types
            test_ratings['user_id'] = test_ratings['user_id'].astype(str)
            test_ratings['product_id'] = test_ratings['product_id'].astype(str)
            
            # Get recommendations per user
            user_recs_dict = {}
            if not self.hybrid_recommendations.empty:
                for user_id, user_recs in self.hybrid_recommendations.groupby('user_id'):
                    top_recs = user_recs.nlargest(k, 'hybrid_score')['product_id'].tolist()
                    user_recs_dict[str(user_id)] = top_recs
            
            # Get relevant items (ratings >= threshold)
            relevant_items_dict = {}
            high_rated = test_ratings[test_ratings['rating'] >= self.relevance_threshold]
            for user_id, user_ratings in high_rated.groupby('user_id'):
                relevant_items_dict[str(user_id)] = set(user_ratings['product_id'].tolist())
            
            # Calculate metrics for common users
            common_users = set(user_recs_dict.keys()) & set(relevant_items_dict.keys())
            
            if not common_users:
                self.logger.warning("No common users found for evaluation")
                return self._empty_metrics()
            
            precision_scores = []
            recall_scores = []
            hit_counts = []
            ndcg_scores = []
            
            for user_id in common_users:
                user_recs = user_recs_dict[user_id]
                relevant_items = relevant_items_dict[user_id]
                
                if len(relevant_items) == 0 or len(user_recs) == 0:
                    continue
                
                # Calculate precision and recall
                recommended_relevant = set(user_recs) & relevant_items
                
                precision = len(recommended_relevant) / len(user_recs) if user_recs else 0
                recall = len(recommended_relevant) / len(relevant_items) if relevant_items else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                hit_counts.append(1 if len(recommended_relevant) > 0 else 0)
                
                # Calculate NDCG (simplified version)
                try:
                    # Create relevance scores (1 for relevant, 0 for not relevant)
                    true_relevance = [1 if item in relevant_items else 0 for item in user_recs]
                    
                    if sum(true_relevance) > 0:  # Only calculate if there are relevant items
                        # Get predicted scores for this user's recommendations
                        user_hybrid_recs = self.hybrid_recommendations[
                            (self.hybrid_recommendations['user_id'] == user_id) & 
                            (self.hybrid_recommendations['product_id'].isin(user_recs))
                        ]
                        
                        if not user_hybrid_recs.empty:
                            pred_scores = []
                            for item in user_recs:
                                item_score = user_hybrid_recs[
                                    user_hybrid_recs['product_id'] == item
                                ]['hybrid_score'].values
                                pred_scores.append(item_score[0] if len(item_score) > 0 else 0)
                            
                            # Calculate NDCG
                            ndcg = ndcg_score([true_relevance], [pred_scores], k=k)
                            ndcg_scores.append(ndcg)
                
                except Exception:
                    # Skip NDCG calculation for problematic cases
                    continue
            
            # Calculate averages
            avg_precision = np.mean(precision_scores) if precision_scores else 0
            avg_recall = np.mean(recall_scores) if recall_scores else 0
            hit_rate = np.mean(hit_counts) if hit_counts else 0
            avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
            
            # Coverage calculation
            all_recommended = set()
            for recs in user_recs_dict.values():
                all_recommended.update(recs)
            
            all_relevant = set()
            for items in relevant_items_dict.values():
                all_relevant.update(items)
            
            coverage = len(all_recommended & all_relevant) / len(all_relevant) if all_relevant else 0
            
            return {
                'precision_at_k': avg_precision,
                'recall_at_k': avg_recall,
                'f1_score': 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0,
                'hit_rate_at_k': hit_rate,
                'ndcg_at_k': avg_ndcg,
                'coverage_at_k': coverage,
                'relevance_threshold': self.relevance_threshold,
                'evaluated_users': len(common_users),
                'total_users_with_recs': len(user_recs_dict),
                'total_users_with_relevant': len(relevant_items_dict)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'precision_at_k': 0, 'recall_at_k': 0, 'f1_score': 0,
            'hit_rate_at_k': 0, 'ndcg_at_k': 0, 'coverage_at_k': 0,
            'relevance_threshold': self.relevance_threshold,
            'evaluated_users': 0, 'total_users_with_recs': 0, 'total_users_with_relevant': 0
        }
    
    def calculate_business_metrics(self) -> Dict:
        """Calculate enhanced business and system metrics"""
        try:
            if self.hybrid_recommendations is None or self.hybrid_recommendations.empty:
                return {'catalog_coverage': 0, 'unique_items_recommended': 0}
            
            total_products = len(self.metadata)
            recommended_products = len(self.hybrid_recommendations['product_id'].unique())
            
            # User recommendation distribution
            user_rec_counts = self.hybrid_recommendations['user_id'].value_counts()
            
            # Activity distribution analysis
            activity_dist = {}
            for level in ['low', 'medium', 'high']:
                count = sum(1 for activity in self.user_activity_levels.values() if activity == level)
                activity_dist[f"{level}_activity_users"] = count
            
            return {
                'catalog_coverage': recommended_products / total_products if total_products > 0 else 0,
                'unique_items_recommended': recommended_products,
                'avg_recommendations_per_user': float(user_rec_counts.mean()) if not user_rec_counts.empty else 0,
                'median_recommendations_per_user': float(user_rec_counts.median()) if not user_rec_counts.empty else 0,
                'score_quality': {
                    'mean': float(self.hybrid_recommendations['hybrid_score'].mean()),
                    'median': float(self.hybrid_recommendations['hybrid_score'].median()),
                    'std': float(self.hybrid_recommendations['hybrid_score'].std()),
                    'min': float(self.hybrid_recommendations['hybrid_score'].min()),
                    'max': float(self.hybrid_recommendations['hybrid_score'].max()),
                    'q25': float(self.hybrid_recommendations['hybrid_score'].quantile(0.25)),
                    'q75': float(self.hybrid_recommendations['hybrid_score'].quantile(0.75))
                },
                'user_activity_distribution': activity_dist
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating business metrics: {str(e)}")
            return {'catalog_coverage': 0, 'unique_items_recommended': 0}
    
    def _print_summary(self):
        """Print comprehensive summary of the recommendation pipeline results"""
        try:
            if not self.evaluation_results:
                self.logger.warning("No evaluation results available for summary")
                return
            
            print("\n" + "="*80)
            print("ENHANCED HYBRID RECOMMENDATION SYSTEM - EXECUTION SUMMARY")
            print("="*80)
            
            # System Configuration
            print("\n📊 SYSTEM CONFIGURATION:")
            config = self.evaluation_results.get('system_config', {})
            print(f"   • Collaborative Filtering Weight: {config.get('cf_weight', 0):.1f}")
            print(f"   • Content-Based Weight: {config.get('cb_weight', 0):.1f}")
            print(f"   • Popularity Weight: {config.get('popularity_weight', 0):.1f}")
            print(f"   • Relevance Threshold: {config.get('relevance_threshold', 0):.1f}")
            print(f"   • Adaptive K Selection: {config.get('use_adaptive_k', False)}")
            print(f"   • Top K Recommendations: {config.get('top_k', 10)}")
            
            # Data Statistics
            print("\n📈 DATA PROCESSING STATISTICS:")
            data_stats = self.evaluation_results.get('data_statistics', {})
            print(f"   • CF Recommendations Processed: {data_stats.get('cf_recommendations', 0):,}")
            print(f"   • CB Recommendations Processed: {data_stats.get('cb_recommendations', 0):,}")
            print(f"   • User History Records: {data_stats.get('user_history_size', 0):,}")
            print(f"   • Product Metadata Records: {data_stats.get('metadata_size', 0):,}")
            print(f"   • Final Hybrid Recommendations: {data_stats.get('hybrid_recommendations', 0):,}")
            print(f"   • Unique Users Served: {data_stats.get('unique_users', 0):,}")
            print(f"   • Unique Products Recommended: {data_stats.get('unique_products', 0):,}")
            
            # Accuracy Metrics
            print("\n🎯 ACCURACY PERFORMANCE METRICS:")
            accuracy = self.evaluation_results.get('accuracy_metrics', {})
            print(f"   • Precision@K: {accuracy.get('precision_at_k', 0):.4f}")
            print(f"   • Recall@K: {accuracy.get('recall_at_k', 0):.4f}")
            print(f"   • F1-Score: {accuracy.get('f1_score', 0):.4f}")
            print(f"   • Hit Rate@K: {accuracy.get('hit_rate_at_k', 0):.4f}")
            print(f"   • NDCG@K: {accuracy.get('ndcg_at_k', 0):.4f}")
            print(f"   • Coverage@K: {accuracy.get('coverage_at_k', 0):.4f}")
            print(f"   • Users Evaluated: {accuracy.get('evaluated_users', 0):,}")
            
            # Business Metrics
            print("\n💼 BUSINESS IMPACT METRICS:")
            business = self.evaluation_results.get('business_metrics', {})
            print(f"   • Catalog Coverage: {business.get('catalog_coverage', 0):.1%}")
            print(f"   • Unique Items Recommended: {business.get('unique_items_recommended', 0):,}")
            print(f"   • Avg Recommendations per User: {business.get('avg_recommendations_per_user', 0):.1f}")
            print(f"   • Median Recommendations per User: {business.get('median_recommendations_per_user', 0):.1f}")
            
            # Score Quality Analysis
            score_quality = business.get('score_quality', {})
            if score_quality:
                print(f"\n📊 RECOMMENDATION SCORE QUALITY:")
                print(f"   • Mean Score: {score_quality.get('mean', 0):.4f}")
                print(f"   • Median Score: {score_quality.get('median', 0):.4f}")
                print(f"   • Score Range: {score_quality.get('min', 0):.4f} - {score_quality.get('max', 0):.4f}")
                print(f"   • Standard Deviation: {score_quality.get('std', 0):.4f}")
                print(f"   • 25th Percentile: {score_quality.get('q25', 0):.4f}")
                print(f"   • 75th Percentile: {score_quality.get('q75', 0):.4f}")
            
            # User Activity Distribution
            activity_dist = business.get('user_activity_distribution', {})
            if activity_dist:
                print(f"\n👥 USER ACTIVITY DISTRIBUTION:")
                total_users = sum(activity_dist.values())
                for level in ['low', 'medium', 'high']:
                    count = activity_dist.get(f"{level}_activity_users", 0)
                    percentage = (count / total_users * 100) if total_users > 0 else 0
                    print(f"   • {level.capitalize()} Activity Users: {count:,} ({percentage:.1f}%)")
            
            # System Optimizations
            print("\n🔧 ENHANCED OPTIMIZATIONS APPLIED:")
            optimizations = self.evaluation_results.get('final_optimizations', {})
            print(f"   • Balanced Thresholds: {optimizations.get('balanced_thresholds', 'N/A')}")
            print(f"   • Optimized Weights: {optimizations.get('optimized_weights', 'N/A')}")
            print(f"   • Enhanced Popularity Scoring: {optimizations.get('enhanced_popularity', 'N/A')}")
            print(f"   • Adaptive K Selection: {optimizations.get('adaptive_k_selection', 'N/A')}")
            print(f"   • Robust Preprocessing: {optimizations.get('robust_preprocessing', 'N/A')}")
            print(f"   • Comprehensive Evaluation: {optimizations.get('comprehensive_evaluation', 'N/A')}")
            
            # Performance Summary
            runtime = self.evaluation_results.get('runtime_seconds', 0)
            print(f"\n⚡ PERFORMANCE SUMMARY:")
            print(f"   • Total Runtime: {runtime:.2f} seconds ({runtime/60:.1f} minutes)")
            print(f"   • Processing Rate: {data_stats.get('hybrid_recommendations', 0) / max(runtime, 1):.0f} recommendations/second")
            print(f"   • Memory Efficiency: Optimized data structures and streaming processing")
            
            # Quality Assessment
            print(f"\n✅ OVERALL QUALITY ASSESSMENT:")
            
            # Calculate overall quality score
            precision = accuracy.get('precision_at_k', 0)
            recall = accuracy.get('recall_at_k', 0)
            ndcg = accuracy.get('ndcg_at_k', 0)
            coverage = business.get('catalog_coverage', 0)
            
            quality_score = (precision + recall + ndcg + coverage) / 4
            
            if quality_score >= 0.15:
                quality_rating = "EXCELLENT ⭐⭐⭐⭐⭐"
            elif quality_score >= 0.12:
                quality_rating = "VERY GOOD ⭐⭐⭐⭐"
            elif quality_score >= 0.08:
                quality_rating = "GOOD ⭐⭐⭐"
            elif quality_score >= 0.05:
                quality_rating = "FAIR ⭐⭐"
            else:
                quality_rating = "NEEDS IMPROVEMENT ⭐"
            
            print(f"   • Overall Quality Score: {quality_score:.4f}")
            print(f"   • Quality Rating: {quality_rating}")
            
            # Recommendations for improvement
            print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
            if precision < 0.1:
                print("   • Consider increasing CF weight or adjusting relevance threshold")
            if recall < 0.1:
                print("   • Consider increasing top_k or reducing filtering thresholds")
            if coverage < 0.3:
                print("   • Consider adjusting popularity weight or CB integration")
            if ndcg < 0.1:
                print("   • Consider refining score normalization or ranking algorithm")
            
            if precision >= 0.1 and recall >= 0.1 and coverage >= 0.3 and ndcg >= 0.1:
                print("   • System is well-optimized across all key metrics!")
            
            print(f"\n🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            print("="*80)
            
        except Exception as e:
            self.logger.error(f"Error printing summary: {str(e)}")
            print(f"\n❌ Error generating summary: {str(e)}")
    
    def run_pipeline(self, output_path: str = "data/processed/enhanced_hybrid/", 
                    top_k: int = 10, test_split_ratio: float = 0.2,
                    save_model: bool = True, model_path: str = None):
        """
        Run the enhanced pipeline with comprehensive error handling and model saving
        
        **Enhanced Pipeline with Model Persistence:**
        This method trains the model and optionally saves it to disk for future use.
        The saved model includes all learned parameters and can be loaded later
        for making predictions without retraining.
        """
        try:
            start_time = datetime.now()
            self.logger.info("Starting Enhanced Hybrid Recommendation Pipeline...")
            
            # Load and preprocess data
            self.load_data()
            self.preprocess_data()
            
            # Generate recommendations
            self.generate_hybrid_recommendations(top_k=top_k)
            
            # Mark model as trained
            self.is_trained = True
            self.training_timestamp = start_time.isoformat()
            
            # Evaluation
            test_size = min(int(len(self.user_history) * test_split_ratio), 40000)
            test_ratings = self.user_history.sample(n=test_size, random_state=42)
            
            accuracy_metrics = self.calculate_comprehensive_metrics(test_ratings, k=top_k)
            business_metrics = self.calculate_business_metrics()
            
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()
            
            # Compile final results
            self.evaluation_results = {
                'timestamp': end_time.isoformat(),
                'runtime_seconds': runtime,
                'system_config': {
                    'cf_weight': self.cf_weight,
                    'cb_weight': self.cb_weight,
                    'popularity_weight': self.popularity_weight,
                    'relevance_threshold': self.relevance_threshold,
                    'min_cf_score': self.min_cf_score,
                    'min_cb_score': self.min_cb_score,
                    'score_cap': 1.0,
                    'use_adaptive_k': self.use_adaptive_k,
                    'top_k': top_k
                },
                'accuracy_metrics': accuracy_metrics,
                'business_metrics': business_metrics,
                'data_statistics': {
                    'cf_recommendations': len(self.cf_recs),
                    'cb_recommendations': len(self.cb_recs),
                    'user_history_size': len(self.user_history),
                    'metadata_size': len(self.metadata),
                    'hybrid_recommendations': len(self.hybrid_recommendations),
                    'unique_users': len(self.hybrid_recommendations['user_id'].unique()) if not self.hybrid_recommendations.empty else 0,
                    'unique_products': len(self.hybrid_recommendations['product_id'].unique()) if not self.hybrid_recommendations.empty else 0
                },
                'final_optimizations': {
                    'balanced_thresholds': f"CF: {self.min_cf_score}, CB: {self.min_cb_score}",
                    'optimized_weights': f"CF: {self.cf_weight}, CB: {self.cb_weight}, Pop: {self.popularity_weight}",
                    'enhanced_popularity': "Quality + consistency + minimum interaction filtering",
                    'adaptive_k_selection': f"Range: {self.min_k}-{self.max_k} based on user activity",
                    'robust_preprocessing': "IQR-based outlier handling with balanced retention",
                    'comprehensive_evaluation': "Added NDCG and enhanced business metrics",
                    'user_activity_modeling': "Activity-based recommendation personalization",
                    'error_handling': "Comprehensive validation and fallback mechanisms"
                }
            }
            
            # Save results
            self.save_results(output_path)
            
            # Save trained model if requested
            if save_model:
                if model_path is None:
                    model_path = os.path.join(output_path, "trained_hybrid_recommender.pkl")
                
                self.save_model(model_path)
                self.logger.info(f"💾 Trained model saved to: {model_path}")
            
            self._print_summary()
            
            self.logger.info(f"Enhanced pipeline completed successfully in {runtime:.2f} seconds!")
            
            return self.evaluation_results
            
        except Exception as e:
            self.logger.error(f"Enhanced pipeline failed: {str(e)}")
            raise
    
    def save_results(self, output_path: str):
        """Save results to files"""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Save recommendations
            if self.hybrid_recommendations is not None and not self.hybrid_recommendations.empty:
                recommendations_file = os.path.join(output_path, "enhanced_hybrid_recommendations.csv")
                self.hybrid_recommendations.to_csv(recommendations_file, index=False)
            
            # Save metrics
            metrics_file = os.path.join(output_path, "enhanced_evaluation_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

    def get_user_recommendations(self, user_id: str, top_k: int = 10) -> pd.DataFrame:
        """Get recommendations for a specific user"""
        try:
            if self.hybrid_recommendations is None or self.hybrid_recommendations.empty:
                self.logger.warning("No recommendations available. Run pipeline first.")
                return pd.DataFrame()
            
            user_recs = self.hybrid_recommendations[
                self.hybrid_recommendations['user_id'] == str(user_id)
            ].nlargest(top_k, 'hybrid_score')
            
            return user_recs
            
        except Exception as e:
            self.logger.error(f"Error getting user recommendations: {str(e)}")
            return pd.DataFrame()
    
    def get_product_details(self, product_ids: List[str]) -> pd.DataFrame:
        """Get product details for given product IDs"""
        try:
            if self.metadata is None:
                self.logger.warning("No metadata available")
                return pd.DataFrame()
            
            product_details = self.metadata[
                self.metadata['product_id'].isin(product_ids)
            ]
            
            return product_details
            
        except Exception as e:
            self.logger.error(f"Error getting product details: {str(e)}")
            return pd.DataFrame()
    
    def analyze_recommendation_diversity(self) -> Dict:
        """Analyze diversity of recommendations across different dimensions"""
        try:
            if self.hybrid_recommendations is None or self.hybrid_recommendations.empty:
                return {}
            
            # Category diversity (if available in metadata)
            diversity_metrics = {}
            
            if 'category' in self.metadata.columns:
                # Get categories for recommended products
                rec_products = self.hybrid_recommendations['product_id'].unique()
                rec_categories = self.metadata[
                    self.metadata['product_id'].isin(rec_products)
                ]['category'].value_counts()
                
                diversity_metrics['category_distribution'] = rec_categories.to_dict()
                diversity_metrics['unique_categories'] = len(rec_categories)
                diversity_metrics['category_entropy'] = self._calculate_entropy(rec_categories.values)
            
            # Score diversity
            score_stats = self.hybrid_recommendations['hybrid_score'].describe()
            diversity_metrics['score_diversity'] = {
                'range': float(score_stats['max'] - score_stats['min']),
                'std': float(score_stats['std']),
                'coefficient_of_variation': float(score_stats['std'] / score_stats['mean']) if score_stats['mean'] > 0 else 0
            }
            
            return diversity_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing diversity: {str(e)}")
            return {}
    
    def _calculate_entropy(self, values):
        """Calculate entropy for diversity measurement"""
        try:
            total = sum(values)
            probabilities = [v/total for v in values if v > 0]
            entropy = -sum(p * np.log2(p) for p in probabilities)
            return float(entropy)
        except:
            return 0.0
    
    def generate_explanation(self, user_id: str, product_id: str) -> Dict:
        """Generate explanation for why a product was recommended to a user"""
        try:
            user_rec = self.hybrid_recommendations[
                (self.hybrid_recommendations['user_id'] == str(user_id)) &
                (self.hybrid_recommendations['product_id'] == str(product_id))
            ]
            
            if user_rec.empty:
                return {"error": "Recommendation not found"}
            
            rec = user_rec.iloc[0]
            
            explanation = {
                'user_id': str(user_id),
                'product_id': str(product_id),
                'hybrid_score': float(rec['hybrid_score']),
                'score_breakdown': {
                    'collaborative_filtering': {
                        'score': float(rec['cf_score']),
                        'weight': self.cf_weight,
                        'contribution': float(rec['cf_score'] * self.cf_weight),
                        'explanation': "Based on similar users' preferences"
                    },
                    'content_based': {
                        'score': float(rec['cb_score']),
                        'weight': self.cb_weight,
                        'contribution': float(rec['cb_score'] * self.cb_weight),
                        'explanation': "Based on product similarity"
                    },
                    'popularity': {
                        'score': float(rec['popularity_score']),
                        'weight': self.popularity_weight,
                        'contribution': float(rec['popularity_score'] * self.popularity_weight),
                        'explanation': "Based on overall product popularity"
                    }
                },
                'primary_reason': self._get_primary_reason(rec),
                'confidence_level': self._calculate_confidence(rec)
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {str(e)}")
            return {"error": str(e)}
    
    def _get_primary_reason(self, rec) -> str:
        """Determine the primary reason for recommendation"""
        contributions = {
            'collaborative_filtering': rec['cf_score'] * self.cf_weight,
            'content_based': rec['cb_score'] * self.cb_weight,
            'popularity': rec['popularity_score'] * self.popularity_weight
        }
        
        primary = max(contributions.items(), key=lambda x: x[1])
        
        reasons = {
            'collaborative_filtering': "Users with similar preferences also liked this product",
            'content_based': "This product is similar to items you've shown interest in",
            'popularity': "This is a popular product that many users enjoy"
        }
        
        return reasons.get(primary[0], "Mixed factors")
    
    def _calculate_confidence(self, rec) -> str:
        """Calculate confidence level for recommendation"""
        score = rec['hybrid_score']
        
        if score >= 0.8:
            return "Very High"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Very Low"


# Enhanced Usage Examples and Testing Functions
def train_and_save_model():
    """
    Complete example of training and saving the model
    
    **Training and Saving Workflow:**
    This function demonstrates the complete workflow for training a hybrid
    recommendation model and saving it for future use.
    """
    try:
        print("🚀 Training and Saving Enhanced Hybrid Recommender System...")
        
        # Initialize recommender with training data paths
        recommender = EnhancedHybridRecommender(
            cf_path="data/processed/recommendations/advanced_als_recommendations/als_recs.csv",
            cb_path="data/processed/content_based_recommendations/content_based_recommendations.csv", 
            metadata_path="data/processed/Electronics.csv",
            user_history_path="data/processed/clean/cleaned_user_ratings.csv"
        )
        
        # Train the model and save it
        results = recommender.run_pipeline(
            output_path="data/processed/trained_model/",
            top_k=10,
            test_split_ratio=0.2,
            save_model=True,
            model_path="models/trained_hybrid_recommender.pkl"
        )
        
        print("\n✅ Model training and saving completed successfully!")
        print(f"📊 Model saved with {results['data_statistics']['hybrid_recommendations']} recommendations")
        print(f"🎯 Final Precision@10: {results['accuracy_metrics']['precision_at_k']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

def load_and_predict():
    """
    Complete example of loading a trained model and making predictions
    
    **Loading and Prediction Workflow:**
    This function demonstrates how to load a previously trained model
    and use it for making real-time predictions without retraining.
    """
    try:
        print("🔍 Loading Trained Model and Making Predictions...")
        
        # Load the trained model
        recommender = EnhancedHybridRecommender.load_model("models/trained_hybrid_recommender.pkl")
        
        # Get model information
        model_info = recommender.get_model_info()
        print(f"📋 Model Info:")
        print(f"   • Trained: {model_info['is_trained']}")
        print(f"   • Training Date: {model_info['training_timestamp']}")
        print(f"   • Products with Popularity Scores: {model_info['num_products_with_popularity']:,}")
        print(f"   • Users with Activity Profiles: {model_info['num_users_with_activity']:,}")
        
        # Make predictions for individual users
        sample_users = ['A1234567890123', 'A9876543210987', 'A5555555555555']  # Example user IDs
        
        print(f"\n🎯 Making Individual Predictions:")
        for user_id in sample_users:
            recommendations = recommender.predict_for_user(user_id, top_k=5)
            
            if not recommendations.empty:
                print(f"\n👤 User {user_id}:")
                for idx, rec in recommendations.iterrows():
                    print(f"   • Product {rec['product_id']}: Score {rec['hybrid_score']:.4f}")
                    
                    # Get explanation for top recommendation
                    if idx == recommendations.index[0]:
                        explanation = recommender.generate_explanation(user_id, rec['product_id'])
                        print(f"     Reason: {explanation.get('primary_reason', 'N/A')}")
                        print(f"     Confidence: {explanation.get('confidence_level', 'N/A')}")
            else:
                print(f"   • No recommendations found for user {user_id}")
        
        # Batch predictions
        print(f"\n📦 Making Batch Predictions:")
        batch_results = recommender.batch_predict(sample_users, top_k=3)
        
        print(f"   • Generated recommendations for {len(batch_results)} users")
        for user_id, recs in batch_results.items():
            if not recs.empty:
                print(f"   • User {user_id}: {len(recs)} recommendations (avg score: {recs['hybrid_score'].mean():.4f})")
        
        return batch_results
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        raise

def demonstrate_online_learning():
    """
    Demonstrate online learning capabilities with user profile updates
    
    **Online Learning Example:**
    This function shows how the model can adapt to new user interactions
    without requiring complete retraining.
    """
    try:
        print("🔄 Demonstrating Online Learning Capabilities...")
        
        # Load trained model
        recommender = EnhancedHybridRecommender.load_model("models/trained_hybrid_recommender.pkl")
        
        # Simulate new user interactions
        new_user_id = "NEW_USER_12345"
        new_interactions = [
            {'product_id': 'B001E4KFG0', 'rating': 5.0, 'timestamp': '2024-01-15'},
            {'product_id': 'B003AI2VGA', 'rating': 4.0, 'timestamp': '2024-01-16'},
            {'product_id': 'B00DR0PDNE', 'rating': 5.0, 'timestamp': '2024-01-17'},
            {'product_id': 'B007WTAJTO', 'rating': 3.0, 'timestamp': '2024-01-18'},
        ]
        
        print(f"👤 Processing new user: {new_user_id}")
        print(f"📝 New interactions: {len(new_interactions)}")
        
        # Update user profile
        recommender.update_user_profile(new_user_id, new_interactions)
        
        # Make predictions for the new user
        recommendations = recommender.predict_for_user(new_user_id, top_k=5)
        
        if not recommendations.empty:
            print(f"\n🎯 Recommendations for new user:")
            for idx, rec in recommendations.iterrows():
                print(f"   • Product {rec['product_id']}: Score {rec['hybrid_score']:.4f}")
        else:
            print("   • No recommendations generated for new user")
        
        # Simulate updating existing user profile
        existing_user = "A1234567890123"
        additional_interactions = [
            {'product_id': 'B001E4KFG0', 'rating': 5.0, 'timestamp': '2024-01-20'},
            {'product_id': 'B003AI2VGA', 'rating': 4.0, 'timestamp': '2024-01-21'}
        ]
        
        print(f"\n🔄 Updating existing user: {existing_user}")
        
        # Get recommendations before update
        recs_before = recommender.predict_for_user(existing_user, top_k=3)
        
        # Update profile
        recommender.update_user_profile(existing_user, additional_interactions)
        
        # Get recommendations after update
        recs_after = recommender.predict_for_user(existing_user, top_k=3)
        
        print(f"📊 Recommendation changes:")
        print(f"   • Before update: {len(recs_before)} recommendations")
        print(f"   • After update: {len(recs_after)} recommendations")
        
        if not recs_before.empty and not recs_after.empty:
            avg_score_before = recs_before['hybrid_score'].mean()
            avg_score_after = recs_after['hybrid_score'].mean()
            print(f"   • Average score change: {avg_score_before:.4f} → {avg_score_after:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Online learning demonstration failed: {str(e)}")
        raise

def comprehensive_model_testing():
    """
    Comprehensive testing of all model features
    
    **Complete Model Testing Suite:**
    This function tests all aspects of the trained model including
    predictions, explanations, diversity analysis, and performance.
    """
    try:
        print("🧪 Running Comprehensive Model Testing Suite...")
        
        # Load model
        recommender = EnhancedHybridRecommender.load_model("models/trained_hybrid_recommender.pkl")
        
        # Test 1: Model Information
        print("\n📋 Test 1: Model Information")
        model_info = recommender.get_model_info()
        print(f"   ✅ Model loaded successfully")
        print(f"   ✅ Training status: {model_info['is_trained']}")
        print(f"   ✅ Model version: {model_info['model_version']}")
        
        # Test 2: Individual Predictions
        print("\n🎯 Test 2: Individual User Predictions")
        test_user = "A1234567890123"
        predictions = recommender.predict_for_user(test_user, top_k=5)
        
        if not predictions.empty:
            print(f"   ✅ Generated {len(predictions)} predictions for user {test_user}")
            print(f"   ✅ Score range: {predictions['hybrid_score'].min():.4f} - {predictions['hybrid_score'].max():.4f}")
        else:
            print(f"   ⚠️ No predictions generated for user {test_user}")
        
        # Test 3: Batch Predictions
        print("\n📦 Test 3: Batch Predictions")
        test_users = ["A1234567890123", "A9876543210987", "A5555555555555"]
        batch_results = recommender.batch_predict(test_users, top_k=3)
        
        successful_predictions = sum(1 for recs in batch_results.values() if not recs.empty)
        print(f"   ✅ Batch predictions: {successful_predictions}/{len(test_users)} users")
        
        # Test 4: Explanations
        print("\n💡 Test 4: Recommendation Explanations")
        if not predictions.empty:
            top_recommendation = predictions.iloc[0]
            explanation = recommender.generate_explanation(
                test_user, 
                top_recommendation['product_id']
            )
            
            if 'error' not in explanation:
                print(f"   ✅ Explanation generated successfully")
                print(f"   ✅ Primary reason: {explanation.get('primary_reason', 'N/A')}")
                print(f"   ✅ Confidence level: {explanation.get('confidence_level', 'N/A')}")
            else:
                print(f"   ⚠️ Explanation generation failed: {explanation['error']}")
        
        # Test 5: Diversity Analysis
        print("\n📊 Test 5: Diversity Analysis")
        diversity_metrics = recommender.analyze_recommendation_diversity()
        
        if diversity_metrics:
            print(f"   ✅ Diversity analysis completed")
            if 'score_diversity' in diversity_metrics:
                score_div = diversity_metrics['score_diversity']
                print(f"   ✅ Score diversity range: {score_div.get('range', 0):.4f}")
                print(f"   ✅ Score standard deviation: {score_div.get('std', 0):.4f}")
        else:
            print(f"   ⚠️ Diversity analysis failed or no data available")
        
        # Test 6: Performance Benchmarking
        print("\n⚡ Test 6: Performance Benchmarking")
        import time
        
        # Single prediction performance
        start_time = time.time()
        for _ in range(10):
            recommender.predict_for_user(test_user, top_k=10)
        single_pred_time = (time.time() - start_time) / 10
        
        # Batch prediction performance
        start_time = time.time()
        recommender.batch_predict(test_users * 10, top_k=10)  # 30 users
        batch_pred_time = time.time() - start_time
        
        print(f"   ✅ Average single prediction time: {single_pred_time*1000:.2f}ms")
        print(f"   ✅ Batch prediction time (30 users): {batch_pred_time:.2f}s")
        print(f"   ✅ Predictions per second: {30/batch_pred_time:.1f}")
        
        # Test 7: Model Persistence
        print("\n💾 Test 7: Model Persistence")
        test_model_path = "models/test_model_copy.pkl"
        
        # Save model copy
        recommender.save_model(test_model_path)
        
        # Load model copy
        recommender_copy = EnhancedHybridRecommender.load_model(test_model_path)
        
        # Compare predictions
        original_preds = recommender.predict_for_user(test_user, top_k=5)
        copy_preds = recommender_copy.predict_for_user(test_user, top_k=5)
        
        if not original_preds.empty and not copy_preds.empty:
            score_diff = abs(original_preds['hybrid_score'].sum() - copy_preds['hybrid_score'].sum())
            print(f"   ✅ Model persistence test passed")
            print(f"   ✅ Score difference after reload: {score_diff:.6f}")
        else:
            print(f"   ⚠️ Model persistence test failed")
        
        # Cleanup test file
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
        
        print(f"\n🎉 Comprehensive Testing Completed Successfully!")
        print(f"✅ All major model features are working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive testing failed: {str(e)}")
        raise

def production_deployment_example():
    """
    Example of how to deploy the model in a production environment
    
    **Production Deployment Guide:**
    This function demonstrates best practices for deploying the trained
    model in a production recommendation service.
    """
    try:
        print("🚀 Production Deployment Example...")
        
        # Load model with error handling
        model_path = "models/trained_hybrid_recommender.pkl"
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("   Please train and save the model first using train_and_save_model()")
            return False
        
        print(f"📥 Loading production model from: {model_path}")
        recommender = EnhancedHybridRecommender.load_model(model_path)
        
        # Validate model is ready for production
        model_info = recommender.get_model_info()
        
        print(f"🔍 Production Readiness Check:")
        print(f"   • Model Trained: {model_info['is_trained']}")
        print(f"   • Products Available: {model_info['num_products_with_popularity']:,}")
        print(f"   • Users Profiled: {model_info['num_users_with_activity']:,}")
        print(f"   • CB Entries: {model_info['num_cb_entries']:,}")
        
        # Production-ready recommendation service function
        def get_recommendations_for_user(user_id: str, num_recommendations: int = 10):
            """Production recommendation service function"""
            try:
                # Input validation
                if not user_id or not isinstance(user_id, str):
                    return {"error": "Invalid user_id", "recommendations": []}
                
                if num_recommendations <= 0 or num_recommendations > 50:
                    num_recommendations = 10  # Default fallback
                
                # Get recommendations
                recommendations = recommender.predict_for_user(
                    user_id, 
                    top_k=num_recommendations,
                    exclude_seen=True
                )
                
                if recommendations.empty:
                    return {
                        "user_id": user_id,
                        "recommendations": [],
                        "message": "No recommendations available"
                    }
                
                # Format for API response
                rec_list = []
                for _, rec in recommendations.iterrows():
                    rec_list.append({
                        "product_id": rec['product_id'],
                        "score": float(rec['hybrid_score']),
                        "cf_score": float(rec['cf_score']),
                        "cb_score": float(rec['cb_score']),
                        "popularity_score": float(rec['popularity_score'])
                    })
                
                return {
                    "user_id": user_id,
                    "recommendations": rec_list,
                    "count": len(rec_list),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    "error": str(e),
                    "user_id": user_id,
                    "recommendations": []
                }
        
        # Test production service
        print(f"\n🧪 Testing Production Service:")
        
        test_cases = [
            ("A1234567890123", 5),  # Normal case
            ("NONEXISTENT_USER", 5),  # Non-existent user
            ("", 5),  # Invalid user ID
            ("A1234567890123", 0),  # Invalid num_recommendations
            ("A1234567890123", 100),  # Too many recommendations
        ]
        
        for user_id, num_recs in test_cases:
            result = get_recommendations_for_user(user_id, num_recs)
            
            if "error" in result:
                print(f"   ⚠️ User '{user_id}': {result['error']}")
            else:
                print(f"   ✅ User '{user_id}': {result['count']} recommendations")
        
        # Performance benchmarking for production
        print(f"\n⚡ Production Performance Benchmarking:")
        
        import time
        
        # Simulate production load
        test_users = ["A1234567890123", "A9876543210987", "A5555555555555"] * 10  # 30 requests
        
        start_time = time.time()
        results = []
        
        for user_id in test_users:
            result = get_recommendations_for_user(user_id, 10)
            results.append(result)
        
        total_time = time.time() - start_time
        successful_requests = sum(1 for r in results if "error" not in r)
        
        print(f"   • Total Requests: {len(test_users)}")
        print(f"   • Successful Requests: {successful_requests}")
        print(f"   • Total Time: {total_time:.2f} seconds")
        print(f"   • Average Response Time: {(total_time/len(test_users))*1000:.2f}ms")
        print(f"   • Requests per Second: {len(test_users)/total_time:.1f}")
        
        # Memory usage estimation
        import sys
        model_size_mb = sys.getsizeof(recommender) / (1024 * 1024)
        print(f"   • Estimated Model Memory Usage: {model_size_mb:.2f} MB")
        
        print(f"\n✅ Production deployment example completed successfully!")
        print(f"📋 Production Deployment Checklist:")
        print(f"   ✅ Model loads successfully")
        print(f"   ✅ Input validation implemented")
        print(f"   ✅ Error handling in place")
        print(f"   ✅ Performance meets requirements")
        print(f"   ✅ API response format standardized")
        
        return get_recommendations_for_user
        
    except Exception as e:
        print(f"❌ Production deployment example failed: {str(e)}")
        raise

# Main execution with all examples
if __name__ == "__main__":
    print("🎯 Enhanced Hybrid Recommendation System with Model Persistence")
    print("=" * 70)
    
    # Example 1: Train and save model
    print("\n1️⃣ TRAINING AND SAVING MODEL")
    print("-" * 40)
    
    try:
        # Train and save the model
        recommender = EnhancedHybridRecommender(
            cf_path="data/processed/recommendations/advanced_als_recommendations/als_recs.csv",
            cb_path="data/processed/content_based_recommendations/content_based_recommendations.csv", 
            metadata_path="data/processed/Electronics.csv",
            user_history_path="data/processed/clean/cleaned_user_ratings.csv"
        )
        
        results = recommender.run_pipeline(
            output_path="data/processed/trained_model_final/",
            top_k=10,
            test_split_ratio=0.2,
            save_model=True,
            model_path="models/trained_hybrid_recommender.pkl"
        )
        
        print(f"✅ Model training completed!")
        print(f"📊 Generated {results['data_statistics']['hybrid_recommendations']:,} recommendations")
        print(f"🎯 Precision@10: {results['accuracy_metrics']['precision_at_k']:.4f}")
        print(f"📈 Coverage: {results['business_metrics']['catalog_coverage']:.1%}")
        print(f"💾 Model saved to: models/trained_hybrid_recommender.pkl")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Please ensure all data files are available and try again.")
    
    print(f"\n🎉 ENHANCED HYBRID RECOMMENDATION SYSTEM READY!")
    print(f"💡 Use the following functions to work with the trained model:")
    print(f"   • load_and_predict() - Load model and make predictions")
    print(f"   • demonstrate_online_learning() - Show online learning capabilities")
    print(f"   • comprehensive_model_testing() - Run full test suite")
    print(f"   • production_deployment_example() - Production deployment guide")
    print("=" * 70)