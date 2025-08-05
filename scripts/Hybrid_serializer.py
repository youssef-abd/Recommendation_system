import pandas as pd
import numpy as np
import pickle
import json
import os
import redis
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import joblib
from pathlib import Path

@dataclass
class ModelMetadata:
    """Metadata for the saved model"""
    model_type: str
    version: str
    timestamp: str
    weights: Dict[str, float]
    thresholds: Dict[str, float]
    k_params: Dict[str, any]
    data_stats: Dict[str, int]
    feature_names: List[str]
    model_size_mb: float

class KafkaModelSerializer:
    """
    Optimized model serialization for Kafka real-time predictions
    """
    
    def __init__(self, recommender_instance):
        self.recommender = recommender_instance
        self.logger = logging.getLogger(__name__)
    
    def save_for_kafka(self, model_path: str = "models/hybrid_recommender_kafka/"):
        """
        Save model components optimized for Kafka real-time predictions
        """
        try:
            # Create directory structure
            model_path = Path(model_path)
            model_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("🚀 Preparing model for Kafka deployment...")
            
            # 1. Build optimized lookup dictionaries
            model_components = self._build_lookup_tables()
            
            # 2. Save different formats for different use cases
            self._save_pickle_model(model_path, model_components)
            self._save_redis_model(model_components)  # Optional: for ultra-fast lookup
            self._save_json_model(model_path, model_components)  # For lightweight services
            self._save_numpy_arrays(model_path, model_components)  # For numerical computations
            
            # 3. Save metadata and configuration
            metadata = self._create_metadata(model_components)
            self._save_metadata(model_path, metadata)
            
            # 4. Create deployment scripts
            self._create_deployment_scripts(model_path)
            
            # 5. Test the saved model
            self._test_saved_model(model_path)
            
            self.logger.info(f"✅ Model successfully saved for Kafka at: {model_path}")
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error saving model for Kafka: {str(e)}")
            raise
    
    def _build_lookup_tables(self) -> Dict:
        """Build optimized lookup tables for fast predictions"""
        
        # CF lookup: user_id -> {product_id: score}
        cf_lookup = {}
        if self.recommender.cf_recs is not None:
            self.logger.info("Building CF lookup table...")
            for _, row in self.recommender.cf_recs.iterrows():
                user_id = str(row['user_id'])
                product_id = str(row['product_id'])
                score = float(row.get('cf_score_norm', row.get('predictedRating', 0)))
                
                if user_id not in cf_lookup:
                    cf_lookup[user_id] = {}
                cf_lookup[user_id][product_id] = score
        
        # CB lookup: product_id -> score (aggregated)
        cb_lookup = {}
        if self.recommender.cb_recs is not None:
            self.logger.info("Building CB lookup table...")
            if 'recommended_product_id' in self.recommender.cb_recs.columns:
                # CB format: source_product -> recommended_product
                for _, row in self.recommender.cb_recs.iterrows():
                    product_id = str(row['recommended_product_id'])
                    score = float(row.get('cb_score_norm', row.get('cb_score', 0)))
                    cb_lookup[product_id] = max(cb_lookup.get(product_id, 0), score)
            else:
                # Direct product scores
                cb_lookup = self.recommender.cb_recs.groupby('product_id')['cb_score_norm'].mean().to_dict()
                cb_lookup = {str(k): float(v) for k, v in cb_lookup.items()}
        
        # Product features for content-based similarity (if available)
        product_features = {}
        if self.recommender.metadata is not None:
            self.logger.info("Building product features...")
            feature_columns = [col for col in self.recommender.metadata.columns 
                             if col not in ['product_id'] and self.recommender.metadata[col].dtype in [np.float64, np.int64]]
            
            for _, row in self.recommender.metadata.iterrows():
                product_id = str(row['product_id'])
                features = {col: float(row[col]) for col in feature_columns if pd.notna(row[col])}
                if features:
                    product_features[product_id] = features
        
        # User profiles (aggregated from history)
        user_profiles = {}
        if self.recommender.user_history is not None:
            self.logger.info("Building user profiles...")
            for user_id, user_data in self.recommender.user_history.groupby('user_id'):
                profile = {
                    'avg_rating': float(user_data['rating'].mean()),
                    'total_ratings': int(len(user_data)),
                    'rating_std': float(user_data['rating'].std()) if len(user_data) > 1 else 0.0,
                    'preferred_categories': user_data.get('category', pd.Series()).value_counts().head(5).to_dict() if 'category' in user_data.columns else {},
                    'activity_level': self.recommender.user_activity_levels.get(str(user_id), 'medium')
                }
                user_profiles[str(user_id)] = profile
        
        return {
            'cf_lookup': cf_lookup,
            'cb_lookup': cb_lookup,
            'popularity_scores': {str(k): float(v) for k, v in self.recommender.popularity_scores.items()},
            'user_activity_levels': {str(k): v for k, v in self.recommender.user_activity_levels.items()},
            'user_profiles': user_profiles,
            'product_features': product_features,
            'weights': {
                'cf_weight': float(self.recommender.cf_weight),
                'cb_weight': float(self.recommender.cb_weight),
                'popularity_weight': float(self.recommender.popularity_weight)
            },
            'thresholds': {
                'min_cf_score': float(self.recommender.min_cf_score),
                'min_cb_score': float(self.recommender.min_cb_score),
                'relevance_threshold': float(self.recommender.relevance_threshold)
            },
            'k_params': {
                'min_k': int(self.recommender.min_k),
                'max_k': int(self.recommender.max_k),
                'default_k': int(self.recommender.default_k),
                'use_adaptive_k': bool(self.recommender.use_adaptive_k)
            }
        }
    
    def _save_pickle_model(self, model_path: Path, model_components: Dict):
        """Save as pickle for Python services"""
        pickle_path = model_path / "hybrid_model.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_components, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Also save with joblib for better performance
        joblib_path = model_path / "hybrid_model.joblib"
        joblib.dump(model_components, joblib_path, compress=3)
        
        self.logger.info(f"✅ Saved pickle model: {pickle_path}")
    
    def _save_redis_model(self, model_components: Dict):
        """Save to Redis for ultra-fast lookup (optional)"""
        try:
            # This is optional - only if Redis is available
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            
            # Test connection
            r.ping()
            
            # Store lookups as Redis hashes
            pipe = r.pipeline()
            
            # CF scores: user:product -> score
            for user_id, products in model_components['cf_lookup'].items():
                for product_id, score in products.items():
                    pipe.hset(f"cf:{user_id}", product_id, score)
            
            # CB scores: cb:product_id -> score
            for product_id, score in model_components['cb_lookup'].items():
                pipe.hset("cb_scores", product_id, score)
            
            # Popularity scores
            for product_id, score in model_components['popularity_scores'].items():
                pipe.hset("popularity", product_id, score)
            
            # User profiles
            for user_id, profile in model_components['user_profiles'].items():
                pipe.hset(f"profile:{user_id}", mapping=profile)
            
            # Execute pipeline
            pipe.execute()
            
            # Store configuration
            r.hset("model_config", mapping={
                'weights': json.dumps(model_components['weights']),
                'thresholds': json.dumps(model_components['thresholds']),
                'k_params': json.dumps(model_components['k_params'])
            })
            
            self.logger.info("✅ Saved model to Redis for ultra-fast lookup")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Redis not available, skipping: {str(e)}")
    
    def _save_json_model(self, model_path: Path, model_components: Dict):
        """Save as JSON for lightweight services"""
        # Create a JSON-serializable version (smaller, for microservices)
        json_model = {
            'weights': model_components['weights'],
            'thresholds': model_components['thresholds'],
            'k_params': model_components['k_params'],
            'top_users': dict(list(model_components['cf_lookup'].items())[:1000]),  # Top 1000 users
            'top_products_cb': dict(list(model_components['cb_lookup'].items())[:5000]),  # Top 5000 products
            'top_popularity': dict(list(model_components['popularity_scores'].items())[:5000])
        }
        
        json_path = model_path / "hybrid_model_lite.json"
        with open(json_path, 'w') as f:
            json.dump(json_model, f, indent=2)
        
        self.logger.info(f"✅ Saved lightweight JSON model: {json_path}")
    
    def _save_numpy_arrays(self, model_path: Path, model_components: Dict):
        """Save as numpy arrays for numerical computation"""
        arrays_path = model_path / "arrays"
        arrays_path.mkdir(exist_ok=True)
        
        # Convert lookups to numpy arrays if possible
        if model_components['cf_lookup']:
            # Create user-item matrix (sparse representation)
            users = list(model_components['cf_lookup'].keys())
            all_products = set()
            for products in model_components['cf_lookup'].values():
                all_products.update(products.keys())
            products = list(all_products)
            
            # Save user and product mappings
            np.save(arrays_path / "user_mapping.npy", np.array(users))
            np.save(arrays_path / "product_mapping.npy", np.array(products))
            
            # Create user-item score matrix
            user_item_matrix = np.zeros((len(users), len(products)))
            user_idx = {user: i for i, user in enumerate(users)}
            product_idx = {product: i for i, product in enumerate(products)}
            
            for user_id, user_products in model_components['cf_lookup'].items():
                u_idx = user_idx[user_id]
                for product_id, score in user_products.items():
                    p_idx = product_idx[product_id]
                    user_item_matrix[u_idx, p_idx] = score
            
            np.save(arrays_path / "user_item_matrix.npy", user_item_matrix)
        
        # Save other arrays
        if model_components['popularity_scores']:
            pop_products = list(model_components['popularity_scores'].keys())
            pop_scores = list(model_components['popularity_scores'].values())
            np.save(arrays_path / "popularity_products.npy", np.array(pop_products))
            np.save(arrays_path / "popularity_scores.npy", np.array(pop_scores))
        
        self.logger.info(f"✅ Saved numpy arrays: {arrays_path}")
    
    def _create_metadata(self, model_components: Dict) -> ModelMetadata:
        """Create comprehensive model metadata"""
        
        # Calculate model size
        import sys
        model_size_mb = sys.getsizeof(json.dumps(model_components, default=str)) / (1024 * 1024)
        
        return ModelMetadata(
            model_type="hybrid_recommender",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            weights=model_components['weights'],
            thresholds=model_components['thresholds'],
            k_params=model_components['k_params'],
            data_stats={
                'cf_users': len(model_components['cf_lookup']),
                'cf_total_pairs': sum(len(products) for products in model_components['cf_lookup'].values()),
                'cb_products': len(model_components['cb_lookup']),
                'popularity_products': len(model_components['popularity_scores']),
                'user_profiles': len(model_components['user_profiles']),
                'product_features': len(model_components['product_features'])
            },
            feature_names=['cf_score', 'cb_score', 'popularity_score', 'hybrid_score'],
            model_size_mb=model_size_mb
        )
    
    def _save_metadata(self, model_path: Path, metadata: ModelMetadata):
        """Save model metadata"""
        metadata_path = model_path / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Also save as pickle for Python
        with open(model_path / "model_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"✅ Saved metadata: {metadata_path}")
    
    def _create_deployment_scripts(self, model_path: Path):
        """Create deployment and testing scripts"""
        
        # Kafka producer script
        producer_script = '''
import json
import pickle
from kafka import KafkaProducer
from kafka_model_loader import KafkaHybridPredictor

class RecommendationProducer:
    def __init__(self, model_path="models/hybrid_recommender_kafka/"):
        self.predictor = KafkaHybridPredictor(model_path)
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def send_recommendation(self, user_id, candidate_products, top_k=10):
        predictions = self.predictor.predict(user_id, candidate_products, top_k)
        
        recommendation_event = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'recommendations': predictions,
            'model_version': self.predictor.metadata.version
        }
        
        self.producer.send('recommendations', recommendation_event)
        return recommendation_event
'''
        
        with open(model_path / "kafka_producer.py", 'w') as f:
            f.write(producer_script)
        
        # Model loader script
        loader_script = '''
import pickle
import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import logging

class KafkaHybridPredictor:
    """
    Optimized predictor for Kafka real-time recommendations
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.logger = logging.getLogger(__name__)
        
        # Load model components
        self.model_components = self._load_model()
        self.metadata = self._load_metadata()
        
        # Extract frequently used components
        self.cf_lookup = self.model_components['cf_lookup']
        self.cb_lookup = self.model_components['cb_lookup']
        self.popularity_scores = self.model_components['popularity_scores']
        self.weights = self.model_components['weights']
        self.k_params = self.model_components['k_params']
        
        self.logger.info(f"✅ Model loaded: {len(self.cf_lookup)} users, {len(self.cb_lookup)} products")
    
    def _load_model(self) -> Dict:
        """Load model with fallback options"""
        try:
            # Try joblib first (faster)
            joblib_path = self.model_path / "hybrid_model.joblib"
            if joblib_path.exists():
                import joblib
                return joblib.load(joblib_path)
        except:
            pass
        
        # Fallback to pickle
        with open(self.model_path / "hybrid_model.pkl", 'rb') as f:
            return pickle.load(f)
    
    def _load_metadata(self):
        """Load model metadata"""
        with open(self.model_path / "model_metadata.pkl", 'rb') as f:
            return pickle.load(f)
    
    def predict(self, user_id: str, candidate_products: List[str], top_k: int = 10) -> List[Dict]:
        """
        Fast prediction for real-time use
        """
        user_id = str(user_id)
        predictions = []
        
        # Get adaptive K
        actual_k = self._get_adaptive_k(user_id, top_k)
        
        # Get user's CF scores
        user_cf_scores = self.cf_lookup.get(user_id, {})
        
        # Calculate hybrid scores
        for product_id in candidate_products:
            product_id = str(product_id)
            
            # Get component scores
            cf_score = user_cf_scores.get(product_id, 0.0)
            cb_score = self.cb_lookup.get(product_id, 0.0)
            popularity_score = self.popularity_scores.get(product_id, 0.5)
            
            # Calculate hybrid score
            hybrid_score = (
                self.weights['cf_weight'] * cf_score +
                self.weights['cb_weight'] * cb_score +
                self.weights['popularity_weight'] * popularity_score
            )
            
            predictions.append({
                'product_id': product_id,
                'hybrid_score': hybrid_score,
                'cf_score': cf_score,
                'cb_score': cb_score,
                'popularity_score': popularity_score,
                'confidence': self._calculate_confidence(hybrid_score)
            })
        
        # Sort and return top K
        predictions.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return predictions[:actual_k]
    
    def _get_adaptive_k(self, user_id: str, requested_k: int) -> int:
        """Get adaptive K based on user activity"""
        if not self.k_params.get('use_adaptive_k', False):
            return requested_k
        
        activity_level = self.model_components.get('user_activity_levels', {}).get(user_id, 'medium')
        
        if activity_level == 'low':
            return min(requested_k, self.k_params['min_k'])
        elif activity_level == 'high':
            return min(requested_k, self.k_params['max_k'])
        else:
            return min(requested_k, self.k_params['default_k'])
    
    def _calculate_confidence(self, score: float) -> str:
        """Calculate confidence level"""
        if score >= 0.8:
            return "very_high"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile information"""
        return self.model_components.get('user_profiles', {}).get(str(user_id))
    
    def batch_predict(self, requests: List[Dict]) -> List[Dict]:
        """Batch prediction for multiple users"""
        results = []
        for request in requests:
            predictions = self.predict(
                request['user_id'],
                request['candidate_products'],
                request.get('top_k', 10)
            )
            results.append({
                'user_id': request['user_id'],
                'predictions': predictions
            })
        return results
'''
        
        with open(model_path / "kafka_model_loader.py", 'w') as f:
            f.write(loader_script)
        
        # Test script
        test_script = '''
from kafka_model_loader import KafkaHybridPredictor
import time

def test_model_performance():
    """Test model loading and prediction performance"""
    
    # Load model
    start_time = time.time()
    predictor = KafkaHybridPredictor(".")
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.3f} seconds")
    
    # Test prediction
    sample_user = list(predictor.cf_lookup.keys())[0] if predictor.cf_lookup else "test_user"
    sample_products = list(predictor.cb_lookup.keys())[:20] if predictor.cb_lookup else ["prod1", "prod2"]
    
    start_time = time.time()
    predictions = predictor.predict(sample_user, sample_products, top_k=10)
    pred_time = time.time() - start_time
    
    print(f"✅ Prediction completed in {pred_time*1000:.2f} ms")
    print(f"📊 Sample predictions for user {sample_user}:")
    for i, pred in enumerate(predictions[:3]):
        print(f"   {i+1}. Product {pred['product_id']}: {pred['hybrid_score']:.4f} (confidence: {pred['confidence']})")
    
    # Test batch prediction
    batch_requests = [
        {'user_id': sample_user, 'candidate_products': sample_products[:10], 'top_k': 5}
        for _ in range(10)
    ]
    
    start_time = time.time()
    batch_results = predictor.batch_predict(batch_requests)
    batch_time = time.time() - start_time
    
    print(f"✅ Batch prediction (10 users) completed in {batch_time*1000:.2f} ms")
    print(f"⚡ Average time per user: {batch_time*1000/10:.2f} ms")

if __name__ == "__main__":
    test_model_performance()
'''
        
        with open(model_path / "test_model.py", 'w') as f:
            f.write(test_script)
        
        self.logger.info("✅ Created deployment scripts")
    
    def _test_saved_model(self, model_path: Path):
        """Test the saved model"""
        try:
            # Test loading
            with open(model_path / "hybrid_model.pkl", 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Basic validation
            required_keys = ['cf_lookup', 'cb_lookup', 'popularity_scores', 'weights']
            for key in required_keys:
                assert key in loaded_model, f"Missing key: {key}"
            
            # Test a simple prediction
            if loaded_model['cf_lookup']:
                sample_user = list(loaded_model['cf_lookup'].keys())[0]
                sample_products = list(loaded_model['cb_lookup'].keys())[:5] if loaded_model['cb_lookup'] else []
                
                if sample_products:
                    # Quick prediction test
                    weights = loaded_model['weights']
                    user_cf = loaded_model['cf_lookup'].get(sample_user, {})
                    
                    for product in sample_products:
                        cf_score = user_cf.get(product, 0.0)
                        cb_score = loaded_model['cb_lookup'].get(product, 0.0)
                        pop_score = loaded_model['popularity_scores'].get(product, 0.5)
                        
                        hybrid_score = (
                            weights['cf_weight'] * cf_score +
                            weights['cb_weight'] * cb_score +
                            weights['popularity_weight'] * pop_score
                        )
                        
                        assert 0 <= hybrid_score <= 1.1, f"Invalid hybrid score: {hybrid_score}"
            
            self.logger.info("✅ Model validation passed")
            
        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {str(e)}")
            raise


# Usage example
def save_model_for_kafka_deployment(recommender_instance):
    """
    Main function to save your trained model for Kafka
    """
    serializer = KafkaModelSerializer(recommender_instance)
    model_path = serializer.save_for_kafka()
    
    print(f"\n🎉 Model successfully prepared for Kafka!")
    print(f"📁 Saved to: {model_path}")
    print(f"\n📋 Files created:")
    print(f"   • hybrid_model.pkl - Main model (pickle)")
    print(f"   • hybrid_model.joblib - Optimized model (joblib)")
    print(f"   • hybrid_model_lite.json - Lightweight version")
    print(f"   • model_metadata.json - Model information")
    print(f"   • kafka_producer.py - Kafka producer script")
    print(f"   • kafka_model_loader.py - Model loader for Kafka")
    print(f"   • test_model.py - Performance testing script")
    print(f"   • arrays/ - Numpy arrays for numerical computation")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Copy the model folder to your Kafka service")
    print(f"   2. Use kafka_model_loader.py in your Kafka consumer")
    print(f"   3. Run test_model.py to verify performance")
    print(f"   4. Configure your streaming pipeline")
    
    return model_path

# Integration with your existing code
def add_to_existing_recommender():
    """
    Add this method to your existing EnhancedHybridRecommender class
    """
    def save_for_kafka_deployment(self, model_path: str = "models/hybrid_recommender_kafka/"):
        """Save model optimized for Kafka real-time predictions"""
        serializer = KafkaModelSerializer(self)
        return serializer.save_for_kafka(model_path)
    
    return save_for_kafka_deployment