from kafka import KafkaConsumer, KafkaProducer 
import json
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import time
import signal
import threading

from hybrid_saved import EnhancedHybridRecommender

class RecommendationConsumer:
    def __init__(self, kafka_server='localhost:9092', model_path='models/hybrid_model.pkl'):
        # Kafka consumer setup - FIXED: Better configuration for testing
        self.consumer = KafkaConsumer(
            'user-ratings',
            bootstrap_servers=[kafka_server],
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest',  # CHANGED: Get all messages for testing
            group_id='hybrid-recommender-consumer',  # CHANGED: Different group ID
            consumer_timeout_ms=5000,  # CHANGED: Shorter timeout
            enable_auto_commit=True,
            session_timeout_ms=10000
        )
        
        # Kafka producer setup
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_server],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('HybridRecommenderConsumer')
        
        # Error tracking
        self.last_error_time = 0
        self.error_count = 0
        
        # Shutdown flag
        self.shutdown_flag = threading.Event()
        
        # Model loading
        self.model_path = self._resolve_model_path(model_path)
        self.recommender = self._load_model_with_fallback()
        
        if not self.recommender:
            self.logger.critical("Failed to load recommender model - exiting")
            sys.exit(1)
            
        self.logger.info("Recommendation consumer initialized successfully")

    def _resolve_model_path(self, model_path):
        """Find the model file in common locations"""
        if model_path and os.path.exists(model_path):
            return os.path.abspath(model_path)
            
        possible_locations = [
            'models/hybrid_model.pkl',
            '../models/hybrid_model.pkl',
            os.path.join(Path(__file__).parent.parent, 'models', 'hybrid_model.pkl'),
            os.path.join('Recommender_system', 'models', 'hybrid_model.pkl')
        ]
        
        for path in possible_locations:
            if os.path.exists(path):
                abs_path = os.path.abspath(path)
                self.logger.info(f"Found model at: {abs_path}")
                return abs_path
                
        self.logger.error("Could not locate model file in any of these locations:")
        for path in possible_locations:
            self.logger.error(f"- {os.path.abspath(path)}")
        
        return model_path

    def _load_model_with_fallback(self):
        """Attempt multiple loading strategies"""
        try:
            if not os.path.exists(self.model_path):
                self.log_error(f"Model file not found at: {self.model_path}")
                return None
                
            file_size = os.path.getsize(self.model_path)
            if file_size == 0:
                self.log_error("Model file is empty (0 bytes)")
                return None
                
            self.logger.info(f"Loading model ({file_size} bytes) from: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                loaded = pickle.load(f)
            
            if isinstance(loaded, EnhancedHybridRecommender):
                self.logger.info("Loaded complete model instance")
                return loaded
                
            if isinstance(loaded, dict):
                self.logger.info("Reconstructing from state dictionary")
                model = EnhancedHybridRecommender()
                required = [
                    'cf_weight', 'cb_weight', 'popularity_weight',
                    'popularity_scores', 'user_activity_levels',
                    'cb_lookup', 'is_trained'
                ]
                for attr in required:
                    if attr not in loaded:
                        raise ValueError(f"Missing required attribute: {attr}")
                    setattr(model, attr, loaded[attr])
                optional = [
                    'user_profiles', 'product_features',
                    'training_timestamp', 'model_version',
                    'evaluation_results'
                ]
                for attr in optional:
                    if attr in loaded:
                        setattr(model, attr, loaded[attr])
                return model
                
            try:
                model = EnhancedHybridRecommender()
                model.__dict__.update(loaded.__dict__ if hasattr(loaded, '__dict__') else loaded)
                return model
            except Exception as e:
                self.log_error(f"Direct transfer failed: {str(e)}")
                raise ValueError("Unknown model format") from e
                
        except Exception as e:
            self.log_error(f"Model loading failed: {str(e)}", exc_info=True)
            return None

    def log_error(self, message, exc_info=False):
        current_time = time.time()
        if current_time - self.last_error_time > 60:
            self.error_count = 0
        if self.error_count < 5:
            self.logger.error(message, exc_info=exc_info)
        self.error_count += 1
        self.last_error_time = current_time

    def get_recommendations(self, user_id, num_recommendations=10):
        if not self.recommender:
            self.log_error("No recommender model available - returning fallback")
            return self.get_fallback_recommendations(user_id, num_recommendations)
        
        try:
            recommendations = self.recommender.predict_for_user(
                user_id=user_id,
                top_k=num_recommendations,
                exclude_seen=True
            )
            
            if recommendations.empty:
                self.logger.info(f"No recommendations generated for user {user_id} - using fallback")
                return self.get_fallback_recommendations(user_id, num_recommendations)
            
            formatted_recs = []
            for _, row in recommendations.iterrows():
                rec = {
                    'product_id': str(row['product_id']),  # FIXED: Consistent naming
                    'score': float(row['hybrid_score']),
                    'cf_score': float(row.get('cf_score', 0)),
                    'cb_score': float(row.get('cb_score', 0)),
                    'popularity_score': float(row.get('popularity_score', 0.5)),
                    'type': 'hybrid',
                    'explanation': 'Recommended based on your preferences',
                    'confidence': 'medium'
                }
                formatted_recs.append(rec)
            
            self.logger.info(f"Generated {len(formatted_recs)} recommendations for user {user_id}")
            return formatted_recs
            
        except Exception as e:
            self.log_error(f"Recommendation error for {user_id}: {str(e)}", exc_info=True)
            return self.get_fallback_recommendations(user_id, num_recommendations)

    def get_fallback_recommendations(self, user_id, num_recommendations=10):
        if not self.recommender or not hasattr(self.recommender, 'popularity_scores'):
            self.log_error("No recommender or popularity scores available for fallback")
            # Return some dummy recommendations for testing
            return [{
                'product_id': f'DUMMY_PRODUCT_{i}',
                'score': 0.5 - (i * 0.05),
                'cf_score': 0.0,
                'cb_score': 0.0,
                'popularity_score': 0.5 - (i * 0.05),
                'type': 'dummy_fallback',
                'explanation': 'Dummy recommendation for testing',
                'confidence': 'low'
            } for i in range(min(num_recommendations, 5))]
        
        try:
            popularity_items = sorted(
                self.recommender.popularity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:num_recommendations]
            
            return [{
                'product_id': str(product_id),  # FIXED: Consistent naming
                'score': float(score),
                'cf_score': 0.0,
                'cb_score': 0.0,
                'popularity_score': float(score),
                'type': 'popularity_fallback',
                'explanation': 'Popular item recommended as fallback',
                'confidence': 'medium'
            } for product_id, score in popularity_items]
            
        except Exception as e:
            self.log_error(f"Fallback recommendation error: {str(e)}", exc_info=True)
            return []

    def process_rating_message(self, rating_data):
        try:
            # FIXED: Handle both userId and user_id formats
            user_id = str(rating_data.get('userId') or rating_data.get('user_id', ''))
            if not user_id:
                raise ValueError("Missing user ID in rating data")
            
            # FIXED: Handle both productId and product_id formats
            product_id = str(rating_data.get('productId') or rating_data.get('product_id', ''))
            rating = float(rating_data.get('rating', 0))
            timestamp = rating_data.get('timestamp', datetime.now().isoformat())
            
            self.logger.info(
                f"Processing rating: user={user_id}, product={product_id}, "
                f"rating={rating}, timestamp={timestamp}"
            )
            
            # Generate recommendations immediately
            recommendations = self.get_recommendations(user_id)
            
            if recommendations:
                # Send recommendations
                self._send_recommendations(user_id, recommendations, timestamp)
                return True
            else:
                self.logger.warning(f"No recommendations generated for user {user_id}")
                return False
            
        except Exception as e:
            self.log_error(f"Rating processing failed: {str(e)}", exc_info=True)
            return False

    def _send_recommendations(self, user_id, recommendations, timestamp):
        try:
            message = {
                'user_id': user_id,  # FIXED: Consistent naming
                'recommendations': recommendations,
                'timestamp': timestamp,
                'trigger': 'new_rating',
                'model_version': getattr(self.recommender, 'model_version', 'v1.0'),
                'model_timestamp': getattr(self.recommender, 'training_timestamp', 'unknown'),
                'generated_at': datetime.now().isoformat()  # ADDED: Generation timestamp
            }
            
            # Send to recommendations topic
            future = self.producer.send('recommendations', value=message)
            record_metadata = future.get(timeout=10)  # Wait for confirmation
            
            self.logger.info(
                f"✅ Sent {len(recommendations)} recommendations for user {user_id} "
                f"to topic '{record_metadata.topic}', partition {record_metadata.partition}, "
                f"offset {record_metadata.offset}"
            )
            
        except Exception as e:
            self.log_error(f"Failed to send recommendations: {str(e)}", exc_info=True)

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum} - initiating shutdown")
            self.shutdown_flag.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start_consuming(self):
        self.logger.info("🚀 Starting recommendation consumer...")
        
        if not self.recommender:
            self.logger.critical("No recommender model available - cannot start")
            return
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Test the recommendation system
        self._run_startup_tests()
        
        self.logger.info("🔄 Listening for rating messages...")
        
        try:
            message_count = 0
            for message in self.consumer:
                # Check for shutdown signal
                if self.shutdown_flag.is_set():
                    self.logger.info("Shutdown signal received - stopping consumer")
                    break
                
                message_count += 1
                self.logger.info(f"📨 Message #{message_count} received from topic '{message.topic}'")
                
                try:
                    success = self.process_rating_message(message.value)
                    if success:
                        self.logger.info(f"✅ Successfully processed message #{message_count}")
                    else:
                        self.logger.warning(f"⚠️ Failed to process message #{message_count}")
                        
                except Exception as e:
                    self.log_error(f"Message processing failed: {str(e)}", exc_info=True)
                    
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal - shutting down")
        except Exception as e:
            self.logger.critical(f"Fatal consumer error: {str(e)}", exc_info=True)
        finally:
            self.close()

    def _run_startup_tests(self):
        """Run tests to verify the system works"""
        self.logger.info("🧪 Running startup tests...")
        
        # Test 1: Check if we have any users to test with
        if hasattr(self.recommender, 'user_activity_levels') and self.recommender.user_activity_levels:
            test_users = list(self.recommender.user_activity_levels.keys())[:3]
            
            for user_id in test_users:
                test_recs = self.get_recommendations(user_id, 3)
                self.logger.info(f"✅ Test user {user_id}: {len(test_recs)} recommendations generated")
                
                # Send a test recommendation to verify Kafka producer works
                if test_recs:
                    try:
                        test_message = {
                            'user_id': user_id,
                            'recommendations': test_recs[:2],  # Send just 2 for testing
                            'timestamp': datetime.now().isoformat(),
                            'trigger': 'startup_test',
                            'model_version': 'test',
                            'model_timestamp': 'test'
                        }
                        
                        future = self.producer.send('recommendations', value=test_message)
                        future.get(timeout=5)
                        self.logger.info(f"✅ Test message sent for user {user_id}")
                        break  # Only send one test message
                        
                    except Exception as e:
                        self.logger.error(f"❌ Test message failed: {e}")
        else:
            self.logger.warning("⚠️ No user activity data found for testing")
        
        self.logger.info("🧪 Startup tests completed")

    def close(self):
        self.logger.info("Shutting down consumer...")
        
        try:
            if hasattr(self, 'consumer'):
                self.consumer.close()
                self.logger.info("Kafka consumer closed")
        except Exception as e:
            self.logger.error(f"Error closing consumer: {str(e)}")
        
        try:
            if hasattr(self, 'producer'):
                self.producer.flush()  # Flush pending messages
                self.producer.close()
                self.logger.info("Kafka producer closed")
        except Exception as e:
            self.logger.error(f"Error closing producer: {str(e)}")
        
        self.logger.info("Consumer shutdown complete")

if __name__ == "__main__":
    consumer = None
    try:
        consumer = RecommendationConsumer(
            kafka_server='localhost:9092',
            model_path='hybrid_test_output/enhanced_hybrid_model.pkl'
        )
        consumer.start_consuming()
    except Exception as e:
        logging.critical(f"Failed to start consumer: {str(e)}", exc_info=True)
    finally:
        if consumer:
            consumer.close()