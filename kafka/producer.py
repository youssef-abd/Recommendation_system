from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import csv
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RatingProducer:
    def __init__(self, kafka_server='localhost:9092'):
        self.kafka_server = kafka_server
        self.topic = 'user-ratings'
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[kafka_server],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=3,  # Retry failed sends
                acks='all',  # Wait for all replicas to acknowledge
                request_timeout_ms=10000,  # 10 second timeout
            )
            logger.info(f"Successfully connected to Kafka at {kafka_server}")
            
            # Test connection by getting metadata
            self.producer.bootstrap_connected()
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka at {kafka_server}")
            logger.error(f"Error: {e}")
            logger.error("\nPlease make sure Kafka is running:")
            logger.error("1. Start Zookeeper: bin/zookeeper-server-start.sh config/zookeeper.properties")
            logger.error("2. Start Kafka: bin/kafka-server-start.sh config/server.properties")
            logger.error("3. Or use Docker: docker-compose up -d")
            raise
    
    def send_rating(self, user_id, product_id, rating, normalized_rating=None):
        """Send a single rating to Kafka with error handling"""
        try:
            # Calculate normalized rating if not provided
            if normalized_rating is None:
                normalized_rating = float(rating) / 5.0
            
            message = {
                'userId': str(user_id),
                'productId': str(product_id),
                'rating': float(rating),
                'normalized_rating': float(normalized_rating),
                'timestamp': datetime.now().isoformat()
            }
            
            # Send message asynchronously but get future for error handling
            future = self.producer.send(self.topic, value=message)
            
            # Wait for send to complete (with timeout)
            record_metadata = future.get(timeout=10)
            
            logger.info(f"Successfully sent: {user_id} rated {product_id} with {rating}")
            logger.debug(f"Message sent to topic: {record_metadata.topic}, partition: {record_metadata.partition}, offset: {record_metadata.offset}")
            
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error sending rating: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending rating: {e}")
            return False
    
    def send_ratings_from_csv(self, csv_file_path, delay=0.1, skip_header=True):
        """Send ratings from your CSV file with improved error handling"""
        if not csv_file_path:
            logger.error("CSV file path is required")
            return 0
        
        successful_sends = 0
        failed_sends = 0
        
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                
                # Skip header row if requested
                if skip_header:
                    try:
                        header = next(reader)
                        logger.info(f"Skipping header row: {header}")
                    except StopIteration:
                        logger.warning("CSV file appears to be empty")
                        return 0
                
                for row_num, row in enumerate(reader, start=2 if skip_header else 1):
                    if len(row) >= 3:  # Minimum: user_id, product_id, rating
                        try:
                            user_id = row[0].strip()
                            product_id = row[1].strip()
                            rating = float(row[2])
                            
                            # Use normalized rating from CSV if available
                            normalized_rating = float(row[3]) if len(row) > 3 else None
                            
                            if self.send_rating(user_id, product_id, rating, normalized_rating):
                                successful_sends += 1
                            else:
                                failed_sends += 1
                                
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Skipping invalid row {row_num}: {row} - Error: {e}")
                            failed_sends += 1
                            continue
                    else:
                        logger.warning(f"Skipping incomplete row {row_num}: {row}")
                        failed_sends += 1
                    
                    # Add delay between sends
                    if delay > 0:
                        time.sleep(delay)
                        
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_file_path}")
            return 0
        except PermissionError:
            logger.error(f"Permission denied reading CSV file: {csv_file_path}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error reading CSV: {e}")
            return 0
        
        logger.info(f"CSV processing complete: {successful_sends} successful, {failed_sends} failed")
        return successful_sends
    
    def send_new_rating(self, user_id, product_id, rating):
        """Send a new rating (calculate normalized rating automatically)"""
        if not all([user_id, product_id, rating]):
            logger.error("user_id, product_id, and rating are all required")
            return False
        
        try:
            rating = float(rating)
            if not (0 <= rating <= 5):
                logger.warning(f"Rating {rating} is outside typical 0-5 range")
        except ValueError:
            logger.error(f"Invalid rating value: {rating}")
            return False
        
        return self.send_rating(user_id, product_id, rating)
    
    def send_multiple_ratings(self, ratings_list, delay=0.1):
        """Send multiple ratings from a list of dictionaries"""
        if not ratings_list:
            logger.warning("No ratings provided")
            return 0
        
        successful_sends = 0
        
        for i, rating_data in enumerate(ratings_list):
            try:
                required_fields = ['user_id', 'product_id', 'rating']
                if not all(field in rating_data for field in required_fields):
                    logger.warning(f"Skipping rating {i}: missing required fields {required_fields}")
                    continue
                
                if self.send_rating(
                    rating_data['user_id'],
                    rating_data['product_id'],
                    rating_data['rating'],
                    rating_data.get('normalized_rating')
                ):
                    successful_sends += 1
                    
            except Exception as e:
                logger.error(f"Error processing rating {i}: {e}")
                continue
            
            if delay > 0:
                time.sleep(delay)
        
        logger.info(f"Sent {successful_sends}/{len(ratings_list)} ratings successfully")
        return successful_sends
    
    def test_connection(self):
        """Test if Kafka connection is working"""
        try:
            # Try to get topic metadata
            metadata = self.producer.bootstrap_connected()
            logger.info("Kafka connection test successful")
            return True
        except Exception as e:
            logger.error(f"Kafka connection test failed: {e}")
            return False
    
    def close(self):
        """Close the producer with proper cleanup"""
        try:
            # Flush any pending messages
            self.producer.flush(timeout=10)
            logger.info("Flushed pending messages")
            
            # Close the producer
            self.producer.close(timeout=10)
            logger.info("Rating producer closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing producer: {e}")

# Example usage
if __name__ == "__main__":
    producer = None
    
    try:
        # Initialize producer
        producer = RatingProducer()
        
        # Test connection
        if not producer.test_connection():
            logger.error("Cannot proceed - Kafka connection failed")
            exit(1)
        
        # Example 1: Send a single new rating
        logger.info("=== Sending single rating ===")
        success = producer.send_new_rating("USER123", "PRODUCT456", 4.5)
        if success:
            logger.info("Single rating sent successfully")
        
        # Example 2: Send multiple ratings
        logger.info("=== Sending multiple ratings ===")
        sample_ratings = [
            {'user_id': 'A10056VD1UVVB2', 'product_id': 'B001PRODUCT', 'rating': 5.0},
            {'user_id': 'A10056VD1UVVB2', 'product_id': 'B002PRODUCT', 'rating': 3.5},
            {'user_id': 'USER456', 'product_id': 'B000NOAY6I', 'rating': 4.0},
        ]
        producer.send_multiple_ratings(sample_ratings, delay=0.5)
        
        # Example 3: Send ratings from CSV file
        logger.info("=== Sending ratings from CSV ===")
        # Replace with your actual CSV file path
        csv_file = "data/processed/clean/cleaned_user_ratings.csv"  # Change this to your CSV file path
        
        # Check if file exists before trying to read
        import os
        if os.path.exists(csv_file):
            successful_sends = producer.send_ratings_from_csv(csv_file, delay=0.2)
            logger.info(f"CSV processing completed: {successful_sends} ratings sent")
        else:
            logger.warning(f"CSV file not found: {csv_file}")
            logger.info("Create a CSV file with format: user_id,product_id,rating,normalized_rating")
            logger.info("Example:")
            logger.info("USER123,PRODUCT456,4.5,0.9")
            logger.info("USER124,PRODUCT789,3.0,0.6")
        
        # Give time for all messages to be sent
        logger.info("Waiting for messages to be sent...")
        time.sleep(2)
        
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if producer:
            producer.close()