import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config.settings import DATA_PATHS

@st.cache_data
def load_all_data():
    """Load and preprocess all datasets"""
    try:
        # Load datasets
        user_ratings = pd.read_csv(DATA_PATHS['user_ratings'])
        product_ratings = pd.read_csv(DATA_PATHS['product_ratings'])
        transactions = pd.read_csv(DATA_PATHS['transactions'])
        
        # Process user ratings
        user_ratings = process_user_ratings(user_ratings)
        
        return user_ratings, product_ratings, transactions
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

def process_user_ratings(user_ratings):
    """Process user ratings data"""
    # Convert Unix timestamps to datetime
    if 'time' in user_ratings.columns:
        user_ratings['date'] = pd.to_datetime(user_ratings['time'], unit='s')
        user_ratings['formatted_date'] = user_ratings['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Check required columns
    required_cols = ['userId', 'productId', 'rating']
    missing_cols = [col for col in required_cols if col not in user_ratings.columns]
    if missing_cols:
        st.error(f"Missing columns in user_ratings: {', '.join(missing_cols)}")
        st.stop()
    
    # Normalize ratings
    scaler = MinMaxScaler()
    user_ratings['normalized_rating'] = scaler.fit_transform(user_ratings[['rating']])
    
    return user_ratings

@st.cache_data
def load_recommendations_data(rec_type):
    """Load specific recommendation data"""
    try:
        if rec_type == 'als':
            return pd.read_csv(DATA_PATHS['als_recommendations'])
        elif rec_type == 'content':
            return pd.read_csv(DATA_PATHS['content_recommendations'])
        elif rec_type == 'hybrid':
            return pd.read_csv(DATA_PATHS['hybrid_recommendations'])
    except FileNotFoundError:
        return None