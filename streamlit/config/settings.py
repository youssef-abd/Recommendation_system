# Page configuration
PAGE_CONFIG = {
    "page_title": "E-commerce Analytics Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Sidebar options
SIDEBAR_OPTIONS = [
    "📊 Dashboard Overview",
    "👤 User Analysis", 
    "📦 Product Analysis",
    "💳 Transactions",
    "📈 Visualizations",
    "🎯 Recommendation Systems"
]

# Data file paths
DATA_PATHS = {
    'user_ratings': 'data/processed/clean/cleaned_user_ratings.csv',
    'product_ratings': 'data/processed/clean/cleaned_product_ratings2.csv',
    'transactions': 'data/processed/clean/cleaned_transactions.csv',
    'interaction_matrix': 'data/processed/interaction_matrix_sparse.npz',
    'als_recommendations': 'data/processed/recommendations/advanced_als_recommendations/als_recs.csv',
    'content_recommendations': 'data/processed/content_based_recommendations/content_based_recommendations.csv',
    'hybrid_recommendations': 'data/processed/hybrid_recommendations/hybrid_recommendations.csv'
}

# Default values
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_NUM_RECOMMENDATIONS = 10