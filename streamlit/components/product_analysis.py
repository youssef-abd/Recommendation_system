import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for enhanced styling
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def render(product_ratings, user_ratings):
    """Enhanced main render function with improved layout and features"""
    load_custom_css()
    
    # Main header with enhanced styling
    st.markdown('<h1 class="main-header">🛒 Product Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar with enhanced controls
    render_sidebar_controls(product_ratings, user_ratings)
    
    # Main content tabs (only keeping Product Search and Analytics Overview)
    tab1, tab2 = st.tabs(["🔍 Product Search", "📊 Analytics Overview"])
    
    with tab1:
        render_enhanced_product_search(product_ratings, user_ratings)
    
    with tab2:
        render_analytics_overview(product_ratings, user_ratings)

def render_sidebar_controls(product_ratings, user_ratings):
    """Enhanced sidebar with filtering and control options"""
    st.sidebar.markdown("### 🎛️ Dashboard Controls")
    
    # Data overview in sidebar
    with st.sidebar.expander("📊 Data Overview", expanded=False):
        st.metric("Total Products", len(product_ratings['productId'].unique()) if 'productId' in product_ratings.columns else 0)
        st.metric("Total Users", len(user_ratings['userId'].unique()) if 'userId' in user_ratings.columns else 0)
        st.metric("Total Ratings", len(user_ratings))
    
    # Filtering options
    st.sidebar.markdown("### 🔧 Filters")
    
    # Price range filter
    if 'price' in product_ratings.columns:
        price_range = st.sidebar.slider(
            "Price Range ($)",
            min_value=float(product_ratings['price'].min()),
            max_value=float(product_ratings['price'].max()),
            value=(float(product_ratings['price'].min()), float(product_ratings['price'].max())),
            step=0.01
        )
        st.session_state['price_filter'] = price_range
    
    # Rating filter
    if 'rating' in user_ratings.columns:
        rating_filter = st.sidebar.multiselect(
            "Rating Filter",
            options=sorted(user_ratings['rating'].unique()),
            default=sorted(user_ratings['rating'].unique())
        )
        st.session_state['rating_filter'] = rating_filter
    
    # Visualization preferences
    st.sidebar.markdown("### 🎨 Visualization Settings")
    chart_style = st.sidebar.selectbox(
        "Chart Style",
        ["Modern", "Classic", "Minimal"],
        index=0
    )
    st.session_state['chart_style'] = chart_style
    
    color_scheme = st.sidebar.selectbox(
        "Color Scheme",
        ["Default", "Viridis", "Plasma", "Inferno"],
        index=0
    )
    st.session_state['color_scheme'] = color_scheme

def render_enhanced_product_search(product_ratings, user_ratings):
    """Enhanced product search with autocomplete and advanced features"""
    st.markdown("## 🔍 Intelligent Product Search")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced search with autocomplete
        if 'productId' in product_ratings.columns:
            available_products = product_ratings['productId'].unique().tolist()
            
            # Search input with suggestions
            search_term = st.text_input(
                "🔍 Search Product ID",
                placeholder="Start typing to see suggestions...",
                help="Enter a product ID or use the selectbox below for suggestions"
            )
            
            # Dropdown for product selection
            selected_product = st.selectbox(
                "Or select from available products:",
                options=[""] + available_products[:100],  # Limit for performance
                format_func=lambda x: f"Product {x}" if x else "Select a product..."
            )
            
            product_id = selected_product if selected_product else search_term
    
    with col2:
        # Quick stats
        if 'productId' in product_ratings.columns:
            st.markdown("### 📊 Quick Stats")
            total_products = len(product_ratings['productId'].unique())
            st.metric("Available Products", f"{total_products:,}")
            
            if 'price' in product_ratings.columns:
                avg_price = product_ratings['price'].mean()
                st.metric("Average Price", f"${avg_price:.2f}")
    
    # Enhanced product analysis
    if product_id:
        analyze_product_comprehensive(product_id, product_ratings, user_ratings)

def analyze_product_comprehensive(product_id, product_ratings, user_ratings):
    """Comprehensive product analysis with enhanced visualizations"""
    with st.spinner("🔄 Analyzing product data..."):
        if 'productId' not in product_ratings.columns:
            st.error("❌ Product ID column not found in product data")
            return
        
        product_data = product_ratings[product_ratings['productId'] == product_id]
        product_user_ratings = user_ratings[user_ratings['productId'] == product_id]
        
        if not product_data.empty:
            # Success message with enhanced styling
            st.markdown(f'<div class="success-box">✅ Successfully found product <strong>{product_id}</strong></div>', 
                       unsafe_allow_html=True)
            
            # Enhanced metrics display
            display_enhanced_product_metrics(product_data, product_user_ratings)
            
            # Interactive visualizations
            display_interactive_product_charts(product_data, product_user_ratings, product_id)
            
            # User engagement analysis
            display_user_engagement_analysis(product_user_ratings, product_id)
                
        else:
            st.markdown(f'<div class="warning-box">⚠️ Product <strong>{product_id}</strong> not found. Please verify the ID and try again.</div>', 
                       unsafe_allow_html=True)

def display_enhanced_product_metrics(product_data, product_user_ratings):
    """Enhanced metrics display with better visual design"""
    st.markdown("### 📊 Product Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'price' in product_data.columns:
            avg_price = product_data['price'].mean()
            price_std = product_data['price'].std()
            st.metric(
                "💰 Average Price", 
                f"${avg_price:.2f}",
                delta=f"±${price_std:.2f}" if not pd.isna(price_std) else None
            )
    
    with col2:
        if not product_user_ratings.empty:
            avg_rating = product_user_ratings['rating'].mean()
            rating_count = len(product_user_ratings)
            st.metric(
                "⭐ Average Rating", 
                f"{avg_rating:.2f}/5.0",
                delta=f"{rating_count} reviews"
            )
    
    with col3:
        if not product_user_ratings.empty:
            unique_users = product_user_ratings['userId'].nunique()
            st.metric("👥 Unique Users", f"{unique_users:,}")
    
    with col4:
        if 'formatted_date' in product_user_ratings.columns:
            try:
                recent_reviews = len(product_user_ratings[
                    pd.to_datetime(product_user_ratings['formatted_date']) > 
                    (datetime.now() - timedelta(days=30))
                ])
                st.metric("📅 Recent Reviews", f"{recent_reviews}")
            except:
                st.metric("📅 Total Reviews", f"{len(product_user_ratings)}")

def display_interactive_product_charts(product_data, product_user_ratings, product_id):
    """Interactive charts using Plotly for better user experience"""
    st.markdown("### 📈 Interactive Analytics")
    
    if not product_user_ratings.empty:
        # Interactive rating distribution
        fig_rating = px.histogram(
            product_user_ratings, 
            x='rating',
            title=f"Rating Distribution - Product {product_id}",
            color_discrete_sequence=['#1E88E5'],
            nbins=10
        )
        fig_rating.update_layout(
            xaxis_title="Rating",
            yaxis_title="Count",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_rating, use_container_width=True)

def display_user_engagement_analysis(product_user_ratings, product_id):
    """Enhanced user engagement analysis"""
    st.markdown("### 👥 User Engagement Analysis")
    
    # Filter out 'unknown' users if they exist
    if not product_user_ratings.empty:
        if 'userId' in product_user_ratings.columns:
            product_user_ratings = product_user_ratings[product_user_ratings['userId'] != 'unknown']
    
    if not product_user_ratings.empty:
        # User ratings table with enhanced formatting
        display_cols = ['userId', 'rating']
        if 'formatted_date' in product_user_ratings.columns:
            display_cols.append('formatted_date')
        
        # Sort and display top ratings (excluding unknown users)
        top_ratings = product_user_ratings[display_cols].sort_values('rating', ascending=False).head(10)
        
        st.markdown("#### 🏆 Top User Ratings")
        st.dataframe(
            top_ratings.style.highlight_max(subset=['rating']),
            use_container_width=True
        )
        
        # Engagement statistics (calculated without unknown users)
        st.markdown("#### 📊 Engagement Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Ratings", len(product_user_ratings))
        
        with col2:
            avg_rating = product_user_ratings['rating'].mean()
            st.metric("⭐ Average Rating", f"{avg_rating:.2f}")
        
        with col3:
            rating_std = product_user_ratings['rating'].std()
            st.metric("📈 Rating Consistency", f"{rating_std:.2f}")
    else:
        st.markdown('<div class="info-box">ℹ️ No user ratings found for this product.</div>', 
                   unsafe_allow_html=True)

def render_analytics_overview(product_ratings, user_ratings):
    """Enhanced analytics overview with comprehensive insights"""
    st.markdown("## 📊 Comprehensive Analytics Overview")
    
    # Key performance indicators
    display_kpi_dashboard(product_ratings, user_ratings)
    
    # Market analysis
    display_market_analysis(product_ratings, user_ratings)

def display_kpi_dashboard(product_ratings, user_ratings):
    """Enhanced KPI dashboard with better metrics"""
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_products = len(product_ratings['productId'].unique()) if 'productId' in product_ratings.columns else 0
        st.metric("🛍️ Total Products", f"{total_products:,}")
    
    with col2:
        total_users = len(user_ratings['userId'].unique()) if 'userId' in user_ratings.columns else 0
        st.metric("👥 Active Users", f"{total_users:,}")
    
    with col3:
        total_ratings = len(user_ratings)
        st.metric("⭐ Total Ratings", f"{total_ratings:,}")
    
    with col4:
        if 'rating' in user_ratings.columns:
            avg_rating = user_ratings['rating'].mean()
            st.metric("📊 Avg Rating", f"{avg_rating:.2f}")
    
    with col5:
        if 'price' in product_ratings.columns:
            avg_price = product_ratings['price'].mean()
            st.metric("💰 Avg Price", f"${avg_price:.2f}")

def display_market_analysis(product_ratings, user_ratings):
    """Enhanced market analysis with advanced insights"""
    st.markdown("### 🏢 Market Analysis")
    
    if 'productId' not in product_ratings.columns:
        st.error("❌ Product ID column not found - cannot perform market analysis")
        return
    
    # Calculate comprehensive product statistics
    product_summary = calculate_enhanced_product_statistics(product_ratings, user_ratings)
    
    if not product_summary.empty:
        # Enhanced product summary table
        st.markdown("#### 📋 Product Performance Summary")
        st.dataframe(
            product_summary.head(20).style.highlight_max(axis=0),
            use_container_width=True,
            height=400
        )

def calculate_enhanced_product_statistics(product_ratings, user_ratings):
    """Calculate enhanced statistics for all products"""
    # Start with user ratings statistics
    user_stats = user_ratings.groupby('productId').agg({
        'rating': ['count', 'mean', 'std'],
        'userId': 'nunique'
    }).round(2)
    
    # Flatten column names
    user_stats.columns = ['Total Ratings', 'Avg Rating', 'Rating Std', 'Unique Users']
    
    # Add product-specific statistics if available
    if 'price' in product_ratings.columns:
        price_stats = product_ratings.groupby('productId')['price'].agg([
            ('Avg Price', 'mean'),
            ('Price Std', 'std')
        ]).round(2)
        user_stats = user_stats.join(price_stats, how='left')
    
    # Fill NaN values
    user_stats = user_stats.fillna(0)
    
    # Sort by total ratings
    user_stats = user_stats.sort_values('Total Ratings', ascending=False)
    
    return user_stats

# Running the Streamlit application
if __name__ == "__main__":
    # Load sample data for demonstration purposes.
    product_ratings = pd.DataFrame({
        'productId': ['A', 'B', 'C', 'D', 'E'],
        'price': [10.00, 15.50, 7.50, 20.00, 12.99],
        'description': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    })
    
    user_ratings = pd.DataFrame({
        'userId': ['user1', 'user1', 'user2', 'user2', 'user3'],
        'productId': ['A', 'B', 'C', 'D', 'A'],
        'rating': [5, 3, 4, 2, 5],
        'formatted_date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05']
    })
    
    # Render the application
    render(product_ratings, user_ratings)