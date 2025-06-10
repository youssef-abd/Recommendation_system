# =============================================================================
# FILE: components/product_analysis.py
# =============================================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def render(product_ratings, user_ratings):
    """Render product analysis page"""
    st.subheader("📦 Product Analysis")
    
    tab1, tab2 = st.tabs(["🔍 Search Product", "📊 All Products"])
    
    with tab1:
        render_product_search(product_ratings, user_ratings)
    
    with tab2:
        render_all_products_summary(product_ratings)

def render_product_search(product_ratings, user_ratings):
    """Render individual product search and analysis"""
    product_id = st.text_input("Enter Product ID", help="Enter a specific product ID to view its details")
    
    if product_id:
        with st.spinner("Searching product data..."):
            if 'productId' not in product_ratings.columns:
                st.error("Product ID column not found in product data")
                return
            
            product_data = product_ratings[product_ratings['productId'] == product_id]
            
            if not product_data.empty:
                st.success(f"Found product {product_id}")
                
                # Display product metrics
                display_product_metrics(product_data)
                
                # Display product details table
                display_product_details(product_data)
                
                # Display user ratings for this product
                display_product_user_ratings(user_ratings, product_id)
            else:
                st.warning("Product not found.")

def display_product_metrics(product_data):
    """Display product-specific metrics"""
    cols = st.columns(2)
    metric_idx = 0
    
    if 'price' in product_data.columns:
        with cols[metric_idx]:
            avg_price = product_data['price'].mean()
            st.metric("Average Price", f"${avg_price:.2f}")
        metric_idx += 1
    
    if 'rating' in product_data.columns:
        with cols[metric_idx % 2]:
            avg_rating = product_data['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}")

def display_product_details(product_data):
    """Display product details in a table"""
    st.subheader("Product Details")
    st.dataframe(product_data, use_container_width=True)

def display_product_user_ratings(user_ratings, product_id):
    """Display user ratings for a specific product"""
    product_user_ratings = user_ratings[user_ratings['productId'] == product_id]
    
    if not product_user_ratings.empty:
        st.subheader(f"User Ratings for Product {product_id}")
        
        # Display user ratings table
        display_cols = ['userId', 'rating']
        if 'formatted_date' in product_user_ratings.columns:
            display_cols.append('formatted_date')
        
        st.dataframe(
            product_user_ratings[display_cols].sort_values('rating', ascending=False),
            use_container_width=True
        )
        
        # Display rating distribution for this product
        display_product_rating_distribution(product_user_ratings, product_id)
        
        # Display product rating statistics
        display_product_rating_stats(product_user_ratings, product_id)
    else:
        st.info("No user ratings found for this product.")

def display_product_rating_distribution(product_user_ratings, product_id):
    """Display rating distribution chart for a specific product"""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(x='rating', data=product_user_ratings, ax=ax)
    plt.title(f"Rating Distribution for Product {product_id}")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    st.pyplot(fig)

def display_product_rating_stats(product_user_ratings, product_id):
    """Display rating statistics for a specific product"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Ratings", len(product_user_ratings))
    
    with col2:
        st.metric("Average Rating", f"{product_user_ratings['rating'].mean():.2f}")
    
    with col3:
        st.metric("Rating Std Dev", f"{product_user_ratings['rating'].std():.2f}")

def render_all_products_summary(product_ratings):
    """Render summary of all products"""
    st.subheader("All Products Summary")
    
    if 'productId' not in product_ratings.columns:
        st.error("Product ID column not found - cannot group products")
        return
    
    # Calculate product statistics
    product_summary = calculate_product_statistics(product_ratings)
    
    if not product_summary.empty:
        # Display product summary table
        st.dataframe(
            product_summary.head(1000),
            use_container_width=True
        )
        
        # Display product analysis charts
        display_product_analysis_charts(product_summary, product_ratings)
    else:
        st.warning("No product summary data available")

def calculate_product_statistics(product_ratings):
    """Calculate statistics for all products"""
    # Initialize with basic count
    product_summary = product_ratings.groupby('productId').size().to_frame('Total Records')
    
    # Add price statistics if available
    if 'price' in product_ratings.columns:
        price_stats = product_ratings.groupby('productId')['price'].agg([
            ('Avg Price', 'mean'),
            ('Price Variation', 'std')
        ])
        product_summary = product_summary.join(price_stats)
    
    # Add rating statistics if available
    if 'rating' in product_ratings.columns:
        rating_stats = product_ratings.groupby('productId')['rating'].agg([
            ('Avg Rating', 'mean'),
            ('Rating Consistency', 'std')
        ])
        product_summary = product_summary.join(rating_stats)
    
    # Fill NaN values and round numeric columns
    numeric_cols = product_summary.select_dtypes(include=['float', 'int']).columns
    product_summary[numeric_cols] = product_summary[numeric_cols].fillna(0).round(2)
    
    # Sort by total records
    product_summary = product_summary.sort_values('Total Records', ascending=False)
    
    return product_summary

def display_product_analysis_charts(product_summary, product_ratings):
    """Display product analysis charts"""
    st.subheader("Product Analysis Charts")
    
    # Price analysis if price data is available
    if 'Avg Price' in product_summary.columns:
        display_price_analysis(product_summary)
    
    # Rating analysis if rating data is available
    if 'Avg Rating' in product_summary.columns:
        display_rating_analysis(product_summary)
    
    # Product popularity analysis
    display_popularity_analysis(product_summary)

def display_price_analysis(product_summary):
    """Display price analysis charts"""
    st.subheader("Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(product_summary['Avg Price'].dropna(), bins=50, alpha=0.7, color='gold', edgecolor='black')
        ax.set_xlabel('Average Price ($)')
        ax.set_ylabel('Number of Products')
        ax.set_title('Distribution of Product Prices')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Price vs Rating scatter (if both available)
        if 'Avg Rating' in product_summary.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sample_data = product_summary.dropna(subset=['Avg Price', 'Avg Rating']).sample(
                min(1000, len(product_summary))
            )
            ax.scatter(sample_data['Avg Price'], sample_data['Avg Rating'], alpha=0.6, color='coral')
            ax.set_xlabel('Average Price ($)')
            ax.set_ylabel('Average Rating')
            ax.set_title('Price vs Rating')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

def display_rating_analysis(product_summary):
    """Display rating analysis charts"""
    st.subheader("Rating Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(product_summary['Avg Rating'].dropna(), bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Number of Products')
        ax.set_title('Distribution of Product Ratings')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Rating consistency
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(product_summary['Rating Consistency'].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Rating Standard Deviation')
        ax.set_ylabel('Number of Products')
        ax.set_title('Rating Consistency Distribution')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def display_popularity_analysis(product_summary):
    """Display product popularity analysis"""
    st.subheader("Product Popularity")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(product_summary))
    
    with col2:
        avg_records = product_summary['Total Records'].mean()
        st.metric("Avg Records per Product", f"{avg_records:.1f}")
    
    with col3:
        most_popular = product_summary['Total Records'].max()
        st.metric("Most Popular Product", f"{most_popular} records")
    
    with col4:
        if 'Avg Rating' in product_summary.columns:
            overall_avg = product_summary['Avg Rating'].mean()
            st.metric("Overall Avg Rating", f"{overall_avg:.2f}")
    
    # Popularity distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(product_summary['Total Records'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.set_xlabel('Number of Records per Product')
    ax.set_ylabel('Number of Products')
    ax.set_title('Product Popularity Distribution')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)