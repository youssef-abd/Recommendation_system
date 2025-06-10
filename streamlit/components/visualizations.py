import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import numpy as np
from config.settings import DATA_PATHS

def render(user_ratings, product_ratings, transactions):
    """Render visualizations page"""
    st.subheader("📊 Data Visualizations")
    
    viz_option = st.selectbox("Select visualization", 
                            ["Rating Distribution", 
                             "Price vs Rating", 
                             "User-Product Interaction",
                             "Rating Timeline"])
    
    if viz_option == "Rating Distribution":
        render_rating_distribution(user_ratings)
    elif viz_option == "Price vs Rating":
        render_price_vs_rating(product_ratings)
    elif viz_option == "User-Product Interaction":
        render_interaction_matrix()
    elif viz_option == "Rating Timeline":
        render_rating_timeline(user_ratings)

def render_rating_distribution(user_ratings):
    """Render rating distribution analysis"""
    st.subheader("Rating Distribution Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.radio("Select plot type", 
                            ["Histogram", "Box Plot", "Violin Plot"])
    
    with col2:
        data_type = st.radio("Select data", 
                           ["Original Ratings", "Normalized Ratings"])
    
    fig = plt.figure(figsize=(10, 6))
    data = user_ratings['rating'] if data_type == "Original Ratings" else user_ratings['normalized_rating']
    
    if plot_type == "Histogram":
        sns.histplot(data, kde=True, bins=20)
        plt.title(f"{data_type} Distribution")
    elif plot_type == "Box Plot":
        sns.boxplot(x=data)
        plt.title(f"{data_type} Box Plot")
    else:
        sns.violinplot(x=data)
        plt.title(f"{data_type} Violin Plot")
    
    st.pyplot(fig)
    
    # Add statistical summary
    st.subheader("Statistical Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{data.mean():.3f}")
    with col2:
        st.metric("Median", f"{data.median():.3f}")
    with col3:
        st.metric("Std Dev", f"{data.std():.3f}")
    with col4:
        st.metric("Range", f"{data.max() - data.min():.3f}")

def render_price_vs_rating(product_ratings):
    """Render price vs rating analysis"""
    st.subheader("Price vs Rating Analysis")
    
    if 'price' in product_ratings.columns and 'rating' in product_ratings.columns:
        sample_size = st.slider("Sample size", 100, 5000, 1000)
        sampled_data = product_ratings.sample(min(sample_size, len(product_ratings)))
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plt.figure(figsize=(10, 6))
            sns.scatterplot(x='price', y='rating', data=sampled_data, alpha=0.6)
            plt.title("Price vs Rating Scatter Plot")
            plt.xlabel("Price")
            plt.ylabel("Rating")
            st.pyplot(fig)
        
        with col2:
            # Create price bins for better analysis
            sampled_data['price_bin'] = pd.cut(sampled_data['price'], bins=10)
            price_rating_summary = sampled_data.groupby('price_bin')['rating'].agg(['mean', 'count'])
            
            fig = plt.figure(figsize=(10, 6))
            price_rating_summary['mean'].plot(kind='bar')
            plt.title("Average Rating by Price Range")
            plt.xlabel("Price Range")
            plt.ylabel("Average Rating")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Correlation analysis
        correlation = sampled_data['price'].corr(sampled_data['rating'])
        st.metric("Price-Rating Correlation", f"{correlation:.3f}")
        
        if abs(correlation) > 0.3:
            st.success("Strong correlation detected!")
        elif abs(correlation) > 0.1:
            st.info("Moderate correlation detected.")
        else:
            st.warning("Weak correlation detected.")
            
    else:
        missing_cols = []
        if 'price' not in product_ratings.columns:
            missing_cols.append('price')
        if 'rating' not in product_ratings.columns:
            missing_cols.append('rating')
        st.warning(f"Cannot create visualization - missing columns: {', '.join(missing_cols)}")

def render_interaction_matrix():
    """Render user-product interaction matrix"""
    st.subheader("User-Product Interaction Matrix")
    
    try:
        with st.spinner("Loading interaction matrix..."):
            interaction_sparse = sp.load_npz(DATA_PATHS['interaction_matrix'])
            interaction_csr = interaction_sparse.tocsr()
            
            st.success("Interaction matrix loaded successfully!")
            
            # Display matrix information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Matrix Shape", f"{interaction_csr.shape[0]} x {interaction_csr.shape[1]}")
            with col2:
                st.metric("Non-zero Entries", f"{interaction_csr.nnz:,}")
            with col3:
                sparsity = (1 - interaction_csr.nnz / (interaction_csr.shape[0] * interaction_csr.shape[1])) * 100
                st.metric("Sparsity", f"{sparsity:.2f}%")
            
            # Sample interactions
            rows, cols = interaction_csr.nonzero()
            non_zero_data = pd.DataFrame({
                'userId': rows,
                'productId': cols,
                'rating': interaction_csr.data
            })
            
            st.write(f"**Sample of {len(non_zero_data)} total interactions:**")
            sample_size = st.slider("Sample size to display", 10, 500, 100)
            st.dataframe(non_zero_data.sample(min(sample_size, len(non_zero_data))))
            
            # Heatmap visualization
            if st.checkbox("Show heatmap sample"):
                render_interaction_heatmap(non_zero_data)
                
    except FileNotFoundError:
        st.error("Interaction matrix file not found.")
    except Exception as e:
        st.error(f"Failed to load interaction matrix: {str(e)}")

def render_interaction_heatmap(non_zero_data):
    """Render interaction matrix heatmap"""
    sample_users = non_zero_data['userId'].sample(min(10, non_zero_data['userId'].nunique())).unique()
    sample_products = non_zero_data['productId'].sample(min(10, non_zero_data['productId'].nunique())).unique()
    
    sample_df = non_zero_data[
        (non_zero_data['userId'].isin(sample_users)) & 
        (non_zero_data['productId'].isin(sample_products))
    ]
    
    if not sample_df.empty:
        pivot_df = sample_df.pivot(index='userId', columns='productId', values='rating')
        
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt='.1f')
        plt.title("User-Product Interaction Sample (Ratings)")
        plt.xlabel("Product ID")
        plt.ylabel("User ID")
        st.pyplot(fig)
    else:
        st.warning("No data available for heatmap visualization")

def render_rating_timeline(user_ratings):
    """Render rating activity over time"""
    st.subheader("Rating Activity Over Time")
    
    if 'date' in user_ratings.columns:
        # Time period selection
        time_period = st.selectbox("Aggregation period", ["Daily", "Weekly", "Monthly"])
        
        # Create timeline
        daily_ratings = user_ratings.set_index('date').resample('D').size()
        
        if time_period == "Weekly":
            ratings_timeline = daily_ratings.resample('W').sum()
        elif time_period == "Monthly":
            ratings_timeline = daily_ratings.resample('M').sum()
        else:
            ratings_timeline = daily_ratings
        
        # Display chart
        st.line_chart(ratings_timeline)
        
        # Additional statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Timeline Statistics")
            st.metric("Peak Activity", f"{ratings_timeline.max()} ratings")
            st.metric("Average Activity", f"{ratings_timeline.mean():.1f} ratings")
            st.metric("Total Days", len(ratings_timeline))
        
        with col2:
            # Show most active periods
            st.subheader("Most Active Periods")
            top_periods = ratings_timeline.nlargest(5)
            for date, count in top_periods.items():
                st.write(f"**{date.strftime('%Y-%m-%d')}**: {count} ratings")
        
        # Activity distribution
        fig = plt.figure(figsize=(10, 6))
        plt.hist(ratings_timeline.values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel(f'{time_period} Rating Count')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {time_period} Rating Activity')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    else:
        st.warning("Date information not available for timeline visualization")
        
        # Alternative: Show rating progression by order
        st.info("Showing rating progression by submission order instead")
        
        # Create artificial timeline based on index
        user_ratings_indexed = user_ratings.reset_index()
        window_size = st.slider("Moving average window", 100, 2000, 500)
        
        # Calculate moving average of ratings
        moving_avg = user_ratings_indexed['rating'].rolling(window=window_size).mean()
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(moving_avg.index, moving_avg.values)
        plt.xlabel('Rating Submission Order')
        plt.ylabel('Moving Average Rating')
        plt.title(f'Rating Trends (Moving Average with window={window_size})')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)