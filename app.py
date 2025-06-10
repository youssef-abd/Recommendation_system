import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import numpy as np
from datetime import datetime

# Custom CSS for better styling
st.markdown("""
<style>
    .main { padding-top: 1.5rem; }
    .sidebar .sidebar-content { padding-top: 1rem; }
    h1 { color: #2b5876; }
    h2 { color: #4e4376; border-bottom: 1px solid #eee; padding-bottom: 0.3rem; }
    .stDataFrame { width: 100%; }
    .stAlert { border-radius: 0.5rem; }
    .css-1aumxhk { background-color: #f8f9fa; }
    .css-1v3fvcr { padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# Load processed data with timestamp conversion
@st.cache_data
def load_data():
    try:
        # Load datasets
        user_ratings = pd.read_csv('data/processed/clean/cleaned_user_ratings.csv')
        product_ratings = pd.read_csv('data/processed/clean/cleaned_product_ratings2.csv')
        transactions = pd.read_csv('data/processed/clean/cleaned_transactions.csv')
        
        # Convert Unix timestamps to datetime in user_ratings
        if 'time' in user_ratings.columns:  # Check if timestamp column exists
            user_ratings['date'] = pd.to_datetime(user_ratings['time'], unit='s')
            user_ratings['formatted_date'] = user_ratings['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Check required columns in user_ratings
        required_user_cols = ['userId', 'productId', 'rating']
        missing_user_cols = [col for col in required_user_cols if col not in user_ratings.columns]
        if missing_user_cols:
            st.error(f"Error: Missing columns in user_ratings - {', '.join(missing_user_cols)}")
            st.stop()
            
        return user_ratings, product_ratings, transactions
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

user_ratings, product_ratings, transactions = load_data()

# Normalize ratings
scaler = MinMaxScaler()
user_ratings['normalized_rating'] = scaler.fit_transform(user_ratings[['rating']])

# Sidebar configuration
st.sidebar.header("🔍 Navigation")
st.sidebar.markdown("---")

view_option = st.sidebar.radio("Select View", [
    "📊 Dashboard Overview",
    "👤 User Analysis",
    "📦 Product Analysis",
    "💳 Transactions",
    "📈 Visualizations",
    "🎯 Recommendation Systems"
])

# Main content
st.title("📊 E-commerce Analytics Dashboard")

if view_option == "📊 Dashboard Overview":
    st.subheader("📌 Key Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", user_ratings['userId'].nunique())
    with col2:
        st.metric("Total Products", product_ratings['productId'].nunique() if 'productId' in product_ratings.columns else "N/A")
    with col3:
        st.metric("Total Ratings", len(user_ratings))
    
    st.markdown("---")
    st.subheader("🔍 Quick Data Preview")
    
    preview_option = st.radio("Select data to preview", 
                            ["User Ratings", "Product Info", "Transactions"])
    
    if preview_option == "User Ratings":
        cols_to_show = ['userId', 'productId', 'rating']
        if 'formatted_date' in user_ratings.columns:
            cols_to_show.append('formatted_date')
        st.dataframe(user_ratings[cols_to_show].head(1000).sort_values('rating', ascending=False))
    elif preview_option == "Product Info":
        if not product_ratings.empty:
            unique_products = product_ratings.drop_duplicates(subset=['productId'], keep='first')
            st.dataframe(unique_products.head(1000).reset_index(drop=True))
        else:
            st.warning("No product data available")
    else:
        if not transactions.empty:
            # Create a copy to avoid modifying the original dataframe
            transactions_display = transactions.copy()
            
            # Format the date if it exists
            if 'time' in transactions_display.columns:
                transactions_display['date'] = pd.to_datetime(transactions_display['time'], unit='s').dt.date
                transactions_display = transactions_display.drop(columns=['time'])
            
            # Determine which date column to show
            date_col = None
            if 'date' in transactions_display.columns:
                date_col = 'date'
            elif 'transactionDate' in transactions_display.columns:
                date_col = 'transactionDate'
            
            # Select columns to display
            display_cols = [col for col in transactions_display.columns if col != 'formatted_date']
            
            # Sort by date if available
            if date_col:
                transactions_display = transactions_display.sort_values(date_col, ascending=False)
                st.dataframe(transactions_display[display_cols].head(1000))
            else:
                st.dataframe(transactions_display[display_cols].head(1000))
        else:
            st.warning("No transaction data available")

elif view_option == "👤 User Analysis":
    st.subheader("👤 User Rating Analysis")
    
    tab1, tab2 = st.tabs(["🔎 Search User", "📊 All Users"])
    
    with tab1:
        user_id = st.text_input("Enter User ID", help="Enter a specific user ID to view their ratings")
        if user_id:
            with st.spinner("Searching user data..."):
                user_data = user_ratings[user_ratings['userId'] == user_id]
                if not user_data.empty:
                    st.success(f"Found {len(user_data)} ratings for user {user_id}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_rating = user_data['rating'].mean()
                        st.metric("Average Rating", f"{avg_rating:.2f}")
                    with col2:
                        st.metric("Normalized Avg", f"{user_data['normalized_rating'].mean():.2f}")
                    
                    display_cols = ['productId', 'rating']
                    if 'formatted_date' in user_data.columns:
                        display_cols.append('formatted_date')
                    st.dataframe(user_data[display_cols].sort_values('rating', ascending=False))
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.countplot(x='rating', data=user_data, ax=ax)
                    plt.title(f"Rating Distribution for User {user_id}")
                    st.pyplot(fig)
                    
                    if 'date' in user_data.columns:
                        st.subheader("Rating Timeline")
                        timeline = user_data.set_index('date')['rating'].sort_index()
                        st.line_chart(timeline)
                else:
                    st.warning("No ratings found for this user.")
    
    with tab2:
        st.subheader("All Users Summary")
        st.dataframe(
            user_ratings.groupby('userId')['rating']
            .agg(['count', 'mean', 'std'])
            .rename(columns={'count': 'Rating Count', 'mean': 'Avg Rating', 'std': 'Rating Std'})
            .sort_values('Rating Count', ascending=False)
            .head(1000)
        )

elif view_option == "📦 Product Analysis":
    st.subheader("📦 Product Analysis")
    
    tab1, tab2 = st.tabs(["🔍 Search Product", "📊 All Products"])
    
    with tab1:
        product_id = st.text_input("Enter Product ID", help="Enter a specific product ID to view its details")
        if product_id:
            with st.spinner("Searching product data..."):
                if 'productId' not in product_ratings.columns:
                    st.error("Product ID column not found in product data")
                else:
                    product_data = product_ratings[product_ratings['productId'] == product_id]
                    if not product_data.empty:
                        st.success(f"Found product {product_id}")
                        
                        cols = st.columns(2)
                        metric_cols = []
                        
                        if 'price' in product_data.columns:
                            with cols[0]:
                                st.metric("Average Price", f"${product_data['price'].mean():.2f}")
                            metric_cols.append('price')
                        
                        if 'rating' in product_data.columns:
                            with cols[1] if 'price' in product_data.columns else cols[0]:
                                st.metric("Average Rating", f"{product_data['rating'].mean():.2f}")
                            metric_cols.append('rating')
                        
                        if not metric_cols:
                            st.info("No numeric metrics available for this product")
                        
                        st.dataframe(product_data)
                        
                        product_ratings = user_ratings[user_ratings['productId'] == product_id]
                        if not product_ratings.empty:
                            st.subheader(f"User Ratings for {product_id}")
                            display_cols = ['userId', 'rating']
                            if 'formatted_date' in product_ratings.columns:
                                display_cols.append('formatted_date')
                            st.dataframe(product_ratings[display_cols].sort_values('rating', ascending=False))
                            
                            fig, ax = plt.subplots(figsize=(10, 4))
                            sns.countplot(x='rating', data=product_ratings, ax=ax)
                            plt.title(f"Rating Distribution for Product {product_id}")
                            st.pyplot(fig)
                    else:
                        st.warning("Product not found.")
    
    with tab2:
        st.subheader("All Products Summary")
    
    if 'productId' not in product_ratings.columns:
        st.error("Product ID column not found - cannot group products")
        st.stop()
    
    # Initialize summary dataframe
    product_summary = product_ratings.groupby('productId').size().to_frame('Total Ratings')
    
    # Add price statistics if available
    if 'price' in product_ratings.columns:
        product_summary = product_summary.join(
            product_ratings.groupby('productId')['price']
            .agg(**{
                'Avg Price': 'mean',
                'Price Variation': 'std'
            })
        )
    
    # Add rating statistics if available
    if 'rating' in product_ratings.columns:
        product_summary = product_summary.join(
            product_ratings.groupby('productId')['rating']
            .agg(**{
                'Avg Rating': 'mean',
                'Rating Consistency': 'std'
            })
        )
    
    if not product_summary.empty:
        # Display with simplified column names and sorting
        st.dataframe(
            product_summary
            .sort_values('Total Ratings', ascending=False)
            .head(1000)
            .rename_axis('Product ID')  # Better index name
        )
    else:
        st.warning("No product summary data available")

elif view_option == "💳 Transactions":
    st.subheader("💳 Transaction Analysis")
    
    if not transactions.empty:
        # Convert timestamp if 'time' column exists
        if 'time' in transactions.columns:
            # Convert to datetime and keep only date (no time)
            transactions['date'] = pd.to_datetime(transactions['time'], unit='s').dt.date
            
            # Drop the original 'time' column to avoid duplication
            transactions = transactions.drop(columns=['time'])
        
        # Sort by date (newest first)
        if 'date' in transactions.columns:
            st.write("### Transactions (Sorted by Date)")
            st.dataframe(
                transactions.sort_values('date', ascending=False).head(1000)
            )
        else:
            st.write("### Transactions (No Date Available)")
            st.dataframe(transactions.head(1000))
        
        # Display formatted amounts if available
        if 'amount' in transactions.columns:
            st.write("### Transaction Amounts")
            st.dataframe(
                transactions.head(1000).style.format({'amount': '${:.2f}'})
            )
        
        # Add time-based visualization if date exists
        if 'date' in transactions.columns:
            st.write("### Transaction Volume Over Time")
            time_period = st.selectbox("Aggregation period", ["Daily", "Weekly", "Monthly"])
            
            # Convert date back to datetime for resampling
            transactions_by_time = transactions.set_index(pd.to_datetime(transactions['date'])).resample('D').size()
            
            if time_period == "Weekly":
                transactions_by_time = transactions_by_time.resample('W').sum()
            elif time_period == "Monthly":
                transactions_by_time = transactions_by_time.resample('M').sum()
            
            st.line_chart(transactions_by_time)
    else:
        st.warning("No transaction data available")

elif view_option == "📈 Visualizations":
    st.subheader("📊 Data Visualizations")
    
    viz_option = st.selectbox("Select visualization", 
                            ["Rating Distribution", 
                             "Price vs Rating", 
                             "User-Product Interaction",
                             "Rating Timeline"])
    
    if viz_option == "Rating Distribution":
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
    
    elif viz_option == "Price vs Rating":
        st.subheader("Price vs Rating Analysis")
        
        if 'price' in product_ratings.columns and 'rating' in product_ratings.columns:
            sample_size = st.slider("Sample size", 100, 5000, 1000)
            sampled_data = product_ratings.sample(sample_size)
            
            fig = plt.figure(figsize=(10, 6))
            sns.scatterplot(x='price', y='rating', data=sampled_data, alpha=0.6)
            plt.title("Price vs Rating")
            st.pyplot(fig)
        else:
            missing_cols = []
            if 'price' not in product_ratings.columns:
                missing_cols.append('price')
            if 'rating' not in product_ratings.columns:
                missing_cols.append('rating')
            st.warning(f"Cannot create visualization - missing columns: {', '.join(missing_cols)}")
    
    elif viz_option == "User-Product Interaction":
        st.subheader("User-Product Interaction Matrix")
        
        try:
            with st.spinner("Loading interaction matrix..."):
                interaction_sparse = sp.load_npz('data/processed/interaction_matrix_sparse.npz')
                interaction_csr = interaction_sparse.tocsr()
                
                st.success("Interaction matrix loaded successfully!")
                
                rows, cols = interaction_csr.nonzero()
                non_zero_data = pd.DataFrame({
                    'userId': rows,
                    'productId': cols,
                    'rating': interaction_csr.data
                })
                
                st.write(f"Total interactions: {non_zero_data.shape[0]}")
                
                sample_size = st.slider("Sample size to display", 10, 500, 100)
                st.dataframe(non_zero_data.sample(sample_size))
                
                if st.checkbox("Show heatmap sample"):
                    sample_users = non_zero_data['userId'].sample(10).unique()
                    sample_products = non_zero_data['productId'].sample(10).unique()
                    
                    sample_df = non_zero_data[
                        (non_zero_data['userId'].isin(sample_users)) & 
                        (non_zero_data['productId'].isin(sample_products))
                    ]
                    
                    pivot_df = sample_df.pivot(index='userId', columns='productId', values='rating')
                    
                    fig = plt.figure(figsize=(12, 8))
                    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu")
                    plt.title("User-Product Interaction Sample (Ratings)")
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Failed to load interaction matrix: {str(e)}")
    
    elif viz_option == "Rating Timeline":
        st.subheader("Rating Activity Over Time")
        
        if 'date' in user_ratings.columns:
            daily_ratings = user_ratings.set_index('date').resample('D').size()
            
            time_period = st.selectbox("Aggregation period", ["Daily", "Weekly", "Monthly"])
            
            if time_period == "Weekly":
                ratings_timeline = daily_ratings.resample('W').sum()
            elif time_period == "Monthly":
                ratings_timeline = daily_ratings.resample('M').sum()
            else:
                ratings_timeline = daily_ratings
            
            st.line_chart(ratings_timeline)
        else:
            st.warning("Date information not available for timeline visualization")

elif view_option == "🎯 Recommendation Systems":
    st.subheader("🎯 Product Recommendation Engine")
    
    rec_method = st.selectbox("Select recommendation method", 
                            ["Collaborative Filtering (ALS)", 
                             "Content-Based", 
                             "Hybrid"])
    
    if rec_method == "Collaborative Filtering (ALS)":
        try:
            with st.spinner("Loading collaborative filtering recommendations..."):
                recs_df = pd.read_csv('data/processed/recommendations/advanced_als_recommendations/als_recs.csv')
                
                if 'productId' not in recs_df.columns or 'userId' not in recs_df.columns or 'predictedRating' not in recs_df.columns:
                    st.error("Required columns missing in recommendations data")
                else:
                    if 'description' in product_ratings.columns:
                        recs_df = recs_df.merge(product_ratings[['productId', 'description']], on='productId', how='left')
                    
                    recs_df = recs_df.drop_duplicates(subset=['userId', 'productId'])
                    
                    st.success("Recommendations loaded successfully!")
                    
                    tab1, tab2 = st.tabs(["🔍 User Recommendations", "🏆 Top Global Recommendations"])
                    
                    with tab1:
                        user_input = st.text_input("Enter User ID for personalized recommendations")
                        if user_input:
                            user_recs = recs_df[recs_df['userId'] == user_input]
                            if not user_recs.empty:
                                num_recs = st.slider("Number of recommendations", 5, 20, 10)
                                user_recs = user_recs.sort_values('predictedRating', ascending=False).head(num_recs)
                                user_recs['rank'] = range(1, len(user_recs) + 1)
                                
                                st.subheader(f"Top {num_recs} Recommendations for User {user_input}")
                                
                                display_cols = ['rank', 'productId', 'predictedRating']
                                if 'description' in user_recs.columns:
                                    display_cols.insert(2, 'description')
                                
                                st.dataframe(user_recs[display_cols])
                            else:
                                st.warning("No recommendations found for this user.")
                    
                    with tab2:
                        st.subheader("Top Global Recommendations")
                        num_global = st.slider("Number of global recommendations", 5, 20, 10)
                        
                        top_recs = (
                            recs_df.groupby('productId')['predictedRating']
                            .agg(['mean', 'count'])
                            .rename(columns={'mean': 'AvgScore', 'count': 'NumRecommendations'})
                            .sort_values('AvgScore', ascending=False)
                            .head(num_global)
                        )
                        
                        if 'description' in product_ratings.columns:
                            top_recs = top_recs.merge(product_ratings[['productId', 'description']], on='productId', how='left')
                        
                        top_recs['rank'] = range(1, len(top_recs) + 1)
                        
                        display_cols = ['rank', 'productId', 'AvgScore', 'NumRecommendations']
                        if 'description' in top_recs.columns:
                            display_cols.insert(2, 'description')
                        
                        st.dataframe(top_recs[display_cols])
        
        except FileNotFoundError:
            st.error("Collaborative filtering recommendations file not found.")
    
    elif rec_method == "Content-Based":
        try:
            # Load data
            content_df = pd.read_csv('data/processed/content_based_recommendations/content_based_recommendations.csv')
            
            if content_df.empty:
                st.error("No recommendation data available")
                st.stop()
            
            # Verify required columns exist
            required_cols = ['source_product_id', 'recommended_product_id', 'similarity_score', 'rank']
            missing_cols = [col for col in required_cols if col not in content_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Create two columns layout
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Product input with search functionality
                all_products = sorted(content_df['source_product_id'].unique())
                product_input = st.selectbox(
                    "Select a product ID to find similar products",
                    options=all_products,
                    help="Choose a product ID to see similar items"
                )
                
            with col2:
                num_recs = st.slider(
                    "Number of recommendations to show",
                    min_value=1,
                    max_value=20,
                    value=5
                )

            if product_input:
                # Get all products similar to the selected one
                similar_products = content_df[
                    content_df['source_product_id'] == product_input
                ].sort_values('similarity_score', ascending=False)
                
                if not similar_products.empty:
                    st.success(f"Found {len(similar_products)} similar products to {product_input}")
                    
                    # Merge with product descriptions if available
                    display_df = similar_products.copy()
                    if 'productId' in product_ratings.columns and 'description' in product_ratings.columns:
                        display_df = display_df.merge(
                            product_ratings[['productId', 'description']], 
                            left_on='recommended_product_id', 
                            right_on='productId', 
                            how='left'
                        )
                    
                    # Display results in a styled table
                    display_cols = ['rank', 'recommended_product_id', 'similarity_score']
                    if 'description' in display_df.columns:
                        display_cols.insert(2, 'description')
                    
                    st.dataframe(
                        display_df[display_cols]
                        .head(num_recs)
                        .style.format({'similarity_score': "{:.3f}"})
                        .background_gradient(subset=['similarity_score'], cmap='YlGnBu')
                    )
                    
                    # Visualize similarity scores
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_data = display_df.head(num_recs)
                    
                    # Use description if available, otherwise use product ID
                    y_labels = plot_data['description'].fillna(plot_data['recommended_product_id']) if 'description' in plot_data.columns else plot_data['recommended_product_id']
                    
                    sns.barplot(
                        x='similarity_score',
                        y=y_labels,
                        data=plot_data.assign(y_labels=y_labels),
                        palette='Blues_d',
                        ax=ax
                    )
                    ax.set_title(f"Top {num_recs} Similar Products to {product_input}")
                    ax.set_xlabel("Similarity Score")
                    ax.set_ylabel("Recommended Products")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show statistics
                    st.subheader("Recommendation Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Highest Similarity", f"{similar_products['similarity_score'].max():.3f}")
                    with col2:
                        st.metric("Average Similarity", f"{similar_products['similarity_score'].mean():.3f}")
                    with col3:
                        st.metric("Total Recommendations", len(similar_products))
                else:
                    st.warning(f"No similar products found for {product_input}")
        
        except FileNotFoundError:
            st.error("Content-based recommendations file not found.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    elif rec_method == "Hybrid":
        try:
            with st.spinner("Loading hybrid recommendations..."):
                # First, try to detect if the file has a proper header
                with open('data/processed/hybrid_recommendations/hybrid_recommendations.csv', 'r') as f:
                    first_line = f.readline().strip()
                
                # Check if first line looks like a header (contains text like 'user_id', 'item_id', etc.)
                has_header = any(keyword in first_line.lower() for keyword in ['user', 'item', 'hybrid', 'score', 'rank'])
                
                if has_header:
                    # Try reading with header
                    try:
                        hybrid_df = pd.read_csv('data/processed/hybrid_recommendations/hybrid_recommendations.csv')
                    except pd.errors.ParserError:
                        # If parsing fails, read without header
                        hybrid_df = pd.read_csv('data/processed/hybrid_recommendations/hybrid_recommendations.csv', header=None)
                        has_header = False
                else:
                    # Read without header since first line appears to be data
                    hybrid_df = pd.read_csv('data/processed/hybrid_recommendations/hybrid_recommendations.csv', header=None)
                
                # If no proper header, set column names based on number of columns
                if not has_header or len([col for col in ['user_id', 'item_id', 'hybrid_score'] if col in hybrid_df.columns]) < 3:
                    if hybrid_df.shape[1] == 4:
                        hybrid_df.columns = ['user_id', 'item_id', 'hybrid_score', 'rank']
                    elif hybrid_df.shape[1] == 3:
                        hybrid_df.columns = ['user_id', 'item_id', 'hybrid_score']
                    else:
                        st.error(f"Unexpected number of columns: {hybrid_df.shape[1]}")
                        st.write(f"Detected columns: {hybrid_df.shape[1]}")
                        st.write("Expected format: user_id, item_id, hybrid_score, [rank]")
                        st.stop()
                
                if hybrid_df.empty:
                    st.error("No hybrid recommendation data available")
                    st.stop()
                
                # Clean and validate the required columns
                required_cols = ['user_id', 'item_id', 'hybrid_score']
                missing_cols = [col for col in required_cols if col not in hybrid_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    st.write(f"Available columns: {list(hybrid_df.columns)}")
                    st.stop()
                
                # Convert data types
                hybrid_df['user_id'] = hybrid_df['user_id'].astype(str)
                hybrid_df['item_id'] = hybrid_df['item_id'].astype(str)
                hybrid_df['hybrid_score'] = pd.to_numeric(hybrid_df['hybrid_score'], errors='coerce')
                
                # Handle rank column
                if 'rank' in hybrid_df.columns:
                    hybrid_df['rank'] = pd.to_numeric(hybrid_df['rank'], errors='coerce')
                else:
                    # Create rank based on hybrid_score within each user
                    hybrid_df = hybrid_df.sort_values(['user_id', 'hybrid_score'], ascending=[True, False])
                    hybrid_df['rank'] = hybrid_df.groupby('user_id').cumcount() + 1
                
                # Remove rows with invalid scores
                hybrid_df = hybrid_df.dropna(subset=['hybrid_score'])
                
                # Remove duplicates
                hybrid_df = hybrid_df.drop_duplicates(subset=['user_id', 'item_id'])
                
                if hybrid_df.empty:
                    st.error("No valid data remaining after processing")
                    st.stop()
                
                st.success(f"Successfully loaded {len(hybrid_df)} hybrid recommendations!")
                
                # Create interface
                tab1, tab2 = st.tabs(["👤 User Recommendations", "📊 Score Analysis"])
                
                with tab1:
                    st.subheader("User-Specific Recommendations")
                    
                    available_users = sorted(hybrid_df['user_id'].unique())
                    
                    # User selection
                    user_input = st.selectbox(
                        "Select User ID",
                        options=available_users[:500],  # Limit for performance
                        help=f"Choose from {len(available_users)} available users"
                    )
                    
                    num_recs = st.slider("Number of recommendations to show", 5, 50, 10)
                    
                    if user_input:
                        user_recs = hybrid_df[hybrid_df['user_id'] == user_input]
                        user_recs = user_recs.sort_values('hybrid_score', ascending=False).head(num_recs)
                        
                        if not user_recs.empty:
                            st.write(f"**Top {num_recs} recommendations for User {user_input}:**")
                            
                            # Display recommendations
                            display_cols = ['rank', 'item_id', 'hybrid_score']
                            display_data = user_recs[display_cols].copy()
                            display_data['hybrid_score'] = display_data['hybrid_score'].round(4)
                            
                            st.dataframe(display_data, use_container_width=True)
                            
                            # Visualization
                            if len(user_recs) > 1:
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                # Create bar chart
                                bars = ax.bar(range(len(user_recs)), user_recs['hybrid_score'])
                                ax.set_xlabel('Recommendation Rank')
                                ax.set_ylabel('Hybrid Score')
                                ax.set_title(f'Hybrid Recommendation Scores for User {user_input}')
                                
                                # Customize x-axis
                                ax.set_xticks(range(len(user_recs)))
                                if len(user_recs) <= 20:
                                    ax.set_xticklabels([f'#{i+1}' for i in range(len(user_recs))])
                                else:
                                    ax.set_xticklabels([f'#{i+1}' if i % 5 == 0 else '' for i in range(len(user_recs))])
                                
                                # Add grid for better readability
                                ax.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            # Show statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Recommendations", len(user_recs))
                            with col2:
                                st.metric("Highest Score", f"{user_recs['hybrid_score'].max():.4f}")
                            with col3:
                                st.metric("Average Score", f"{user_recs['hybrid_score'].mean():.4f}")
                        else:
                            st.warning("No recommendations found for this user.")
                
                with tab2:
                    st.subheader("Overall Score Analysis")
                    
                    # Global statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Users", hybrid_df['user_id'].nunique())
                    with col2:
                        st.metric("Total Items", hybrid_df['item_id'].nunique())
                    with col3:
                        st.metric("Total Recommendations", len(hybrid_df))
                    with col4:
                        st.metric("Avg Score", f"{hybrid_df['hybrid_score'].mean():.4f}")
                    
                    # Score distribution
                    st.subheader("Hybrid Score Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(hybrid_df['hybrid_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.set_xlabel('Hybrid Score')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Distribution of Hybrid Scores')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col2:
                        # Box plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.boxplot(hybrid_df['hybrid_score'], vert=True)
                        ax.set_ylabel('Hybrid Score')
                        ax.set_title('Hybrid Score Box Plot')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    # Top items by average score
                    st.subheader("Top Items by Average Hybrid Score")
                    
                    num_top_items = st.slider("Number of top items to show", 10, 100, 20)
                    
                    top_items = (
                        hybrid_df.groupby('item_id')['hybrid_score']
                        .agg(['mean', 'count'])
                        .rename(columns={'mean': 'avg_score', 'count': 'recommendation_count'})
                        .sort_values('avg_score', ascending=False)
                        .head(num_top_items)
                        .reset_index()
                    )
                    
                    top_items['rank'] = range(1, len(top_items) + 1)
                    top_items['avg_score'] = top_items['avg_score'].round(4)
                    
                    st.dataframe(
                        top_items[['rank', 'item_id', 'avg_score', 'recommendation_count']], 
                        use_container_width=True
                    )
        
        except FileNotFoundError:
            st.error("Hybrid recommendations file not found at 'data/processed/hybrid_recommendations/hybrid_recommendations.csv'")
            st.info("Please ensure the file exists and contains the required data structure.")
        except Exception as e:
            st.error(f"Error loading hybrid recommendations: {str(e)}")
            st.write("Please check the file format and data structure.")
            
            # Show debug information
            try:
                with open('data/processed/hybrid_recommendations/hybrid_recommendations.csv', 'r') as f:
                    first_few_lines = f.readlines()[:5]
                st.write("**First few lines of the file:**")
                for i, line in enumerate(first_few_lines):
                    st.code(f"Line {i}: {line.strip()}")
            except:
                st.write("Could not read file for debugging")

# Add footer
st.markdown("---")
st.markdown("""
<style>
.footer { font-size: 0.8rem; color: #666; text-align: center; }
</style>
<div class="footer">
    E-commerce Analytics Dashboard • Powered by Streamlit • Data last updated: 2023
</div>
""", unsafe_allow_html=True)