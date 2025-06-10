
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_loader import load_recommendations_data
from config.settings import DEFAULT_NUM_RECOMMENDATIONS

def render(user_ratings, product_ratings):
    """Render recommendation systems page"""
    st.subheader("🎯 Product Recommendation Engine")
    
    rec_method = st.selectbox("Select recommendation method", 
                            ["Collaborative Filtering (ALS)", 
                             "Content-Based", 
                             "Hybrid"])
    
    if rec_method == "Collaborative Filtering (ALS)":
        render_collaborative_filtering(product_ratings)
    elif rec_method == "Content-Based":
        render_content_based(product_ratings)
    elif rec_method == "Hybrid":
        render_hybrid_recommendations()

def render_collaborative_filtering(product_ratings):
    """Render collaborative filtering recommendations"""
    try:
        with st.spinner("Loading collaborative filtering recommendations..."):
            recs_df = load_recommendations_data('als')
            
            if recs_df is None:
                st.error("Collaborative filtering recommendations file not found.")
                return
            
            # Validate required columns
            required_cols = ['productId', 'userId', 'predictedRating']
            if not all(col in recs_df.columns for col in required_cols):
                st.error("Required columns missing in recommendations data")
                return
            
            # Merge with product descriptions if available
            if 'description' in product_ratings.columns:
                recs_df = recs_df.merge(
                    product_ratings[['productId', 'description']], 
                    on='productId', 
                    how='left'
                )
            
            # Remove duplicates
            recs_df = recs_df.drop_duplicates(subset=['userId', 'productId'])
            
            st.success("Recommendations loaded successfully!")
            
            tab1, tab2 = st.tabs(["🔍 User Recommendations", "🏆 Top Global Recommendations"])
            
            with tab1:
                render_user_recommendations(recs_df)
            
            with tab2:
                render_global_recommendations(recs_df, product_ratings)
                
    except Exception as e:
        st.error(f"Error loading collaborative filtering recommendations: {str(e)}")

def render_user_recommendations(recs_df):
    """Render user-specific recommendations"""
    user_input = st.text_input("Enter User ID for personalized recommendations")
    
    if user_input:
        user_recs = recs_df[recs_df['userId'] == user_input]
        
        if not user_recs.empty:
            num_recs = st.slider("Number of recommendations", 5, 20, DEFAULT_NUM_RECOMMENDATIONS)
            user_recs = user_recs.sort_values('predictedRating', ascending=False).head(num_recs)
            user_recs['rank'] = range(1, len(user_recs) + 1)
            
            st.subheader(f"Top {num_recs} Recommendations for User {user_input}")
            
            # Prepare display columns
            display_cols = ['rank', 'productId', 'predictedRating']
            if 'description' in user_recs.columns:
                display_cols.insert(2, 'description')
            
            st.dataframe(user_recs[display_cols])
            
            # Visualization
            if len(user_recs) > 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(len(user_recs)), user_recs['predictedRating'])
                ax.set_xlabel('Recommendation Rank')
                ax.set_ylabel('Predicted Rating')
                ax.set_title(f'Predicted Ratings for User {user_input}')
                ax.set_xticks(range(len(user_recs)))
                ax.set_xticklabels([f'#{i+1}' for i in range(len(user_recs))])
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("No recommendations found for this user.")

def render_global_recommendations(recs_df, product_ratings):
    """Render global top recommendations"""
    st.subheader("Top Global Recommendations")
    num_global = st.slider("Number of global recommendations", 5, 20, DEFAULT_NUM_RECOMMENDATIONS)
    
    top_recs = (
        recs_df.groupby('productId')['predictedRating']
        .agg(['mean', 'count'])
        .rename(columns={'mean': 'AvgScore', 'count': 'NumRecommendations'})
        .sort_values('AvgScore', ascending=False)
        .head(num_global)
        .reset_index()
    )
    
    # Merge with product descriptions if available
    if 'description' in product_ratings.columns:
        top_recs = top_recs.merge(
            product_ratings[['productId', 'description']], 
            on='productId', 
            how='left'
        )
    
    top_recs['rank'] = range(1, len(top_recs) + 1)
    
    # Prepare display columns
    display_cols = ['rank', 'productId', 'AvgScore', 'NumRecommendations']
    if 'description' in top_recs.columns:
        display_cols.insert(2, 'description')
    
    st.dataframe(top_recs[display_cols])

def render_content_based(product_ratings):
    """Render content-based recommendations"""
    try:
        content_df = load_recommendations_data('content')
        
        if content_df is None:
            st.error("Content-based recommendations file not found.")
            return
        
        if content_df.empty:
            st.error("No recommendation data available")
            return
        
        # Verify required columns
        required_cols = ['source_product_id', 'recommended_product_id', 'similarity_score', 'rank']
        missing_cols = [col for col in required_cols if col not in content_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return
        
        # Create layout
        col1, col2 = st.columns([2, 3])
        
        with col1:
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
            render_similar_products(content_df, product_ratings, product_input, num_recs)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def render_similar_products(content_df, product_ratings, product_input, num_recs):
    """Render similar products for content-based recommendations"""
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
        
        # Display results
        display_cols = ['rank', 'recommended_product_id', 'similarity_score']
        if 'description' in display_df.columns:
            display_cols.insert(2, 'description')
        
        st.dataframe(
            display_df[display_cols]
            .head(num_recs)
            .style.format({'similarity_score': "{:.3f}"})
            .background_gradient(subset=['similarity_score'], cmap='YlGnBu')
        )
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_data = display_df.head(num_recs)
        
        y_labels = (plot_data['description'].fillna(plot_data['recommended_product_id']) 
                   if 'description' in plot_data.columns 
                   else plot_data['recommended_product_id'])
        
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
        
        # Statistics
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

def render_hybrid_recommendations():
    """Render hybrid recommendations"""
    try:
        with st.spinner("Loading hybrid recommendations..."):
            hybrid_df = load_and_process_hybrid_data()
            
            if hybrid_df is None or hybrid_df.empty:
                st.error("No hybrid recommendation data available")
                return
            
            st.success(f"Successfully loaded {len(hybrid_df)} hybrid recommendations!")
            
            tab1, tab2 = st.tabs(["👤 User Recommendations", "📊 Score Analysis"])
            
            with tab1:
                render_hybrid_user_recommendations(hybrid_df)
            
            with tab2:
                render_hybrid_score_analysis(hybrid_df)
                
    except Exception as e:
        st.error(f"Error loading hybrid recommendations: {str(e)}")

def load_and_process_hybrid_data():
    """Load and process hybrid recommendation data"""
    hybrid_df = load_recommendations_data('hybrid')
    
    if hybrid_df is None:
        return None
    
    # Detect if file has proper header
    if len([col for col in ['user_id', 'item_id', 'hybrid_score'] if col in hybrid_df.columns]) < 3:
        if hybrid_df.shape[1] == 4:
            hybrid_df.columns = ['user_id', 'item_id', 'hybrid_score', 'rank']
        elif hybrid_df.shape[1] == 3:
            hybrid_df.columns = ['user_id', 'item_id', 'hybrid_score']
        else:
            st.error(f"Unexpected number of columns: {hybrid_df.shape[1]}")
            return None
    
    # Convert data types
    hybrid_df['user_id'] = hybrid_df['user_id'].astype(str)
    hybrid_df['item_id'] = hybrid_df['item_id'].astype(str)
    hybrid_df['hybrid_score'] = pd.to_numeric(hybrid_df['hybrid_score'], errors='coerce')
    
    # Handle rank column
    if 'rank' not in hybrid_df.columns:
        hybrid_df = hybrid_df.sort_values(['user_id', 'hybrid_score'], ascending=[True, False])
        hybrid_df['rank'] = hybrid_df.groupby('user_id').cumcount() + 1
    else:
        hybrid_df['rank'] = pd.to_numeric(hybrid_df['rank'], errors='coerce')
    
    # Clean data
    hybrid_df = hybrid_df.dropna(subset=['hybrid_score'])
    hybrid_df = hybrid_df.drop_duplicates(subset=['user_id', 'item_id'])
    
    return hybrid_df

def render_hybrid_user_recommendations(hybrid_df):
    """Render hybrid user-specific recommendations"""
    st.subheader("User-Specific Recommendations")
    
    available_users = sorted(hybrid_df['user_id'].unique())
    
    user_input = st.selectbox(
        "Select User ID",
        options=available_users[:500],  # Limit for performance
        help=f"Choose from {len(available_users)} available users"
    )
    
    num_recs = st.slider("Number of recommendations to show", 5, 50, DEFAULT_NUM_RECOMMENDATIONS)
    
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
                bars = ax.bar(range(len(user_recs)), user_recs['hybrid_score'])
                ax.set_xlabel('Recommendation Rank')
                ax.set_ylabel('Hybrid Score')
                ax.set_title(f'Hybrid Recommendation Scores for User {user_input}')
                ax.set_xticks(range(len(user_recs)))
                ax.set_xticklabels([f'#{i+1}' for i in range(len(user_recs))])
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Recommendations", len(user_recs))
            with col2:
                st.metric("Highest Score", f"{user_recs['hybrid_score'].max():.4f}")
            with col3:
                st.metric("Average Score", f"{user_recs['hybrid_score'].mean():.4f}")
        else:
            st.warning("No recommendations found for this user.")

def render_hybrid_score_analysis(hybrid_df):
    """Render hybrid score analysis"""
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
