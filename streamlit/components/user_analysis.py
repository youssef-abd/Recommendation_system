# =============================================================================
# FILE: components/user_analysis.py
# =============================================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def render(user_ratings):
    """Render user analysis page"""
    st.subheader("👤 User Rating Analysis")
    
    tab1, tab2 = st.tabs(["🔎 Search User", "📊 All Users"])
    
    with tab1:
        render_user_search(user_ratings)
    
    with tab2:
        render_all_users_summary(user_ratings)

def render_user_search(user_ratings):
    """Render individual user search and analysis"""
    user_id = st.text_input("Enter User ID", help="Enter a specific user ID to view their ratings")
    
    if user_id:
        with st.spinner("Searching user data..."):
            user_data = user_ratings[user_ratings['userId'] == user_id]
            
            if not user_data.empty:
                st.success(f"Found {len(user_data)} ratings for user {user_id}")
                
                # Display user metrics
                display_user_metrics(user_data)
                
                # Display user ratings table
                display_user_ratings_table(user_data)
                
                # Display rating distribution chart
                display_user_rating_distribution(user_data, user_id)
                
                # Display rating timeline if date available
                if 'date' in user_data.columns:
                    display_user_rating_timeline(user_data)
            else:
                st.warning("No ratings found for this user.")

def display_user_metrics(user_data):
    """Display user-specific metrics"""
    col1, col2 = st.columns(2)
    
    with col1:
        avg_rating = user_data['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}")
    
    with col2:
        if 'normalized_rating' in user_data.columns:
            st.metric("Normalized Avg", f"{user_data['normalized_rating'].mean():.2f}")
        else:
            st.metric("Total Ratings", len(user_data))

def display_user_ratings_table(user_data):
    """Display user ratings in a table format"""
    display_cols = ['productId', 'rating']
    if 'formatted_date' in user_data.columns:
        display_cols.append('formatted_date')
    
    st.dataframe(
        user_data[display_cols].sort_values('rating', ascending=False),
        use_container_width=True
    )

def display_user_rating_distribution(user_data, user_id):
    """Display rating distribution chart for a specific user"""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(x='rating', data=user_data, ax=ax)
    plt.title(f"Rating Distribution for User {user_id}")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    st.pyplot(fig)

def display_user_rating_timeline(user_data):
    """Display user rating timeline"""
    st.subheader("Rating Timeline")
    timeline = user_data.set_index('date')['rating'].sort_index()
    st.line_chart(timeline)

def render_all_users_summary(user_ratings):
    """Render summary of all users"""
    st.subheader("All Users Summary")
    
    # Calculate user statistics
    user_stats = calculate_user_statistics(user_ratings)
    
    # Display user statistics table
    st.dataframe(
        user_stats.head(1000),
        use_container_width=True
    )
    
    # Display user engagement distribution
    display_user_engagement_distribution(user_stats)

def calculate_user_statistics(user_ratings):
    """Calculate statistics for all users"""
    user_stats = user_ratings.groupby('userId')['rating'].agg([
        ('Rating Count', 'count'),
        ('Avg Rating', 'mean'),
        ('Rating Std', 'std')
    ]).round(2)
    
    # Fill NaN values in std with 0 (for users with only one rating)
    user_stats['Rating Std'] = user_stats['Rating Std'].fillna(0)
    
    # Sort by rating count descending
    user_stats = user_stats.sort_values('Rating Count', ascending=False)
    
    return user_stats

def display_user_engagement_distribution(user_stats):
    """Display user engagement distribution charts"""
    st.subheader("User Engagement Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating count distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(user_stats['Rating Count'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Number of Ratings per User')
        ax.set_ylabel('Number of Users')
        ax.set_title('Distribution of User Activity')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Average rating distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(user_stats['Avg Rating'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Average Rating per User')
        ax.set_ylabel('Number of Users')
        ax.set_title('Distribution of User Average Ratings')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Summary statistics
    display_user_summary_stats(user_stats)

def display_user_summary_stats(user_stats):
    """Display summary statistics for all users"""
    st.subheader("User Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(user_stats))
    
    with col2:
        st.metric("Avg Ratings per User", f"{user_stats['Rating Count'].mean():.1f}")
    
    with col3:
        st.metric("Most Active User", f"{user_stats['Rating Count'].max()} ratings")
    
    with col4:
        st.metric("Overall Avg Rating", f"{user_stats['Avg Rating'].mean():.2f}")