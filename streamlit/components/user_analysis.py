import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def render(user_ratings):
    """Render enhanced user analysis page"""
    st.title("👤 User Rating Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar filters
    setup_sidebar_filters(user_ratings)
    
    # Main content - simplified to just show user explorer
    render_user_explorer(user_ratings)

def setup_sidebar_filters(user_ratings):
    """Setup sidebar with global filters"""
    st.sidebar.header("🎛️ Filters")
    
    # Rating range filter
    min_rating, max_rating = st.sidebar.select_slider(
        "Rating Range",
        options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        value=(1.0, 5.0),
        format_func=lambda x: f"⭐ {x}"
    )
    
    # Minimum ratings per user filter
    min_user_ratings = st.sidebar.number_input(
        "Min Ratings per User",
        min_value=1,
        max_value=100,
        value=1,
        help="Filter users with at least this many ratings"
    )
    
    # Store filters in session state
    st.session_state.rating_range = (min_rating, max_rating)
    st.session_state.min_user_ratings = min_user_ratings

def render_user_explorer(user_ratings):
    """Enhanced user search and exploration"""
    st.header("🔍 User Explorer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Smart user search with autocomplete simulation
        search_option = st.selectbox(
            "Search Method",
            ["Search by User ID", "Find Top Users", "Random User Sample"]
        )
    
    with col2:
        if search_option == "Find Top Users":
            top_n = st.number_input("Number of users", min_value=1, max_value=50, value=10)
    
    if search_option == "Search by User ID":
        render_user_id_search(user_ratings)
    elif search_option == "Find Top Users":
        render_top_users(user_ratings, top_n)
    else:
        render_random_users(user_ratings)

def render_user_id_search(user_ratings):
    """Enhanced individual user search"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_id = st.text_input(
            "🔎 Enter User ID", 
            placeholder="e.g., A1234567890",
            help="Search for a specific user's rating history"
        )
    
    with col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True)
    
    if user_id and (search_clicked or user_id):
        user_data = user_ratings[user_ratings['userId'] == user_id]
        
        if not user_data.empty:
            render_user_profile(user_data, user_id)
        else:
            st.error("❌ No ratings found for this user.")
            st.info("💡 Try searching for a different user ID or use the 'Find Top Users' option.")

def render_user_profile(user_data, user_id):
    """Render comprehensive user profile"""
    st.success(f"✅ Found **{len(user_data)}** ratings for user **{user_id}**")
    
    # User metrics cards
    render_user_metrics_cards(user_data)
    
    # User analysis tabs - keeping only the relevant tabs
    profile_tab1, profile_tab2 = st.tabs([
        "📋 Rating History", 
        "📊 Rating Patterns & Insights"
    ])
    
    with profile_tab1:
        render_user_rating_history(user_data)
    
    with profile_tab2:
        render_user_rating_patterns(user_data, user_id)
        st.markdown("---")
        render_individual_user_insights(user_data, user_id)

def render_user_metrics_cards(user_data):
    """Render user metrics in card format"""
    col1, col2, col3, col4 = st.columns(4)
    
    avg_rating = user_data['rating'].mean()
    rating_std = user_data['rating'].std()
    total_ratings = len(user_data)
    rating_range = user_data['rating'].max() - user_data['rating'].min()
    
    with col1:
        st.metric(
            "📊 Average Rating", 
            f"{avg_rating:.2f}",
            delta=f"{avg_rating - 3.0:+.2f} vs neutral",
            help="Average rating compared to neutral (3.0)"
        )
    
    with col2:
        st.metric(
            "📈 Total Ratings", 
            f"{total_ratings:,}",
            help="Total number of ratings given"
        )
    
    with col3:
        st.metric(
            "📏 Rating Spread", 
            f"{rating_std:.2f}",
            help="Standard deviation of ratings (consistency measure)"
        )
    
    with col4:
        st.metric(
            "🎯 Rating Range", 
            f"{rating_range:.1f}",
            help="Difference between highest and lowest rating"
        )

def render_user_rating_history(user_data):
    """Render user rating history table with enhanced features"""
    st.subheader("📋 Rating History")
    
    # Sorting options
    col1, col2 = st.columns([1, 1])
    with col1:
        sort_by = st.selectbox("Sort by", ["Rating (High to Low)", "Rating (Low to High)", "Product ID"])
    
    with col2:
        show_all = st.checkbox("Show all columns", value=False)
    
    # Prepare data
    display_data = user_data.copy()
    
    if sort_by == "Rating (High to Low)":
        display_data = display_data.sort_values('rating', ascending=False)
    elif sort_by == "Rating (Low to High)":
        display_data = display_data.sort_values('rating', ascending=True)
    else:
        display_data = display_data.sort_values('productId')
    
    # Select columns to display
    if show_all:
        display_cols = display_data.columns.tolist()
    else:
        display_cols = ['productId', 'rating']
        if 'formatted_date' in display_data.columns:
            display_cols.append('formatted_date')
        if 'helpfulness' in display_data.columns:
            display_cols.append('helpfulness')
    
    # Add rating emoji
    if 'rating_emoji' not in display_data.columns:
        display_data['rating_emoji'] = display_data['rating'].apply(get_rating_emoji)
        display_cols = ['rating_emoji'] + display_cols
    
    st.dataframe(
        display_data[display_cols],
        use_container_width=True,
        height=400
    )

def get_rating_emoji(rating):
    """Convert rating to emoji representation"""
    if rating >= 4.5:
        return "🌟"
    elif rating >= 4.0:
        return "⭐"
    elif rating >= 3.0:
        return "😐"
    elif rating >= 2.0:
        return "👎"
    else:
        return "💔"

def render_user_rating_patterns(user_data, user_id):
    """Render user rating patterns with interactive charts"""
    st.subheader("📊 Rating Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactive rating distribution
        fig = px.histogram(
            user_data, 
            x='rating', 
            nbins=10,
            title=f"Rating Distribution - User {user_id}",
            labels={'rating': 'Rating', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rating trend over time (if date available)
        if 'date' in user_data.columns:
            render_user_timeline_chart(user_data, user_id)
        else:
            # Rating statistics pie chart
            render_user_rating_breakdown(user_data)

def render_user_timeline_chart(user_data, user_id):
    """Render interactive timeline chart"""
    timeline_data = user_data.sort_values('date')
    
    fig = px.line(
        timeline_data,
        x='date',
        y='rating',
        title=f"Rating Timeline - User {user_id}",
        markers=True,
        line_shape='linear'
    )
    fig.update_traces(line=dict(color='#2E8B57', width=3))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_user_rating_breakdown(user_data):
    """Render rating breakdown pie chart"""
    rating_counts = user_data['rating'].value_counts().sort_index()
    
    fig = px.pie(
        values=rating_counts.values,
        names=[f"⭐ {r}" for r in rating_counts.index],
        title="Rating Breakdown"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_individual_user_insights(user_data, user_id):
    """Render insights specific to individual user"""
    st.subheader("🎯 User Insights")
    
    # User behavior analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎭 Rating Personality")
        avg_rating = user_data['rating'].mean()
        rating_std = user_data['rating'].std()
        
        if avg_rating >= 4.0:
            personality = "😊 **Positive Reviewer** - Tends to give high ratings"
        elif avg_rating <= 2.5:
            personality = "😤 **Critical Reviewer** - Tends to give low ratings"
        else:
            personality = "😐 **Balanced Reviewer** - Gives moderate ratings"
        
        st.markdown(personality)
        
        if rating_std <= 0.5:
            consistency = "🎯 **Very Consistent** - Ratings don't vary much"
        elif rating_std <= 1.0:
            consistency = "📊 **Somewhat Consistent** - Moderate rating variation"
        else:
            consistency = "🎲 **Highly Variable** - Ratings vary significantly"
        
        st.markdown(consistency)
    
    with col2:
        st.markdown("#### 📈 Quick Stats")
        most_common_rating = user_data['rating'].mode().iloc[0]
        least_common_rating = user_data['rating'].value_counts().idxmin()
        
        st.markdown(f"**Most Given Rating:** ⭐ {most_common_rating}")
        st.markdown(f"**Least Given Rating:** ⭐ {least_common_rating}")
        st.markdown(f"**Rating Variance:** {user_data['rating'].var():.2f}")

def render_top_users(user_ratings, top_n):
    """Render top users analysis"""
    st.subheader(f"🏆 Top {top_n} Most Active Users")
    
    user_stats = calculate_enhanced_user_statistics(user_ratings)
    top_users = user_stats.head(top_n)
    
    # Interactive bar chart
    fig = px.bar(
        x=top_users.index.astype(str),
        y=top_users['Rating Count'],
        title=f"Top {top_n} Users by Rating Count",
        labels={'x': 'User ID', 'y': 'Number of Ratings'},
        color=top_users['Avg Rating'],
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.dataframe(top_users, use_container_width=True)

def render_random_users(user_ratings):
    """Render random user sample"""
    st.subheader("🎲 Random User Sample")
    
    sample_size = st.slider("Sample Size", min_value=5, max_value=50, value=10)
    
    if st.button("Generate Sample", type="primary"):
        user_stats = calculate_enhanced_user_statistics(user_ratings)
        sample_users = user_stats.sample(n=min(sample_size, len(user_stats)))
        
        st.dataframe(sample_users, use_container_width=True)

def calculate_enhanced_user_statistics(user_ratings):
    """Calculate enhanced statistics for all users"""
    user_stats = user_ratings.groupby('userId')['rating'].agg([
        ('Rating Count', 'count'),
        ('Avg Rating', 'mean'),
        ('Rating Std', 'std'),
        ('Min Rating', 'min'),
        ('Max Rating', 'max')
    ]).round(2)
    
    # Fill NaN values
    user_stats['Rating Std'] = user_stats['Rating Std'].fillna(0)
    
    # Calculate additional metrics
    user_stats['Rating Range'] = user_stats['Max Rating'] - user_stats['Min Rating']
    
    # Sort by rating count descending
    user_stats = user_stats.sort_values('Rating Count', ascending=False)
    
    return user_stats