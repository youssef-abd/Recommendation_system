import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
        font-weight: bold;
    }
    
    /* Data preview container */
    .data-preview {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Success/warning message styling */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def render_dashboard_header():
    """Render modern dashboard header"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Analytics Dashboard</h1>
        <p>Real-time insights into user behavior and product performance</p>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_metrics(user_ratings, product_ratings, transactions):
    """Render enhanced metrics with better styling"""
    st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    # Calculate metrics
    total_users = user_ratings['userId'].nunique()
    total_products = product_ratings['productId'].nunique() if 'productId' in product_ratings.columns else 0
    total_ratings = len(user_ratings)
    avg_rating = user_ratings['rating'].mean() if 'rating' in user_ratings.columns else 0
    
    # Create 4 columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{total_users:,}</p>
            <p class="metric-label">Total Users</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{total_products:,}</p>
            <p class="metric-label">Products</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{total_ratings:,}</p>
            <p class="metric-label">Total Ratings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{avg_rating:.1f}</p>
            <p class="metric-label">Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)

def render_quick_insights(user_ratings, product_ratings, transactions):
    """Render quick visual insights"""
    st.markdown('<div class="section-header">🔍 Quick Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution chart
        if 'rating' in user_ratings.columns:
            rating_dist = user_ratings['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_dist.index, 
                y=rating_dist.values,
                title="Rating Distribution",
                labels={'x': 'Rating', 'y': 'Count'},
                color=rating_dist.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top products by rating count
        if 'productId' in user_ratings.columns:
            top_products = user_ratings['productId'].value_counts().head(10)
            fig = px.pie(
                values=top_products.values,
                names=[f"Product {i}" for i in top_products.index],
                title="Top 10 Products by Rating Count"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True)

def render_enhanced_data_preview(user_ratings, product_ratings, transactions):
    """Render enhanced data preview with better styling"""
    st.markdown('<div class="section-header">📋 Data Explorer</div>', unsafe_allow_html=True)
    
    # Create tabs for different data views
    tab1, tab2, tab3 = st.tabs(["👥 User Ratings", "🛍️ Product Info", "💳 Transactions"])
    
    with tab1:
        render_enhanced_user_ratings(user_ratings)
    
    with tab2:
        render_enhanced_product_info(product_ratings)
    
    with tab3:
        render_enhanced_transactions(transactions)

def render_enhanced_user_ratings(user_ratings):
    """Enhanced user ratings preview with filters"""
    st.markdown("### User Ratings Analysis")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        if 'rating' in user_ratings.columns:
            min_rating = st.slider("Minimum Rating", 1, 5, 1)
            filtered_data = user_ratings[user_ratings['rating'] >= min_rating]
        else:
            filtered_data = user_ratings
    
    with col2:
        show_count = st.selectbox("Show records", [100, 500, 1000, "All"], index=2)
        if show_count != "All":
            filtered_data = filtered_data.head(show_count)
    
    # Display data with better formatting
    cols_to_show = ['userId', 'productId', 'rating']
    if 'formatted_date' in user_ratings.columns:
        cols_to_show.append('formatted_date')
    
    if not filtered_data.empty:
        st.dataframe(
            filtered_data[cols_to_show].sort_values('rating', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Add summary stats
        st.markdown("#### Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records Shown", len(filtered_data))
        with col2:
            if 'rating' in filtered_data.columns:
                st.metric("Average Rating", f"{filtered_data['rating'].mean():.2f}")
        with col3:
            st.metric("Unique Users", filtered_data['userId'].nunique())
    else:
        st.warning("No data matches the selected filters")

def render_enhanced_product_info(product_ratings):
    """Enhanced product info preview"""
    st.markdown("### Product Information")
    
    if not product_ratings.empty:
        unique_products = product_ratings.drop_duplicates(subset=['productId'], keep='first')
        
        # Search functionality
        search_term = st.text_input("🔍 Search products...")
        if search_term:
            unique_products = unique_products[
                unique_products.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
            ]
        
        st.dataframe(
            unique_products.head(1000).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
        
        st.info(f"Showing {len(unique_products)} unique products")
    else:
        st.warning("No product data available")

def render_enhanced_transactions(transactions):
    """Enhanced transactions preview"""
    st.markdown("### Transaction History")
    
    if not transactions.empty:
        transactions_display = transactions.copy()
        
        # Date processing
        if 'time' in transactions_display.columns:
            transactions_display['date'] = pd.to_datetime(
                transactions_display['time'], unit='s'
            ).dt.date
            transactions_display = transactions_display.drop(columns=['time'])
        
        # Date filter
        if 'date' in transactions_display.columns:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", 
                                         value=transactions_display['date'].min())
            with col2:
                end_date = st.date_input("End Date", 
                                       value=transactions_display['date'].max())
            
            transactions_display = transactions_display[
                (transactions_display['date'] >= start_date) & 
                (transactions_display['date'] <= end_date)
            ]
        
        # Display columns (excluding formatted_date)
        display_cols = [col for col in transactions_display.columns 
                       if col != 'formatted_date']
        
        # Sort by date
        date_col = None
        if 'date' in transactions_display.columns:
            date_col = 'date'
        elif 'transactionDate' in transactions_display.columns:
            date_col = 'transactionDate'
        
        if date_col:
            transactions_display = transactions_display.sort_values(
                date_col, ascending=False
            )
        
        st.dataframe(
            transactions_display[display_cols].head(1000),
            use_container_width=True,
            hide_index=True
        )
        
        st.info(f"Showing {len(transactions_display)} transactions")
    else:
        st.warning("No transaction data available")

def render(user_ratings, product_ratings, transactions):
    """Main render function with enhanced design"""
    # Apply custom CSS
    apply_custom_css()
    
    # Render dashboard header
    render_dashboard_header()
    
    # Render enhanced metrics
    render_enhanced_metrics(user_ratings, product_ratings, transactions)
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render quick insights
    render_quick_insights(user_ratings, product_ratings, transactions)
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render enhanced data preview
    render_enhanced_data_preview(user_ratings, product_ratings, transactions)