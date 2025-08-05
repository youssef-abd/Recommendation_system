import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.data_loader import load_recommendations_data
from config.settings import DEFAULT_NUM_RECOMMENDATIONS

# Set page configuration and styling
def apply_custom_styling():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .recommendation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 0.5rem;
    }
    
    .success-banner {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .warning-banner {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def render(user_ratings, product_ratings):
    """Enhanced main render function with improved UI"""
    apply_custom_styling()
    
    # Header with enhanced styling
    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI-Powered Recommendation Engine</h1>
        <p>Discover personalized recommendations using advanced machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Method selection with enhanced UI
    col1, col2 = st.columns([3, 1])
    
    with col1:
        rec_method = st.selectbox(
            "🔍 Choose Your Recommendation Strategy",
            ["Collaborative Filtering (ALS)", "Content-Based Filtering", "Hybrid Intelligence"],
            help="Select the recommendation algorithm that best fits your needs"
        )
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if hasattr(product_ratings, 'shape'):
            st.metric("Products", f"{product_ratings.shape[0]:,}")
    
    # Route to appropriate rendering function
    if rec_method == "Collaborative Filtering (ALS)":
        render_collaborative_filtering_enhanced(product_ratings)
    elif rec_method == "Content-Based Filtering":
        render_content_based_enhanced(product_ratings)
    elif rec_method == "Hybrid Intelligence":
        render_hybrid_recommendations_enhanced()

def render_collaborative_filtering_enhanced(product_ratings):
    """Enhanced collaborative filtering with modern UI"""
    st.markdown("### 🤝 Collaborative Filtering Recommendations")
    st.markdown("*Leveraging user behavior patterns to find products you'll love*")
    
    try:
        with st.spinner("🔄 Analyzing user preferences and generating recommendations..."):
            recs_df = load_recommendations_data('als')
            
            if recs_df is None:
                render_error_state("Collaborative filtering data not found", 
                                 "Please ensure the ALS model has been trained and data is available.")
                return
            
            # Enhanced validation
            if not validate_dataframe(recs_df, ['productId', 'userId', 'predictedRating']):
                return
            
            # Data preprocessing with progress indication
            recs_df = preprocess_recommendations(recs_df, product_ratings)
            
            # Success banner
            st.markdown(f"""
            <div class="success-banner">
                ✅ Successfully loaded {len(recs_df):,} recommendations for {recs_df['userId'].nunique():,} users
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced tabs with icons
            tab1, tab2, tab3 = st.tabs(["👤 Personal Recommendations", "🌟 Trending Products", "📈 Analytics Dashboard"])
            
            with tab1:
                render_user_recommendations_enhanced(recs_df)
            
            with tab2:
                render_global_recommendations_enhanced(recs_df, product_ratings)
            
            with tab3:
                render_collaborative_analytics(recs_df)
                
    except Exception as e:
        render_error_state("Collaborative Filtering Error", str(e))

def render_user_recommendations_enhanced(recs_df):
    """Enhanced user recommendations with modern interface"""
    st.markdown("#### 🎯 Get Personalized Recommendations")
    
    # Enhanced user input section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Smart user selection
        available_users = sorted(recs_df['userId'].unique())
        user_input = st.selectbox(
            "Select User ID",
            options=[""] + available_users[:100],  # Limit for performance
            help=f"Choose from {len(available_users)} available users"
        )
    
    with col2:
        num_recs = st.slider("📊 Number of items", 5, 50, DEFAULT_NUM_RECOMMENDATIONS)
    
    with col3:
        sort_by = st.selectbox("Sort by", ["Rating", "Product ID"])
    
    if user_input:
        user_recs = recs_df[recs_df['userId'] == user_input]
        
        if not user_recs.empty:
            # Sort and limit results
            sort_col = 'predictedRating' if sort_by == "Rating" else 'productId'
            user_recs = user_recs.sort_values(sort_col, ascending=False).head(num_recs)
            user_recs['rank'] = range(1, len(user_recs) + 1)
            
            # Enhanced display with cards
            st.markdown(f"#### 🏆 Top {num_recs} Recommendations for User {user_input}")
            
            for idx, row in user_recs.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"**#{row['rank']}**")
                    
                    with col2:
                        product_desc = row.get('description', f"Product {row['productId']}")
                        st.markdown(f"**{product_desc}**")
                        st.markdown(f"Product ID: `{row['productId']}`")
            
            # Enhanced visualization with Plotly
            render_interactive_rating_chart(user_recs, user_input)
            
        else:
            render_empty_state("No recommendations found", 
                             f"User {user_input} doesn't have any recommendations yet.")

def render_interactive_rating_chart(user_recs, user_id):
    """Create interactive rating visualization"""
    if len(user_recs) > 1:
        fig = px.bar(
            user_recs, 
            x='rank', 
            y='predictedRating',
            title=f'Recommendation Scores for User {user_id}',
            labels={'rank': 'Recommendation Rank', 'predictedRating': 'Predicted Rating'},
            color='predictedRating',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_global_recommendations_enhanced(recs_df, product_ratings):
    """Enhanced global recommendations with analytics"""
    st.markdown("#### 🌟 Discover Trending Products")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        num_global = st.slider("Number of trending items", 5, 100, 20)
    
    with col2:
        min_recommendations = st.slider("Minimum recommendation count", 1, 100, 10)
    
    # Calculate trending products with filtering and remove duplicates
    trending_products = calculate_trending_products(recs_df, num_global, min_recommendations)
    trending_products = trending_products.drop_duplicates(subset=['productId'])
    
    if not trending_products.empty:
        # Merge with product descriptions
        if 'description' in product_ratings.columns:
            trending_products = trending_products.merge(
                product_ratings[['productId', 'description']], 
                on='productId', 
                how='left'
            )
        
        # Display trending products
        st.markdown("##### 🔥 Most Popular Products")
        
        for idx, row in trending_products.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"**#{row['rank']}**")
                
                with col2:
                    product_name = row.get('description', f"Product {row['productId']}")
                    st.markdown(f"**{product_name}**")
                    st.markdown(f"ID: `{row['productId']}`")
        
        # Trending visualization
        render_trending_visualization(trending_products)

def render_trending_visualization(trending_products):
    """Create trending products visualization"""
    fig = px.bar(
        trending_products,
        x='productId',
        y='AvgScore',
        title='Top Trending Products by Average Score',
        labels={'productId': 'Product ID', 'AvgScore': 'Average Score'}
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_content_based_enhanced(product_ratings):
    """Enhanced content-based recommendations"""
    st.markdown("### 🎨 Content-Based Recommendations")
    st.markdown("*Find products similar to what you already love*")
    
    try:
        content_df = load_recommendations_data('content')
        
        if content_df is None:
            render_error_state("Content-based data not found", 
                             "Please ensure content similarity has been computed.")
            return
        
        if not validate_dataframe(content_df, ['source_product_id', 'recommended_product_id', 'similarity_score']):
            return
        
        # Enhanced product selection interface
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            all_products = sorted(content_df['source_product_id'].unique())
            product_input = st.selectbox(
                "🔍 Select a Product",
                options=all_products,
                help="Choose a product to find similar items"
            )
        
        with col2:
            num_recs = st.slider("Similar items", 1, 50, 10)
        
        with col3:
            min_similarity = st.slider("Min similarity", 0.0, 1.0, 0.1, 0.1)
        
        if product_input:
            render_similar_products_enhanced(content_df, product_ratings, product_input, num_recs, min_similarity)
            
    except Exception as e:
        render_error_state("Content-Based Error", str(e))

def render_similar_products_enhanced(content_df, product_ratings, product_input, num_recs, min_similarity):
    """Enhanced similar products display"""
    similar_products = content_df[
        (content_df['source_product_id'] == product_input) & 
        (content_df['similarity_score'] >= min_similarity)
    ].sort_values('similarity_score', ascending=False)
    
    if not similar_products.empty:
        # Success message with stats
        st.markdown(f"""
        <div class="success-banner">
            🎯 Found {len(similar_products)} similar products with similarity ≥ {min_similarity}
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced display
        display_df = similar_products.head(num_recs).copy()
        display_df['rank'] = range(1, len(display_df) + 1)
        
        # Add product descriptions if available
        if 'productId' in product_ratings.columns and 'description' in product_ratings.columns:
            display_df = display_df.merge(
                product_ratings[['productId', 'description']], 
                left_on='recommended_product_id', 
                right_on='productId', 
                how='left'
            )
        
        # Card-based display
        for idx, row in display_df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    st.markdown(f"**#{row['rank']}**")
                
                with col2:
                    product_name = row.get('description', f"Product {row['recommended_product_id']}")
                    st.markdown(f"**{product_name}**")
                    st.markdown(f"ID: `{row['recommended_product_id']}`")
                
                with col3:
                    similarity_pct = row['similarity_score'] * 100
                    st.metric("Similarity", f"{similarity_pct:.1f}%")
        
        # Interactive similarity visualization
        render_similarity_chart(display_df, product_input)
        
    else:
        render_empty_state("No similar products found", 
                         f"Try lowering the similarity threshold or selecting a different product.")

def render_similarity_chart(display_df, product_input):
    """Create interactive similarity chart"""
    fig = px.bar(
        display_df,
        x='similarity_score',
        y='recommended_product_id',
        orientation='h',
        title=f'Product Similarity to {product_input}',
        labels={'similarity_score': 'Similarity Score', 'recommended_product_id': 'Product ID'},
        color='similarity_score',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=max(300, len(display_df) * 40),
        showlegend=False,
        title_x=0.5,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_hybrid_recommendations_enhanced():
    """Enhanced hybrid recommendations interface"""
    st.markdown("### 🧠 Hybrid Intelligence Recommendations")
    st.markdown("*Combining multiple AI approaches for superior recommendations*")
    
    try:
        with st.spinner("🔄 Processing hybrid intelligence algorithms..."):
            hybrid_df = load_and_process_hybrid_data()
            
            if hybrid_df is None or hybrid_df.empty:
                render_error_state("Hybrid data not available", 
                                 "Please ensure hybrid recommendations have been generated.")
                return
            
            # Success metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Users", f"{hybrid_df['user_id'].nunique():,}")
            with col2:
                st.metric("Total Items", f"{hybrid_df['item_id'].nunique():,}")
            with col3:
                st.metric("Recommendations", f"{len(hybrid_df):,}")
            with col4:
                st.metric("Avg Hybrid Score", f"{hybrid_df['hybrid_score'].mean():.3f}")
            
            # Enhanced tabs
            tab1, tab2, tab3 = st.tabs(["👤 Personal AI", "📊 Intelligence Analytics", "🔬 Model Insights"])
            
            with tab1:
                render_hybrid_user_recommendations_enhanced(hybrid_df)
            
            with tab2:
                render_hybrid_analytics_enhanced(hybrid_df)
            
            with tab3:
                render_hybrid_insights(hybrid_df)
                
    except Exception as e:
        render_error_state("Hybrid Intelligence Error", str(e))

def load_and_process_hybrid_data():
    """Load and process hybrid recommendation data with proper column mapping"""
    hybrid_df = load_recommendations_data('hybrid')
    
    if hybrid_df is None:
        return None
    
    # Map the CSV columns to our expected column names
    column_mapping = {
        'user_id': 'user_id',
        'product_id': 'item_id',
        'cf_score': 'cf_score',
        'cb_score': 'cb_score',
        'popularity_score': 'popularity_score',
        'hybrid_score': 'hybrid_score'
    }
    
    # Rename columns to match our expected format
    hybrid_df = hybrid_df.rename(columns=column_mapping)
    
    # Enhanced data cleaning
    hybrid_df = hybrid_df.dropna(subset=['hybrid_score'])
    hybrid_df = hybrid_df.drop_duplicates(subset=['user_id', 'item_id'])
    
    # Ensure proper data types
    hybrid_df['user_id'] = hybrid_df['user_id'].astype(str)
    hybrid_df['item_id'] = hybrid_df['item_id'].astype(str)
    hybrid_df['hybrid_score'] = pd.to_numeric(hybrid_df['hybrid_score'], errors='coerce')
    hybrid_df['cf_score'] = pd.to_numeric(hybrid_df['cf_score'], errors='coerce')
    hybrid_df['cb_score'] = pd.to_numeric(hybrid_df['cb_score'], errors='coerce')
    hybrid_df['popularity_score'] = pd.to_numeric(hybrid_df['popularity_score'], errors='coerce')
    
    return hybrid_df

def render_hybrid_user_recommendations_enhanced(hybrid_df):
    """Enhanced hybrid user recommendations with customizable view options"""
    st.markdown("#### 🎯 AI-Powered Personal Recommendations")
    
    # Configuration columns
    config_col1, config_col2, config_col3 = st.columns([2, 1, 1])
    
    with config_col1:
        available_users = sorted(hybrid_df['user_id'].unique())
        user_input = st.selectbox(
            "Select User",
            options=available_users[:200],
            help=f"Choose from {len(available_users)} users"
        )
    
    with config_col2:
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.01, 
                            help="Filter recommendations by minimum hybrid score")
    
    with config_col3:
        default_num = 10  # Default number of recommendations to show
        max_possible = min(100, len(hybrid_df))  # Don't allow more than available
        num_recs = st.slider("Items to show", 5, max_possible, default_num,
                            help=f"Choose how many recommendations to display (up to {max_possible})")
    
    if user_input:
        # Get filtered and sorted recommendations
        user_recs = hybrid_df[
            (hybrid_df['user_id'] == user_input) & 
            (hybrid_df['hybrid_score'] >= min_score)
        ].sort_values('hybrid_score', ascending=False)
        
        if not user_recs.empty:
            # Show total available count
            st.markdown(f"**Found {len(user_recs)} recommendations matching your criteria**")
            
            # Let user choose between compact and detailed view
            view_mode = st.radio("View Mode", 
                               ["Compact List", "Detailed Cards"], 
                               horizontal=True,
                               help="Choose how to display the recommendations")
            
            # Show the top recommendations based on user's choice
            top_recs = user_recs.head(num_recs).copy()
            top_recs['rank'] = range(1, len(top_recs) + 1)
            
            if view_mode == "Compact List":
                # Display as a compact table with key metrics
                st.markdown(f"### 🏆 Top {len(top_recs)} Recommendations (Compact View)")
                
                # Prepare compact display dataframe
                compact_df = top_recs[['rank', 'item_id', 'hybrid_score']].copy()
                compact_df.columns = ['Rank', 'Item ID', 'Hybrid Score']
                
                # Format for better display
                compact_df['Hybrid Score'] = compact_df['Hybrid Score'].round(3)
                
                # Show interactive table with all selected recommendations
                st.dataframe(
                    compact_df,
                    use_container_width=True,
                    height=min(400, (len(top_recs) + 1) * 35),  # Dynamic height
                    column_config={
                        "Rank": st.column_config.NumberColumn(width="small"),
                        "Item ID": st.column_config.TextColumn(width="medium"),
                        "Hybrid Score": st.column_config.ProgressColumn(
                            "Hybrid Score",
                            help="The hybrid recommendation score",
                            format="%.3f",
                            min_value=0,
                            max_value=1
                        )
                    }
                )
                
                # Quick actions below the table
                with st.expander("⚡ Quick Actions"):
                    action_col1, action_col2 = st.columns(2)
                    with action_col1:
                        if st.button("Show All Scores", help="Reveal all score components"):
                            view_mode = "Detailed Cards"
                            st.experimental_rerun()
                    with action_col2:
                        show_all = st.checkbox("Show Full Dataset", False)
                        if show_all:
                            st.dataframe(top_recs, use_container_width=True)
            
            else:  # Detailed Cards view
                st.markdown(f"### 🏆 Top {len(top_recs)} Recommendations (Detailed View)")
                
                # Configuration for score components to show
                st.sidebar.markdown("### 🔧 Display Configuration")
                show_cf = st.sidebar.checkbox("Show Collaborative Score", True)
                show_cb = st.sidebar.checkbox("Show Content Score", True)
                show_pop = st.sidebar.checkbox("Show Popularity Score", True)
                show_hybrid = st.sidebar.checkbox("Show Hybrid Score", True)
                
                # Check for additional components
                additional_components = [col for col in hybrid_df.columns 
                                      if col.endswith('_score') and 
                                      col not in ['cf_score', 'cb_score', 'popularity_score', 'hybrid_score']]
                
                additional_checks = {}
                for comp in additional_components:
                    additional_checks[comp] = st.sidebar.checkbox(
                        f"Show {comp.replace('_', ' ').title()}", 
                        True
                    )
                
                # Display each recommendation with all selected components
                for idx, row in top_recs.iterrows():
                    with st.expander(f"#{row['rank']}: Item {row['item_id']} (Score: {row['hybrid_score']:.3f})"):
                        # Dynamic columns based on selected components
                        visible_components = []
                        if show_hybrid: visible_components.append(('Hybrid Score', 'hybrid_score'))
                        if show_cf: visible_components.append(('Collaborative', 'cf_score'))
                        if show_cb: visible_components.append(('Content', 'cb_score'))
                        if show_pop: visible_components.append(('Popularity', 'popularity_score'))
                        
                        # Add any additional components
                        for comp, col in additional_checks.items():
                            if col:
                                comp_name = comp.replace('_', ' ').title()
                                visible_components.append((comp_name, comp))
                        
                        # Create columns dynamically
                        cols = st.columns(len(visible_components))
                        
                        for i, (name, col) in enumerate(visible_components):
                            with cols[i]:
                                st.metric(name, f"{row[col]:.3f}")
                        
                        # Visualization of selected components
                        if len(visible_components) > 1:
                            scores_data = {
                                'Component': [name for name, col in visible_components],
                                'Score': [row[col] for name, col in visible_components]
                            }
                            
                            tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
                            
                            with tab1:
                                fig = px.bar(
                                    pd.DataFrame(scores_data),
                                    x='Component',
                                    y='Score',
                                    title='Score Breakdown',
                                    color='Component'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab2:
                                fig = px.pie(
                                    pd.DataFrame(scores_data),
                                    names='Component',
                                    values='Score',
                                    title='Score Composition'
                                )
                                st.plotly_chart(fig, use_container_width=True)
            
            # Overall visualization section (shown in both views)
            st.markdown("### 📊 Recommendations Analysis")
            
            # Prepare data for visualization
            viz_df = top_recs.copy()
            
            # Add all selected components to visualization
            components_to_show = []
            if show_cf: components_to_show.append('cf_score')
            if show_cb: components_to_show.append('cb_score')
            if show_pop: components_to_show.append('popularity_score')
            if show_hybrid: components_to_show.append('hybrid_score')
            
            # Add additional components if selected
            for comp, show in additional_checks.items():
                if show:
                    components_to_show.append(comp)
            
            # Melt the dataframe for visualization
            melted_df = viz_df.melt(
                id_vars=['rank', 'item_id'],
                value_vars=components_to_show,
                var_name='Score Type',
                value_name='Score Value'
            )
            
            # Clean up score type names
            melted_df['Score Type'] = melted_df['Score Type'].str.replace('_score', '').str.replace('_', ' ').str.title()
            
            # Interactive visualization
            fig = px.line(
                melted_df,
                x='rank',
                y='Score Value',
                color='Score Type',
                title=f'Score Components by Recommendation Rank for User {user_input}',
                markers=True
            )
            
            fig.update_layout(
                xaxis_title='Recommendation Rank',
                yaxis_title='Score Value',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed score analysis section
            if view_mode == "Detailed Cards":
                st.markdown("### 🔍 Detailed Item Analysis")
                selected_item = st.selectbox(
                    "Select an item to analyze in detail",
                    options=top_recs['item_id'].tolist(),
                    format_func=lambda x: f"Item {x} (Rank {top_recs[top_recs['item_id'] == x]['rank'].values[0]})"
                )
                
                if selected_item:
                    item_data = top_recs[top_recs['item_id'] == selected_item].iloc[0]
                    
                    st.markdown(f"#### Detailed Analysis for Item {selected_item}")
                    
                    # Create score data for visualization
                    score_data = []
                    if show_cf: score_data.append(('Collaborative', item_data['cf_score']))
                    if show_cb: score_data.append(('Content', item_data['cb_score']))
                    if show_pop: score_data.append(('Popularity', item_data['popularity_score']))
                    if show_hybrid: score_data.append(('Hybrid', item_data['hybrid_score']))
                    
                    # Add additional components if selected
                    for comp, show in additional_checks.items():
                        if show:
                            comp_name = comp.replace('_', ' ').title()
                            score_data.append((comp_name, item_data[comp]))
                    
                    score_df = pd.DataFrame(score_data, columns=['Component', 'Score'])
                    
                    # Visualization tabs
                    tab1, tab2, tab3 = st.tabs(["Pie Chart", "Bar Chart", "Radar Chart"])
                    
                    with tab1:
                        fig = px.pie(
                            score_df,
                            names='Component',
                            values='Score',
                            title='Score Composition'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        fig = px.bar(
                            score_df,
                            x='Component',
                            y='Score',
                            color='Component',
                            text='Score',
                            title='Score Components'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        fig = px.line_polar(
                            score_df,
                            r='Score',
                            theta='Component',
                            line_close=True,
                            title='Score Radar Chart'
                        )
                        fig.update_traces(fill='toself')
                        st.plotly_chart(fig, use_container_width=True)
        else:
            render_empty_state("No recommendations meet criteria", 
                             "Try adjusting the score threshold or selecting a different user.")      

def render_hybrid_score_chart(user_recs, user_id):
    """Create hybrid score visualization"""
    fig = px.bar(
        user_recs.reset_index(),
        x=range(len(user_recs)),
        y='hybrid_score',
        title=f'Hybrid AI Scores for User {user_id}',
        labels={'x': 'Recommendation Rank', 'hybrid_score': 'AI Confidence Score'},
        color='hybrid_score',
        color_continuous_scale='Plasma'
    )
    
    fig.update_layout(
        showlegend=False,
        height=300,
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_hybrid_analytics_enhanced(hybrid_df):
    """Enhanced hybrid analytics dashboard with score components"""
    st.markdown("#### 📊 Intelligence Analytics Dashboard")
    
    # Create tabs for different analytics views
    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Component Analysis", "Top Performers"])
    
    with tab1:
        # Score distribution analysis
        fig = px.histogram(
            hybrid_df,
            x='hybrid_score',
            nbins=50,
            title='Hybrid Score Distribution',
            labels={'hybrid_score': 'Hybrid Score', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Correlation between score components
        st.markdown("##### 🔗 Score Component Relationships")
        
        # Scatter plot matrix
        fig = px.scatter_matrix(
            hybrid_df,
            dimensions=['cf_score', 'cb_score', 'popularity_score', 'hybrid_score'],
            title='Relationship Between Score Components'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        corr_matrix = hybrid_df[['cf_score', 'cb_score', 'popularity_score', 'hybrid_score']].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title='Score Component Correlation'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Top performing items by hybrid score
        top_items = (
            hybrid_df.groupby('item_id')
            .agg({
                'hybrid_score': 'mean',
                'cf_score': 'mean',
                'cb_score': 'mean',
                'popularity_score': 'mean',
                'user_id': 'count'
            })
            .rename(columns={'user_id': 'recommendation_count'})
            .sort_values('hybrid_score', ascending=False)
            .head(20)
            .reset_index()
        )
        
        st.markdown("##### 🏆 Top Performing Items by Hybrid Score")
        st.dataframe(
            top_items.round(3),
            use_container_width=True
        )

def render_hybrid_insights(hybrid_df):
    """Render model insights and statistics with component weights"""
    st.markdown("#### 🔬 Model Intelligence Insights")
    
    # Score component distributions
    st.markdown("##### 📊 Score Component Distributions")
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Collaborative Filtering Scores", 
        "Content-Based Scores",
        "Popularity Scores",
        "Hybrid Scores"
    ))
    
    fig.add_trace(go.Histogram(x=hybrid_df['cf_score'], name='CF Scores'), row=1, col=1)
    fig.add_trace(go.Histogram(x=hybrid_df['cb_score'], name='CB Scores'), row=1, col=2)
    fig.add_trace(go.Histogram(x=hybrid_df['popularity_score'], name='Popularity'), row=2, col=1)
    fig.add_trace(go.Histogram(x=hybrid_df['hybrid_score'], name='Hybrid'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("##### 📈 Statistical Summary")
    
    stats_df = hybrid_df[['cf_score', 'cb_score', 'popularity_score', 'hybrid_score']].describe().T
    st.dataframe(stats_df.round(3), use_container_width=True)

def render_collaborative_analytics(recs_df):
    """Analytics dashboard for collaborative filtering"""
    st.markdown("#### 📈 Collaborative Analytics")
    
    # Rating distribution
    fig = px.histogram(
        recs_df,
        x='predictedRating',
        nbins=30,
        title='Predicted Rating Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)

# Utility functions for enhanced UI
def render_error_state(title, message):
    """Render consistent error state"""
    st.markdown(f"""
    <div class="warning-banner">
        ⚠️ <strong>{title}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def render_empty_state(title, message):
    """Render consistent empty state"""
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px;">
        <h3>📭 {title}</h3>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)

def validate_dataframe(df, required_columns):
    """Validate dataframe has required columns"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        render_error_state("Data validation failed", f"Missing columns: {', '.join(missing_cols)}")
        return False
    return True

def preprocess_recommendations(recs_df, product_ratings):
    """Preprocess recommendations data"""
    # Merge with product descriptions if available
    if 'description' in product_ratings.columns:
        recs_df = recs_df.merge(
            product_ratings[['productId', 'description']], 
            on='productId', 
            how='left'
        )
    
    # Remove duplicates
    recs_df = recs_df.drop_duplicates(subset=['userId', 'productId'])
    
    return recs_df

def calculate_trending_products(recs_df, num_global, min_recommendations):
    """Calculate trending products with filtering"""
    trending = (
        recs_df.groupby('productId')['predictedRating']
        .agg(['mean', 'count'])
        .rename(columns={'mean': 'AvgScore', 'count': 'NumRecommendations'})
        .query(f'NumRecommendations >= {min_recommendations}')
        .sort_values('AvgScore', ascending=False)
        .head(num_global)
        .reset_index()
    )
    
    trending['rank'] = range(1, len(trending) + 1)
    return trending