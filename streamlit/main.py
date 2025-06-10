import streamlit as st
from config.settings import PAGE_CONFIG, SIDEBAR_OPTIONS
from utils.styling import apply_custom_css
from data.data_loader import load_all_data
from components import (
    dashboard, user_analysis, product_analysis, 
    transactions, visualizations, recommendations
)

def main():
    # Page configuration
    st.set_page_config(**PAGE_CONFIG)
    
    # Apply custom styling
    apply_custom_css()
    
    # Load data
    user_ratings, product_ratings, transactions_data = load_all_data()
    
    # Sidebar navigation
    st.sidebar.header("🔍 Navigation")
    st.sidebar.markdown("---")
    
    view_option = st.sidebar.radio("Select View", SIDEBAR_OPTIONS)
    
    # Main title
    st.title("📊 E-commerce Analytics Dashboard")
    
    # Route to appropriate component based on selection
    if view_option == "📊 Dashboard Overview":
        dashboard.render(user_ratings, product_ratings, transactions_data)
    elif view_option == "👤 User Analysis":
        user_analysis.render(user_ratings)
    elif view_option == "📦 Product Analysis":
        product_analysis.render(product_ratings, user_ratings)
    elif view_option == "💳 Transactions":
        transactions.render(transactions_data)
    elif view_option == "📈 Visualizations":
        visualizations.render(user_ratings, product_ratings, transactions_data)
    elif view_option == "🎯 Recommendation Systems":
        recommendations.render(user_ratings, product_ratings)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        E-commerce Analytics Dashboard • Powered by Streamlit • Data last updated: 2023
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()