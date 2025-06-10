import streamlit as st
import pandas as pd

def render(user_ratings, product_ratings, transactions):
    """Render dashboard overview"""
    st.subheader("📌 Key Metrics")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", user_ratings['userId'].nunique())
    with col2:
        st.metric("Total Products", 
                 product_ratings['productId'].nunique() if 'productId' in product_ratings.columns else "N/A")
    with col3:
        st.metric("Total Ratings", len(user_ratings))
    
    st.markdown("---")
    st.subheader("🔍 Quick Data Preview")
    
    preview_option = st.radio("Select data to preview", 
                            ["User Ratings", "Product Info", "Transactions"])
    
    if preview_option == "User Ratings":
        render_user_ratings_preview(user_ratings)
    elif preview_option == "Product Info":
        render_product_info_preview(product_ratings)
    else:
        render_transactions_preview(transactions)

def render_user_ratings_preview(user_ratings):
    """Render user ratings preview"""
    cols_to_show = ['userId', 'productId', 'rating']
    if 'formatted_date' in user_ratings.columns:
        cols_to_show.append('formatted_date')
    st.dataframe(user_ratings[cols_to_show].head(1000).sort_values('rating', ascending=False))

def render_product_info_preview(product_ratings):
    """Render product info preview"""
    if not product_ratings.empty:
        unique_products = product_ratings.drop_duplicates(subset=['productId'], keep='first')
        st.dataframe(unique_products.head(1000).reset_index(drop=True))
    else:
        st.warning("No product data available")

def render_transactions_preview(transactions):
    """Render transactions preview"""
    if not transactions.empty:
        transactions_display = transactions.copy()
        
        if 'time' in transactions_display.columns:
            transactions_display['date'] = pd.to_datetime(transactions_display['time'], unit='s').dt.date
            transactions_display = transactions_display.drop(columns=['time'])
        
        date_col = None
        if 'date' in transactions_display.columns:
            date_col = 'date'
        elif 'transactionDate' in transactions_display.columns:
            date_col = 'transactionDate'
        
        display_cols = [col for col in transactions_display.columns if col != 'formatted_date']
        
        if date_col:
            transactions_display = transactions_display.sort_values(date_col, ascending=False)
        
        st.dataframe(transactions_display[display_cols].head(1000))
    else:
        st.warning("No transaction data available")
