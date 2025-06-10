# =============================================================================
# FILE: components/transactions.py
# =============================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render(transactions):
    """Render transaction analysis page"""
    st.subheader("💳 Transaction Analysis")
    
    if transactions.empty:
        st.warning("No transaction data available")
        return
    
    # Process transaction data
    transactions_processed = process_transaction_data(transactions)
    
    # Display transaction overview
    display_transaction_overview(transactions_processed)
    
    # Display transaction table
    display_transaction_table(transactions_processed)
    
    # Display transaction analytics
    display_transaction_analytics(transactions_processed)

def process_transaction_data(transactions):
    """Process transaction data for display"""
    transactions_processed = transactions.copy()
    
    # Convert timestamp if 'time' column exists
    if 'time' in transactions_processed.columns:
        transactions_processed['date'] = pd.to_datetime(
            transactions_processed['time'], unit='s'
        ).dt.date
        # Keep the original time column for detailed analysis
        transactions_processed['datetime'] = pd.to_datetime(
            transactions_processed['time'], unit='s'
        )
    
    return transactions_processed

def display_transaction_overview(transactions):
    """Display transaction overview metrics"""
    st.subheader("📊 Transaction Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", len(transactions))
    
    with col2:
        if 'amount' in transactions.columns:
            total_amount = transactions['amount'].sum()
            st.metric("Total Amount", f"${total_amount:,.2f}")
        else:
            st.metric("Unique Users", transactions['userId'].nunique() if 'userId' in transactions.columns else "N/A")
    
    with col3:
        if 'amount' in transactions.columns:
            avg_amount = transactions['amount'].mean()
            st.metric("Average Amount", f"${avg_amount:.2f}")
        else:
            st.metric("Unique Products", transactions['productId'].nunique() if 'productId' in transactions.columns else "N/A")
    
    with col4:
        if 'date' in transactions.columns:
            date_range = (transactions['date'].max() - transactions['date'].min()).days
            st.metric("Date Range (days)", date_range)
        else:
            st.metric("Data Points", len(transactions.columns))

def display_transaction_table(transactions):
    """Display transaction data in table format"""
    st.subheader("📋 Transaction Data")
    
    # Prepare display columns
    display_cols = prepare_display_columns(transactions)
    
    # Sort by date if available, otherwise by first column
    if 'date' in transactions.columns:
        transactions_sorted = transactions.sort_values('date', ascending=False)
    else:
        transactions_sorted = transactions
    
    # Display main transaction table
    st.dataframe(
        transactions_sorted[display_cols].head(1000),
        use_container_width=True
    )
    
    # Display formatted amounts if available
    if 'amount' in transactions.columns:
        display_formatted_amounts(transactions_sorted)

def prepare_display_columns(transactions):
    """Prepare columns for display, excluding internal processing columns"""
    exclude_cols = ['time', 'formatted_date', 'datetime']
    display_cols = [col for col in transactions.columns if col not in exclude_cols]
    
    # Ensure date comes first if available
    if 'date' in display_cols:
        display_cols.remove('date')
        display_cols.insert(0, 'date')
    
    return display_cols

def display_formatted_amounts(transactions):
    """Display transactions with formatted amounts"""
    st.subheader("💰 Transaction Amounts")
    
    # Create a sample for formatting display
    amount_display = transactions[['date', 'amount']].head(100) if 'date' in transactions.columns else transactions[['amount']].head(100)
    
    st.dataframe(
        amount_display.style.format({'amount': '${:.2f}'}),
        use_container_width=True
    )

def display_transaction_analytics(transactions):
    """Display transaction analytics and visualizations"""
    if 'date' in transactions.columns:
        display_time_based_analytics(transactions)
    
    if 'amount' in transactions.columns:
        display_amount_analytics(transactions)
    
    display_general_analytics(transactions)

def display_time_based_analytics(transactions):
    """Display time-based transaction analytics"""
    st.subheader("📈 Transaction Volume Over Time")
    
    # Time period selection
    time_period = st.selectbox(
        "Aggregation period", 
        ["Daily", "Weekly", "Monthly"],
        help="Choose how to group transactions by time"
    )
    
    # Convert date to datetime for resampling
    transactions_time = transactions.copy()
    transactions_time['datetime'] = pd.to_datetime(transactions_time['date'])
    transactions_time = transactions_time.set_index('datetime')
    
    # Resample based on selected period
    if time_period == "Daily":
        time_series = transactions_time.resample('D').size()
    elif time_period == "Weekly":
        time_series = transactions_time.resample('W').size()
    else:  # Monthly
        time_series = transactions_time.resample('M').size()
    
    # Display line chart
    st.line_chart(time_series)
    
    # Display time-based statistics
    display_time_statistics(transactions, time_series, time_period)

def display_time_statistics(transactions, time_series, time_period):
    """Display time-based statistics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        peak_period = time_series.idxmax()
        st.metric("Peak Period", peak_period.strftime('%Y-%m-%d'))
    
    with col2:
        peak_volume = time_series.max()
        st.metric(f"Peak {time_period} Volume", f"{peak_volume}")
    
    with col3:
        avg_volume = time_series.mean()
        st.metric(f"Average {time_period} Volume", f"{avg_volume:.1f}")

def display_amount_analytics(transactions):
    """Display amount-based analytics"""
    st.subheader("💰 Amount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount distribution histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle potential outliers by using percentiles
        amount_95th = transactions['amount'].quantile(0.95)
        filtered_amounts = transactions[transactions['amount'] <= amount_95th]['amount']
        
        ax.hist(filtered_amounts, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Transaction Amount ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Transaction Amount Distribution (95th Percentile)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Amount box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(filtered_amounts, vert=True)
        ax.set_ylabel('Transaction Amount ($)')
        ax.set_title('Transaction Amount Box Plot')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Amount statistics
    display_amount_statistics(transactions)

def display_amount_statistics(transactions):
    """Display amount statistics"""
    st.subheader("Amount Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Min Amount", f"${transactions['amount'].min():.2f}")
    
    with col2:
        st.metric("Max Amount", f"${transactions['amount'].max():.2f}")
    
    with col3:
        st.metric("Median Amount", f"${transactions['amount'].median():.2f}")
    
    with col4:
        st.metric("Std Deviation", f"${transactions['amount'].std():.2f}")

def display_general_analytics(transactions):
    """Display general transaction analytics"""
    st.subheader("📊 General Analytics")
    
    # User and product analysis if available
    if 'userId' in transactions.columns:
        display_user_transaction_analysis(transactions)
    
    if 'productId' in transactions.columns:
        display_product_transaction_analysis(transactions)

def display_user_transaction_analysis(transactions):
    """Display user transaction analysis"""
    st.subheader("User Transaction Patterns")
    
    user_stats = transactions.groupby('userId').agg({
        'amount': ['count', 'sum', 'mean'] if 'amount' in transactions.columns else 'count'
    }).round(2)
    
    # Flatten column names if multi-level
    if isinstance(user_stats.columns, pd.MultiIndex):
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_stats = user_stats.rename(columns={
            'amount_count': 'Transaction Count',
            'amount_sum': 'Total Amount',
            'amount_mean': 'Average Amount'
        })
    else:
        user_stats = user_stats.rename(columns={'count': 'Transaction Count'})
    
    # Sort by transaction count
    sort_col = 'Transaction Count'
    user_stats = user_stats.sort_values(sort_col, ascending=False)
    
    st.dataframe(user_stats.head(100), use_container_width=True)

def display_product_transaction_analysis(transactions):
    """Display product transaction analysis"""
    st.subheader("Product Transaction Patterns")
    
    product_stats = transactions.groupby('productId').agg({
        'amount': ['count', 'sum', 'mean'] if 'amount' in transactions.columns else 'count'
    }).round(2)
    
    # Flatten column names if multi-level
    if isinstance(product_stats.columns, pd.MultiIndex):
        product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns]
        product_stats = product_stats.rename(columns={
            'amount_count': 'Transaction Count',
            'amount_sum': 'Total Revenue',
            'amount_mean': 'Average Price'
        })
    else:
        product_stats = product_stats.rename(columns={'count': 'Transaction Count'})
    
    # Sort by transaction count
    sort_col = 'Transaction Count'
    product_stats = product_stats.sort_values(sort_col, ascending=False)
    
    st.dataframe(product_stats.head(100), use_container_width=True)