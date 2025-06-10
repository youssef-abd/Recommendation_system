import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
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
        .footer { font-size: 0.8rem; color: #666; text-align: center; }
    </style>
    """, unsafe_allow_html=True)
