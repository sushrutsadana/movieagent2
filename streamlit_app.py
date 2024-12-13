import streamlit as st
import asyncio
from chatbot import chat_with_user, WELCOME_MESSAGE

# Configure Streamlit page
st.set_page_config(
    page_title="PopcornAI - Movie Booking Assistant",
    page_icon="🍿",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 850px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Message bubbles */
    .message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        line-height: 1.5;
    }
    
    .user-message {
        background-color: #F3F6FC;
        margin-left: auto;
        margin-right: 0;
        max-width: 80%;
    }
    
    .bot-message {
        background-color: #F8F9FA;
        border: 1px solid #E8EAED;
        margin-right: auto;
        margin-left: 0;
        max-width: 80%;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Input container */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #FFFFFF;
        border-top: 1px solid #E8EAED;
        padding: 20px;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 1px solid #E8EAED !important;
        border-radius: 24px !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
        box-shadow: none !important;
    }
    
    /* Example queries */
    .example-query {
        display: inline-block;
        margin: 5px;
        padding: 8px 16px;
        background-color: #F8F9FA;
        border: 1px solid #E8EAED;
        border-radius: 16px;
        color: #202124;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []