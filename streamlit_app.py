import streamlit as st
from llama_index_builder import load_index_from_disk
from chatbot import chat_with_user
import asyncio
import logging

# Configure page settings
st.set_page_config(
    page_title="PopCorn.Ai",
    page_icon="🎬",
    layout="centered"
)

nodes = load_index_from_disk("./movie_index")


# Custom CSS for modern gradient background and styling
st.markdown("""
<style>
    /* Modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Style chat messages */
    .user-message {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-bottom-right-radius: 5px;
    }
    
    .bot-message {
        background: rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-bottom-left-radius: 5px;
    }
    
    /* Style the chat input */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    
    /* Style headings */
    h1, h2, h3 {
        color: white !important;
    }
    
    /* Style examples */
    .example {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .example:hover {
        background: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🎬 PopCornAI")

# Sidebar with examples
with st.sidebar:
    st.header("Example Queries")
    examples = [
        "What movies are playing in Koramangala?",
        "Show me theatres near Forum Mall",
        "Showtimes for Singham Again",
        "Review of Inception",
        "Show me comedy movies"
    ]
    
    for example in examples:
        if st.button(example, key=example):
            st.session_state.messages.append({"role": "user", "content": example})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""<div class="user-message">👤 {message["content"]}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="bot-message">🤖 {message["content"]}</div>""", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about movies, theatres, or showtimes..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    try:
        with st.spinner("Thinking..."):
            response = asyncio.run(chat_with_user(prompt))
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Rerun to update the display
            st.rerun()
    except Exception as e:
        logging.error(f"Error: {e}")
        st.error("Sorry, I encountered an error. Please try again.")

# Initial welcome message
if not st.session_state.messages:
    st.markdown("""
    <div class="bot-message">
    🤖 Welcome! I can help you with:
    
    • Finding movies playing near you
    • Locating theatres in your area
    • Getting movie showtimes
    • Movie reviews and ratings
    • Movie recommendations by genre
    
    Try asking me something or click an example from the sidebar!
    </div>
    """, unsafe_allow_html=True)
