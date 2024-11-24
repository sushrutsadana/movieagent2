import streamlit as st
from llama_index_builder import load_index_from_disk
from chatbot import chat_with_user

# Load nodes for querying
nodes = load_index_from_disk("movie_index.pkl")

st.title("Movie Recommendation Chatbot")
question = st.text_input("Ask about movies, showtimes, or genres:")
if st.button("Ask"):
    response = chat_with_user(question, nodes)
    st.write(response)
