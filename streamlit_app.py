import streamlit as st
from openai import OpenAI
import time

# Title
st.title("🌿 AyurVeda AI ChatBot 🌿")

# Introduction text
st.write("Welcome to AyurVeda AI, bringing you ancient Ayurvedic solutions for health and well-being!")

# ChatBot Image
st.image("AyurVedaBot.webp")

# Nvidia NIM
NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.

# Initialize session state to store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history in sequential order
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👨🏻‍💻"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant", avatar="🌿"):
            st.markdown(message["content"])

# User input field at the bottom, prompts latest message above it
if user_input := st.chat_input("Tell me about your ailment?"):

    # Append user input to messages and display it immediately
    with st.chat_message("user", avatar="👨🏻‍💻"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate a response using the NVIDIA NIM API.
    stream = client.chat.completions.create(
    model="meta/llama-3.1-405b-instruct",
    messages=[{"role":"user","content":user_input}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=True
    ) 

    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        response = st.write_stream(stream)    

    # Store the full response in session state to preserve chat history
    st.session_state.messages.append({"role": "assistant", "content": response})