import streamlit as st
import time

with st.container(border=True):
    # Title
    st.title("🌿 AyurVeda AI ChatBot")

    st.image("AyurVedaBot.webp")

    # Introduction text
    st.write(
        "Welcome to AyurVeda AI, bringing you ancient Ayurvedic solutions for health and well-being!"
    )

    # Nvidia NIM TODO

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.

    # Initialize session state to store chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages in order
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👨🏻‍💻"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🌿"):
                st.markdown(message["content"])

    # Chat input at the bottom of the chat display
    if user_input := st.chat_input("Tell me about your ailment?"):

        # Append user input to messages and display it
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Placeholder for assistant's response to simulate streaming
        assistant_placeholder = st.chat_message("assistant", avatar="🌿")
        full_response = ""
        st.session_state.messages.append({"role": "assistant", "content": full_response})  # Start with empty content

        # Simulate streaming response (or replace this with actual API call for streaming)
        response_chunks = [
            {"role": "assistant", "content": "Based on Ayurveda, "},
            {"role": "assistant", "content": "here's some guidance "},
            {"role": "assistant", "content": "for your well-being."}
        ]

        for chunk in response_chunks:
            full_response += chunk["content"]
            assistant_placeholder.markdown(full_response)  # Update message content progressively
            time.sleep(0.05)  # Adjust delay for realistic streaming

        # Update the last assistant message in session state with the completed response
        st.session_state.messages[-1]["content"] = full_response