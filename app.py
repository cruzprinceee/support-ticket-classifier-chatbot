import streamlit as st
from models.classifier import classify_ticket

# Define responses for different categories
responses = {
    0: "It seems like you’re facing a billing inquiry. Please contact our billing department.",
    1: "This appears to be a technical issue. Restarting your device might help.",
    # Add other responses for other categories
}

# Streamlit interface
st.title("Support Bot")

# User input
user_query = st.text_input("Enter your support ticket query:")

if user_query:
    # Classify the query
    category = classify_ticket(user_query)

    # Provide response based on classification
    response = responses.get(category, "We’re here to assist you. Please provide more details!")
    
    st.write(f"**Category:** {category}")
    st.write(f"**Response:** {response}")
