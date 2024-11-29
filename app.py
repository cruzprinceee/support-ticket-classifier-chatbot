import streamlit as st
from models.classifier import classify_ticket_with_confidence
from utils.helpers import chatbot_response

st.title("Automated Support Ticket Chatbot")

user_query = st.text_input("Enter your support ticket query:")
if user_query:
    category, confidence = classify_ticket_with_confidence(user_query, threshold=0.6)
    
    if category == "uncertain":
        st.write("I'm not sure about the exact issue. Let me route this to a human representative.")
    else:
        response = chatbot_response(category)
        st.write(f"**Category:** {category}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.write(f"**Response:** {response}")
