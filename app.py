import streamlit as st
from models.classifier import classify_ticket
from utils.helpers import chatbot_response

# Set up the page configuration
st.set_page_config(
    page_title="Support Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom styles
st.markdown("""
    <style>
        .stTextInput > label {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }
        .stButton button {
            background-color: #3498db;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton button:hover {
            background-color: #1abc9c;
        }
        .response-box {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .category-box {
            background-color: #dfe6e9;
            padding: 10px;
            border-radius: 10px;
            font-size: 18px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Main App Title
st.title("🤖 AI Support Bot")
st.subheader("Get instant assistance for your queries!")

# User Query Input
st.write("### Enter your support ticket query below:")
user_query = st.text_area("Describe your issue here", placeholder="e.g., My screen keeps flickering when I open videos.")

# Process the query and display the results
if user_query:
    with st.spinner("Processing your query..."):
        category = classify_ticket(user_query)
        response = chatbot_response(category, user_query)

    # Display results with enhanced visuals
    st.write("### Results:")
    st.markdown(f"<div class='category-box'><strong>Category:</strong> {category}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='response-box'><strong>Response:</strong> {response}</div>", unsafe_allow_html=True)

    # Suggest further steps
    st.info("🔍 If this doesn't resolve your issue, please contact our support team.")

