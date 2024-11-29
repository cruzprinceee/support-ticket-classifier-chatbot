# Define predefined responses
responses = {
    "Billing Issue": "It seems like you’re facing a billing issue. Please contact our billing department.",
    "Technical Issue": "This appears to be a technical issue. Restarting your device might help.",
    "Account Issue": "You seem to have an account-related issue. Try resetting your password here: [Reset Password Link]",
    "Unknown": "I couldn’t classify your query. Let me connect you with a human representative.",
}

def chatbot_response(category):
    return responses.get(category, "We’re here to assist you. Please provide more details!")
