# Predefined responses
responses = {
    0: "It seems like you’re facing a billing issue. Please contact our billing department.",
    1: "This appears to be a technical issue. Restarting your device might help.",
    2: "The solution to your problem is Account Assistant. To reset your password, follow the instructions sent to your email. Let us know if you need further help.",
    3: "Please provide more details about your issue."
}

def chatbot_response(category, user_query):
    return responses.get(category, "We’re here to assist you. Please provide more details!")
