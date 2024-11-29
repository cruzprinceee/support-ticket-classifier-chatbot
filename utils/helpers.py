# Predefined responses
responses = {
    0: "It seems like you’re facing a billing issue. Please contact our billing department.",
    1: "This appears to be a technical issue. Restarting your device might help.",
}

def chatbot_response(category, user_query):
    return responses.get(category, "We’re here to assist you. Please provide more details!")

