from transformers import pipeline

# Load a pre-trained language model
nlp = pipeline("text-generation", model="gpt2")

def chatbot_response(category, user_query):
    # Generate a response based on the user query
    response = nlp(f"Category: {category}. User query: {user_query}", max_length=50, num_return_sequences=1)
    return response[0]['generated_text']
