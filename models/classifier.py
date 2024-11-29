from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load fine-tuned model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("/workspaces/support-ticket-classifier-chatbot/fine_tune_model.py")
model = DistilBertForSequenceClassification.from_pretrained("/workspaces/support-ticket-classifier-chatbot/fine_tune_model.py")

# Map labels back to categories
categories = {
    0: "Billing Issue",
    1: "Technical Issue",
    2: "Account Issue",
    # Add more categories as needed
}

def classify_ticket_with_confidence(query, threshold=0.6):
    # Tokenize input
    inputs = tokenizer.encode_plus(query, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_class = probabilities.max(dim=1)

    # Check confidence threshold
    if confidence.item() < threshold:
        return "uncertain", confidence.item()

    # Map predicted class to category
    category = categories.get(predicted_class.item(), "Unknown")
    return category, confidence.item()
