from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Define categories
categories = {
    0: "Billing Inquiry",
    1: "Technical Issue",
    2: "Account Issue",
}

def classify_ticket_with_confidence(query, threshold=0.6):
    # Tokenize input
    inputs = tokenizer.encode_plus(query, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    
    # Get model predictions
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_class = probabilities.max(dim=1)
    
    # Check confidence threshold
    if confidence.item() < threshold:
        return "uncertain", confidence.item()
    
    # Map predicted class to category
    category = categories.get(predicted_class.item(), "Unknown")
    return category, confidence.item()
