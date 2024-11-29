from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

def classify_ticket(query):
    inputs = tokenizer.encode_plus(
        query, return_tensors="pt", truncation=True, max_length=128, padding="max_length"
    )
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=1).item()
    return predicted_class