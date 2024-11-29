import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Load and preprocess the dataset
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['query'] = data['query'].str.lower().str.strip()
    label_mapping = {label: idx for idx, label in enumerate(data['category'].unique())}
    data['label'] = data['category'].map(label_mapping)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data, label_mapping

# Fine-tune the model
@st.cache_resource
def fine_tune_model(train_data, val_data, label_mapping):
    # Convert dataframes to HuggingFace Datasets
    def convert_to_dataset(df):
        return Dataset.from_pandas(df[['query', 'label']])

    train_dataset = convert_to_dataset(train_data)
    val_dataset = convert_to_dataset(val_data)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_mapping)
    )

    # Tokenize datasets
    def tokenize(batch):
        return tokenizer(batch['query'], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    return model, tokenizer

# Classify a query
def classify_ticket(query, model, tokenizer, label_mapping, threshold=0.6):
    categories = {v: k for k, v in label_mapping.items()}
    inputs = tokenizer.encode_plus(query, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_class = probabilities.max(dim=1)

    if confidence.item() < threshold:
        return "uncertain", confidence.item()

    category = categories.get(predicted_class.item(), "Unknown")
    return category, confidence.item()

# Streamlit UI
st.title("Support Bot")

# Load dataset and fine-tune model
if "model" not in st.session_state:
    st.write("Loading dataset and fine-tuning the model...")
    train_data, val_data, test_data, label_mapping = load_data("data/customer_support_tickets.csv")
    model, tokenizer = fine_tune_model(train_data, val_data, label_mapping)
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.session_state.label_mapping = label_mapping
else:
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    label_mapping = st.session_state.label_mapping

# User query input
user_query = st.text_input("Enter your support ticket query:")
if user_query:
    category, confidence = classify_ticket(user_query, model, tokenizer, label_mapping, threshold=0.6)

    if category == "uncertain":
        st.write("I'm not sure about the exact issue. Let me route this to a human representative.")
    else:
        st.write(f"**Category:** {category}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.write(f"Response: This seems to be a {category.lower()} issue. Let me help you!")
