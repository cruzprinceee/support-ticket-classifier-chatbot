from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)  # Set num_labels to match your categories

# Load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data = data[['Ticket Description', 'Ticket Type']]
    data['Ticket Description'] = data['Ticket Description'].apply(lambda x: x.lower().strip())
    label_encoder = LabelEncoder()
    data['Ticket Type'] = label_encoder.fit_transform(data['Ticket Type'])
    return data

def tokenize_function(examples):
    return tokenizer(examples['Ticket Description'], padding="max_length", truncation=True, max_length=128)

def fine_tune_model(data):
    dataset = Dataset.from_pandas(data)
    
    # Tokenize with labels
    def preprocess_data(examples):
        tokens = tokenizer(examples['Ticket Description'], padding="max_length", truncation=True, max_length=128)
        tokens['labels'] = examples['Ticket Type']  # Include labels for classification
        return tokens

    # Apply the tokenization function
    tokenized_datasets = dataset.map(preprocess_data, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="./logs",
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

# Load and preprocess data
data = load_and_preprocess_data('data/customer_support_tickets.csv')

# Fine-tune the model
fine_tune_model(data)
