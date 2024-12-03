from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
from transformers import Trainer, TrainingArguments
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
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(data)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

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
data = load_and_preprocess_data('path_to_your_file.csv')

# Fine-tune the model
fine_tune_model(data)
