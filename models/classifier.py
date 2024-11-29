from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
data = pd.read_csv("cleaned_support_tickets.csv")

# Encode the labels
label_dict = {label: idx for idx, label in enumerate(data['Category'].unique())}
data['Category Label'] = data['Category'].map(label_dict)

# Train-test split
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Dataset class for tokenization
class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        inputs = self.tokenizer.encode_plus(
            row['Ticket Message'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(row['Category Label'], dtype=torch.long)
        }

# Prepare datasets
train_dataset = TicketDataset(train_data, tokenizer)
val_dataset = TicketDataset(val_data, tokenizer)

# Load pre-trained model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_dict))

# Training arguments
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

# Function to classify ticket
def classify_ticket(query):
    inputs = tokenizer.encode_plus(
        query, return_tensors="pt", truncation=True, max_length=128, padding="max_length"
    )
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=1).item()
    return predicted_class

