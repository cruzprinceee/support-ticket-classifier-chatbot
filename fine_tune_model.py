from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from utils.data_loader import load_and_preprocess_data, TicketDataset
import torch
from transformers import Trainer, TrainingArguments

# Load and preprocess data
train_data, val_data, test_data, label_mapping = load_and_preprocess_data("/workspaces/support-ticket-classifier-chatbot/dataset/customer_support_tickets.csv")

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Create PyTorch datasets
train_dataset = TicketDataset(train_data['query'].tolist(), train_data['label'].tolist(), tokenizer, max_len=128)
val_dataset = TicketDataset(val_data['query'].tolist(), val_data['label'].tolist(), tokenizer, max_len=128)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_mapping))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Set training arguments
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

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("/workspaces/support-ticket-classifier-chatbot/fine_tune_model.py")
tokenizer.save_pretrained("/workspaces/support-ticket-classifier-chatbot/fine_tune_model.py")
