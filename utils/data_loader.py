import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Ensure the dataset contains 'query' and 'category' columns
    if 'query' not in data.columns or 'category' not in data.columns:
        raise ValueError("The dataset must contain 'query' and 'category' columns.")

    # Preprocess text (convert to lowercase, strip whitespaces)
    data['query'] = data['query'].str.lower().str.strip()

    # Map categories to integers
    label_mapping = {label: idx for idx, label in enumerate(data['category'].unique())}
    data['label'] = data['category'].map(label_mapping)

    # Split data into train, validation, and test sets
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    return train_data, val_data, test_data, label_mapping

# Custom Dataset Class for PyTorch
class TicketDataset(Dataset):
    def __init__(self, queries, labels, tokenizer, max_len):
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, item):
        query = self.queries[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'query': query,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
