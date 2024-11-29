import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Ensure the dataset contains 'query' and 'category' columns
    if 'query' not in data.columns or 'category' not in data.columns:
        raise ValueError("The dataset must contain 'query' and 'category' columns.")

    # Preprocess queries (convert to lowercase, strip whitespaces)
    data['query'] = data['query'].str.lower().str.strip()

    # Map categories to integers
    label_mapping = {label: idx for idx, label in enumerate(data['category'].unique())}
    data['label'] = data['category'].map(label_mapping)

    # Split data into train, validation, and test sets
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    return train_data, val_data, test_data, label_mapping