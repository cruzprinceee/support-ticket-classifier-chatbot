import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_text(text):
    # Convert to lowercase and strip spaces
    return text.lower().strip()

def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)
    
    # Extract relevant columns: 'Ticket Description' and 'Ticket Type'
    data = data[['Ticket Description', 'Ticket Type']]

    # Preprocess text data
    data['Ticket Description'] = data['Ticket Description'].apply(preprocess_text)

    # Encode labels (Ticket Type)
    label_encoder = LabelEncoder()
    data['Ticket Type'] = label_encoder.fit_transform(data['Ticket Type'])

    return data
