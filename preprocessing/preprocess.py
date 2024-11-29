import pandas as pd

def preprocess_text(text):
    return text.lower().strip()

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['Ticket Message'] = data['Ticket Message'].apply(preprocess_text)
    return data
