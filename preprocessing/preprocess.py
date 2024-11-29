import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters (non-alphanumeric)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Lemmatization (optional but recommended)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Function to load and preprocess the dataset
def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)

    # Drop unnecessary columns (if any)
    data = data.drop(columns=['Ticket ID'], errors='ignore')

    # Preprocess the 'Ticket Message' column
    data['Ticket Message'] = data['Ticket Message'].apply(preprocess_text)

    return data

# Example usage
if __name__ == "__main__":
    data = load_and_preprocess_data("support_tickets.csv")
    data.to_csv("cleaned_support_tickets.csv", index=False)
    print(data.head())
