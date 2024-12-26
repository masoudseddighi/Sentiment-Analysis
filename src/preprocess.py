import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def load_data(filepath):
    """
    Load the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset with target and text columns.
    """
    data = pd.read_csv(filepath, encoding='latin1', names=["target", "ids", "date", "flag", "user", "text"])
    return data[['target', 'text']]

def clean_text(text):
    """
    Clean the text by removing URLs, special characters, and extra spaces.

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned text string.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def preprocess_data(data):
    """
    Apply text cleaning and stopword removal to the dataset.

    Args:
        data (pd.DataFrame): DataFrame with raw text data.

    Returns:
        pd.DataFrame: DataFrame with cleaned text.
    """
    stop_words = set(stopwords.words('english'))
    data['text'] = data['text'].apply(clean_text)
    data['text'] = data['text'].apply(
        lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words])
    )
    return data

if __name__ == "__main__":
    filepath = "../dataset/sample_sentiment140.csv"  # Path to the raw dataset
    output_filepath = "../dataset/cleaned_sentiment140.csv"  # Path to save the cleaned dataset

    # Load dataset
    print("Loading dataset...")
    data = load_data(filepath)

    # Preprocess dataset
    print("Preprocessing dataset...")
    data = preprocess_data(data)

    # Save the cleaned data
    data.to_csv(output_filepath, index=False)
    print(f"Preprocessing complete. Cleaned data saved to: {output_filepath}")
