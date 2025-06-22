# src/data_processing/preprocess_phase1.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import os

# --- Researcher's Justification ---
# We download 'punkt' for tokenization and 'stopwords' for cleaning.
# This is a one-time setup. In a real-world on-device scenario,
# a pre-compiled stopwords list would be bundled with the application.
# ------------------------------------
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase.
    2. Removing punctuation and numbers.
    3. Tokenizing the text.
    4. Removing stopwords.
    5. Applying stemming.
    
    Returns: A cleaned, space-separated string of words.
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # 3. Tokenize
    tokens = nltk.word_tokenize(text)
    
    # 4. Remove stopwords and 5. Apply stemming
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    cleaned_tokens = [
        stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 1
    ]
    
    return " ".join(cleaned_tokens)

def main():
    """
    Main function to load, process, and save the dataset.
    """
    print("--- Starting Data Preprocessing ---")
    
    # Define file paths
    # --- Researcher's Justification ---
    # Using relative paths from a known root makes the project portable.
    # The os.path.join handles differences in OS path separators.
    # ------------------------------------
    raw_data_path = os.path.join('data', 'raw', 'SMSSpamCollection.csv')
    processed_dir = os.path.join('data', 'processed')
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(raw_data_path, sep='\t', header=None, names=['label', 'message'])
    print(f"Loaded {len(df)} messages from {raw_data_path}")

    # Clean text messages
    print("Cleaning and processing text messages...")
    df['cleaned_message'] = df['message'].apply(clean_text)
    
    # Encode labels
    # --- Researcher's Justification ---
    # ML models require numerical inputs. We map 'ham' to 0 (non-spam)
    # and 'spam' to 1. This is a standard and interpretable mapping.
    # ------------------------------------
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    print("Encoded labels: 'ham' -> 0, 'spam' -> 1")

    # Split the data
    # --- Researcher's Justification ---
    # stratify=df['label'] is CRITICAL here due to the class imbalance we found in the EDA.
    # It ensures that the train and test sets have the same proportion of spam/ham messages
    # as the original dataset, leading to a more reliable evaluation.
    # ------------------------------------
    X = df[['cleaned_message']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # Save the processed data
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Processed data saved to {train_path} and {test_path}")
    print("--- Data Preprocessing Complete ---")

if __name__ == '__main__':
    main()