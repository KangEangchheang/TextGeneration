from collections import defaultdict
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams  # Correctly import ngrams from nltk
import random

nltk.download('punkt')

def preprocess_data(file_path, vocab_size=5000):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Read and lower case the text

    # Remove newline characters
    text = text.replace('\n', ' ')  # Replace newlines with a space

    tokens = word_tokenize(text)  # Tokenize the text

    # Filter out symbols and keep only alphanumeric tokens
    tokens = [token for token in tokens if token.isalnum()]

    return tokens

def process_text_files_in_folder(folder_path):
    all_tokens = []

    # Iterate over all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Only process text files
        if filename.endswith('.txt') and os.path.isfile(file_path):
            print(f"Processing file: {filename}")
            tokens = preprocess_data(file_path)
            all_tokens.extend(tokens)  # Add tokens from this file to the combined list

    return all_tokens

def split_data(tokens):
    random.shuffle(tokens)
    train_size = int(len(tokens) * 0.7)
    val_size = int(len(tokens) * 0.1)
    
    train_data = tokens[:train_size]
    val_data = tokens[train_size:train_size + val_size]
    test_data = tokens[train_size + val_size:]
    
    return train_data, val_data, test_data

# Function to generate 4-grams manually
def generate_4grams(tokens):
    # Generate 4-grams using nltk's ngrams function
    ngrams_list = list(ngrams(tokens, 4))
    
    # Create a dictionary to store the n-grams
    ngram_model = defaultdict(list)
    
    # Store the n-grams in the model
    for ngram in ngrams_list:
        # The context is the first 3 words, the next word is the 4th word
        context = ngram[:3]
        next_word = ngram[3]
        ngram_model[context].append(next_word)
    
    return ngram_model
