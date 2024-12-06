from collections import Counter
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams  # Correctly import ngrams from nltk
import random

nltk.download('punkt')


def sanitize_text(text):
     # Remove timestamps in the format HH:MM:SS.MMM
    time_regex = r'\b\d{2}:\d{2}:\d{2}\.\d{3}\b'
    new_text = re.sub(time_regex, '', text)
    
    # Remove text within parentheses () and square brackets []
    new_text = re.sub(r'\([^)]*\)', '', new_text)
    new_text = re.sub(r'\[[^\]]*\]', '', new_text)
    
    # Remove newlines and tabs
    new_text = new_text.replace('\n', '').replace('\t', ' ')
    
    # Remove all characters except letters, numbers, periods, and spaces
    symbolreg = r'[^a-zA-Z0-9.\s\']'
    new_text = re.sub(symbolreg, '', new_text)
    
    # Remove extra spaces
    new_text = re.sub(r'\s+', ' ', new_text).strip()
    
    return new_text

def preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Read and lower case the text
        
    return text

def process_text_files_in_folder(folder_path):
    all_sentence = []

    # Iterate over all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Only process text files
        if filename.endswith('.txt') and os.path.isfile(file_path):
            print(f"Processing file: {filename}")
            text = preprocess_data(file_path)
            
            text = sanitize_text(text)
            sentences = text.split('.')  # Split the text into sentences
            
            random.shuffle(sentences)
            all_sentence.extend(sentences)
            
    if len(all_sentence) == 0:
        raise ValueError("No sentences found in the text files.")
            
    train_size = int(len(all_sentence) * 0.7)
    val_size = int(len(all_sentence) * 0.1)
    
    train_text = all_sentence[:train_size]
    val_text = all_sentence[train_size:train_size + val_size]
    test_text = all_sentence[train_size + val_size:]
    
    train_token = tokenize(train_text)
    val_token = tokenize(val_text)
    test_token = tokenize(test_text)

    return train_token, val_token, test_token


def tokenize(sentences):
    # Flatten the list of sentences into a single string
    tokens = word_tokenize(" ".join(sentences))  # Tokenize the sentences
    vocab_size = 10000

    # Count the most common tokens
    token_counts = Counter(tokens)
    vocab = {word for word, _ in token_counts.most_common(vocab_size)}

    # Replace unknown tokens with '<UNK>'
    processed_token = replace_unk(sentences, vocab)
    return processed_token

def replace_unk(data, vocab):
    return [
        [
            word if word in vocab else '<UNK>' for word in word_tokenize(sentence)
        ]
        for sentence in data
    ]
