import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

nltk.download('punkt')

def preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    tokens = word_tokenize(text)
    return tokens

def build_ngrams(tokens, n=4):
    return list(ngrams(tokens, n))
