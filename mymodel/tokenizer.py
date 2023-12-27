import re
import torch
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def custom_tokenizer(text, max_length=512):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    word_counter = Counter(tokens)
    tokens = [word for word, count in word_counter.items() if word.isalpha() and count > 1]
    tokens = tokens[:max_length] + ['[PAD]'] * (max_length - len(tokens))
    vocab = {token: i + 1 for i, token in enumerate(tokens)}
    numerical_tokens = [vocab[token] for token in tokens]
    return numerical_tokens
