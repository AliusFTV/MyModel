import re
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def custom_tokenizer(text, max_length=512):
    # Удаляем лишние символы и цифры
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Токенизация
    tokens = word_tokenize(text)

    # Приводим к нижнему регистру
    tokens = [token.lower() for token in tokens]

    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = tokens[:max_length] + ['[PAD]'] * (max_length - len(tokens))
    vocab = {token: i + 1 for i, token in enumerate(set(tokens))}
    numerical_tokens = [vocab[token] for token in tokens]
    numerical_tokens = numerical_tokens[:max_length] + [0] * (max_length - len(numerical_tokens))
    return numerical_tokens
dataset = load_dataset("glue", "mnli")
train_data = dataset["train"]
# Пример использования
for example in train_data:
    premise_tokens = custom_tokenizer(example["premise"])
    hypothesis_tokens = custom_tokenizer(example["hypothesis"])
    print("Premise tokens:", premise_tokens)
    print("Hypothesis tokens:", hypothesis_tokens)