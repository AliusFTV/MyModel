import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def custom_tokenizer(text):
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

    return tokens
dataset = load_dataset("glue", "mnli")
train_data = dataset["train"]
# Пример использования
for example in train_data:
    premise_tokens = custom_tokenizer(example["premise"])
    hypothesis_tokens = custom_tokenizer(example["hypothesis"])
    print("Premise tokens:", premise_tokens)
    print("Hypothesis tokens:", hypothesis_tokens)