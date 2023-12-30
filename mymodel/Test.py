from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', return_tensors="pt")
input_texts = ["Hello, how are you?", "I'm doing well, thank you!"]
tokenized_data = tokenizer(input_texts, padding='max_length', max_length=1600)

# Выводим информацию о размерности
for key, value in tokenized_data.items():
    print(f"{key}: {value.size()}")