from datasets import load_dataset
from transformers import BertTokenizer

# ГИПЕРПАРАМЕТРЫ
dataset = load_dataset("glue", "mnli")
train_data = dataset["train"]

# Примерный формат данных:
# {'input_ids': [0, 123, 45, ..., 0], 'label': 1}

# Попробуем вывести структуру данных для одного примера
example = train_data[0]
print(example)

# После понимания структуры, мы можем попытаться извлечь тексты и метки
for example in train_data[:5]:  # выведите первые 5 примеров
    input_ids = example["input_ids"]
    label = example["label"]

    # Создаем список текстов для BERT токенизации
    input_texts = tokenizer.decode(input_ids)

    # Токенизируем текст
    input_data = tokenizer(input_texts, return_tensors="pt", padding=True)["input_ids"]

    # Выводим размерность встроенных представлений
    print(f"Input Texts: {input_texts}")
    print(f"Embedded Input Dimension: {input_data.shape[-1]}")

    # Обновляем d_model в соответствии с фактической размерностью
    d_model = input_data.shape[-1]

# Печатаем обновленное значение d_model
print(f"Updated d_model: {d_model}")



