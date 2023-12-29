import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer
from MyModel import Transformer
import MyModel

# Гиперпараметры
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nhead = 8
d_model = 512
num_layers = 6
dim_feedforward = 2048
num_classes = 3
num_epochs = 3
learning_rate = 2e-5
batch_size = 32

# Инициализация модели
model = Transformer(d_model, nhead, num_layers, dim_feedforward, num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Загрузка тестовых данных
dataset = load_dataset("glue", "mnli")
test_data = dataset["validation_matched"]
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=MyModel.collate_fn)

# Оценка модели
total_correct = 0
total_samples = 0

with torch.no_grad():
    for input_batch, target_batch in tqdm(test_dataloader, desc='Testing', unit='batch', leave=False):
        output_batch = model(input_batch)
        _, predicted = torch.max(output_batch, 1)
        total_correct += (predicted == target_batch).sum().item()
        total_samples += target_batch.size(0)

accuracy = total_correct / total_samples
print(f'Точность модели на тестовых данных: {accuracy:.4f}')