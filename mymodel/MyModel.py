import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer

class MultiHeadAttention(nn.Module):          #ВНИМАНИЕ НЕЙРОНОВ
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, query, key, value):
        return self.attention(query, key, value)[0]

class FeedForwardLayer(nn.Module):      #СЛОЙ ПРЯМОГО ПРОХОДА(FORWARD PASS)
    def __init__(self, d_model, dim_feedforward):
        super(FeedForwardLayer, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        return self.feedforward(x)

class EncoderLayer(nn.Module):          #СКРЫТЫЙ СЛОЙ КОДИРОВКИ(МОЗГ ЧАСТЬ 1)
    def __init__(self, d_model, nhead, dim_feedforward):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feedforward = FeedForwardLayer(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)

        ff_output = self.feedforward(x)
        x = x + ff_output
        x = self.norm2(x)

        return x
class Transformer(nn.Module):         #АРХИТЕКТУРА И ЗАПУСК
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_classes):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src):
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory)
        output = self.classifier(memory[:, :])
        return output

# ФУНКЦИЯ ОБУЧЕНИЯ
def train_model(model, train_dataloader, criterion, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        # ПРОГРЕСС БАР
        for input_batch, target_batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', leave=False):
            optimizer.zero_grad()
            output_batch = model(input_batch)
            loss = criterion(output_batch, target_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for input_batch, target_batch in tqdm(test_dataloader, desc='Testing', unit='batch', leave=False):
                output_batch = model(input_batch)
                _, predicted = torch.max(output_batch, 1)
                total_correct += (predicted == target_batch).sum().item()
                total_samples += target_batch.size(0)

        accuracy = total_correct / total_samples
        print(f'Эпохи {epoch + 1}/{num_epochs}, Потери: {avg_loss:.4f}, Точность: {accuracy:.4f}')

# ГИПЕРПАРАМЕТРЫ
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nhead = 8
d_model = 512
num_layers = 6
dim_feedforward = 2048
num_classes = 3
num_epochs = 3
learning_rate = 2e-5
batch_size = 32

# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ, ОПТИМИЗАТОР И КРИТЕРИИ
model = Transformer(d_model, nhead, num_layers, dim_feedforward, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ДАННЫЕ
dataset = load_dataset("glue", "mnli")
train_data = dataset["train"]
test_data = dataset["validation_matched"]


# ПРЕОБРАЗОВАНИЕ В DATALOADER
def collate_fn(batch):
    input_texts = [example["premise"] + " [SEP] " + example["hypothesis"] for example in batch]
    target_texts = [example["label"] for example in batch]
    input_data = tokenizer(input_texts, return_tensors="pt", padding='max_length', max_length=512)["input_ids"].clone().detach()
    input_data = input_data.float()
    print(f"Expected d_model: {d_model}, Actual d_model: {input_data.size(-1)}")
    target_data = torch.tensor(target_texts)
    return input_data, target_data


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ОБУЧЕНИЕ
train_model(model, train_dataloader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), 'model.pth')
