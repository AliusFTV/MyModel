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

    def forward(self, x):
        return self.attention(x, x, x)[0]

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
        attn_output = self.self_attn(x)
        x = x + attn_output
        x = self.norm1(x)

        ff_output = self.feedforward(x)
        x = x + ff_output
        x = self.norm2(x)

        return x
class DecoderLayer(nn.Module):            #СКРЫТЫЙ СЛОЙ ДЕКОДЕРА(МОЗГ ЧАСТЬ 2)
    def __init__(self, d_model, nhead, dim_feedforward):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.feedforward = FeedForwardLayer(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        # САМО-ВНИМАНИЕ
        attn_output = self.self_attn(x)
        x = x + attn_output
        x = self.norm1(x)

        # ПЕРЕКРЁСТНОЕ ВНИМАНИЕ
        cross_attn_output = self.cross_attn(x, memory, memory)
        x = x + cross_attn_output
        x = self.norm2(x)

        # ПРЯМОЙ ПРОХОД
        ff_output = self.feedforward(x)
        x = x + ff_output
        x = self.norm3(x)

        return x
class Transformer(nn.Module):         #АРХИТЕКТУРА И ЗАПУСК
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])

    def forward(self, src, tgt):
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory)

        output = tgt
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, memory)

        return output

# ФУНКЦИЯ ОБУЧЕНИЯ
def train_model(model, train_dataloader, criterion, optimizer, num_epochs):
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
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

# ГИПЕРПАРАМЕТРЫ
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048
num_epochs = 10
learning_rate = 1e-4
batch_size = 8

# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ, ОПТИМИЗАТОР И КРИТЕРИИ
model = Transformer(d_model, nhead, num_layers, dim_feedforward)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ДАННЫЕ
dataset = load_dataset("glue", "mrpc")
train_data = dataset["train"]
test_data = dataset["test"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ПРЕОБРАЗОВАНИЕ В DATALOADER
def collate_fn(batch):
    input_texts = [example["sentence1"] for example in batch]
    target_texts = [example["sentence2"] for example in batch]

    input_data = torch.tensor(tokenizer(input_texts, return_tensors="pt", padding=True)["input_ids"])
    target_data = torch.tensor(tokenizer(target_texts, return_tensors="pt", padding=True)["input_ids"])

    return input_data, target_data
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ОБУЧЕНИЕ
train_model(model, train_dataloader, criterion, optimizer, num_epochs)
