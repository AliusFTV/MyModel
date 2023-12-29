import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer
import math

# ГИПЕРПАРАМЕТРЫ
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', return_tensors="pt")
nhead = 8
d_model = 1600
num_layers = 6
dim_feedforward = 2048
num_classes = 3
num_epochs = 3
learning_rate = 2e-5
batch_size = 32
dropout = 0.1


class MultiHeadAttention(nn.Module):  # ВНИМАНИЕ НЕЙРОНОВ
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class FeedForwardLayer(nn.Module):  # СЛОЙ ПРЯМОГО ПРОХОДА(FORWARD PASS)
    def __init__(self, d_model, dim_feedforward):
        super(FeedForwardLayer, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        return self.feedforward(x)

class EncoderLayer(nn.Module):  # СКРЫТЫЙ СЛОЙ КОДИРОВКИ(МОЗГ ЧАСТЬ 1)
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feedforward = FeedForwardLayer(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardLayer(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x



class Transformer(nn.Module):  # АРХИТЕКТУРА И ЗАПУСК
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.mask = None


    def forward(self, src, mask):
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory, mask)
        output = self.classifier(memory[:, 0, :])
        return output

# ФУНКЦИЯ ОБУЧЕНИЯ
def train_model(model, train_dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        # ПРОГРЕСС БАР
        for input_batch, target_batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', leave=False):
            optimizer.zero_grad()
            output_batch = model(input_batch, model.mask)
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

# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ, ОПТИМИЗАТОР И КРИТЕРИИ
model = Transformer(d_model, nhead, num_layers, dim_feedforward, num_classes, dropout)
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
    input_data = tokenizer(input_texts, return_tensors="pt", padding=True)["input_ids"].clone().detach()
    mask = (input_data != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
    input_data = input_data.float()
    target_data = torch.tensor(target_texts)
    return input_data, target_data

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ОБУЧЕНИЕ
train_model(model, train_dataloader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), 'adv_model.pth')
