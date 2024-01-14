import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm.notebook
from datasets import load_dataset
from transformers import BertTokenizer
import math
import keyboard
import threading
import time
import queue

# ГИПЕРПАРАМЕТРЫ
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', return_tensors="pt")
nhead = 2
d_model = 512
num_layers = 2
dim_feedforward = 512
num_classes = 3
num_epochs = 3
learning_rate = 2e-5
batch_size = 8
dropout = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_seq_length = 512
src_vocab_size = tokenizer.vocab_size
tgt_vocab_size = src_vocab_size


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

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
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        return self.feedforward(x)


class PositionalEncoding(nn.Module):         #ПОЗИЦИИ СЛОВ
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        if x.dim() == 3:
            return x + self.pe[:, :x.size(1)]
        elif x.dim() == 2:
            return x.unsqueeze(1) + self.pe[:, :x.size(1)]
        else:
            raise ValueError("Input tensor must be 2D or 3D")
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


class DecoderLayer(nn.Module):    # СКРЫТЫЙ СЛОЙ ДЕКОДИРОВКИ(МОЗГ ЧАСТЬ 2)
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


class Transformer(nn.Module):           #АРХИТЕКТУРА
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

        self.classifier = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = max_seq_length
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(device)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src))).to(device)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt))).to(device)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        tgt_mask = tgt_mask.unsqueeze(1)
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.classifier(dec_output[:, 0, :])
        return output
#ТРЭД ДЛЯ ЗАХВАТА КНОПКИ
exit_signal_queue = queue.Queue()
def keyboard_listener():
    while True:
        time.sleep(0.1)
        if keyboard.is_pressed("p"):
            exit_signal_queue.put(True)

keyboard_thread = threading.Thread(target=keyboard_listener)
keyboard_thread.daemon = True
keyboard_thread.start()

def save_checkpoint(epoch, batch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'transformer_model_checkpoint.pth')

def load_checkpoint():
    try:
        checkpoint = torch.load('transformer_model_checkpoint.pth')
        return checkpoint
    except FileNotFoundError:
        return None
# ФУНКЦИЯ ОБУЧЕНИЯ
def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs):
    epoch = 0
    batch = 0
    model.to(device)
    model.train()

    # ЗАГРУЗКА ПРОМЕЖУТОЧНЫХ РЕЗУЛЬТАТОВ (ЕСЛИ ЕСТЬ)
    checkpoint = load_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        batch = checkpoint['batch'] + 1
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Найден файл промежуточных результатов. Начинаем обучение с эпохи {epoch + 1}, батча {batch}.")
    else:
        print("Файл промежуточных результатов не найден. Начинаем обучение с самого начала.")

    for epoch in range(epoch, num_epochs):
        total_loss = 0.0
        progress_bar = tqdm.notebook.tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', leave=False)
        # ПРОГРЕСС БАР
        for input_batch, target_batch in progress_bar:
            optimizer.zero_grad()
            output_batch = model(input_batch, target_batch)
            loss = criterion(output_batch, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch += 1

            # СОХРАНЕНИЕ ПРОМЕЖУТОЧНЫХ РЕЗУЛЬТАТОВ
            try:
                if exit_signal_queue.get_nowait():
                    save_checkpoint(epoch, batch, model, optimizer, total_loss / len(train_dataloader))
                    print('Модель сохранена по запросу пользователя.')
                    sys.exit()
            except queue.Empty:
                pass
        avg_loss = total_loss / len(train_dataloader)

        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for input_batch, target_batch in tqdm.notebook.tqdm(test_dataloader, desc='Testing', unit='batch', leave=False):
                output_batch = model(input_batch)
                _, predicted = torch.max(output_batch, 1)
                total_correct += (predicted == target_batch).sum().item()
                total_samples += target_batch.size(0)

        accuracy = total_correct / total_samples
        print(f'Эпохи {epoch + 1}/{num_epochs}, Потери: {avg_loss:.4f}, Точность: {accuracy:.4f}')


# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ, ОПТИМИЗАТОР И КРИТЕРИИ
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout)
model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.to(device)
criterion = nn.CrossEntropyLoss()

# ДАННЫЕ
dataset = load_dataset("glue", "mnli")
train_data = dataset["train"]
test_data = dataset["validation_matched"]


# ПРЕОБРАЗОВАНИЕ В DATALOADER
def collate_fn(batch):
    input_texts = [example["premise"] + " [SEP] " + example["hypothesis"] for example in batch]
    target_texts = [example["label"] for example in batch]
    tokenized_data = tokenizer(input_texts, return_tensors="pt", padding='longest')
    input_data = tokenized_data["input_ids"].clone().detach().to(device)
    target_data = torch.tensor(target_texts, dtype=torch.long).to(device)
    return input_data, target_data


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ОБУЧЕНИЕ
train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), 'adv_model.pth')
