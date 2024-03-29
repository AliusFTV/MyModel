import threading
import math
import time
import queue
import torch
from torch import nn
from torch.utils.data import DataLoader
from adabound import AdaBound
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer
import keyboard

# HYPERPARAMS
TK = BertTokenizer.from_pretrained('bert-base-uncased', return_tensors="pt")
HEADS = 8
IDS = 768
LAYERS = 6
CLASSES = 3
B_SIZE = 32
DROPOUT = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 512
SRC_SIZE = TK.vocab_size
TGT_SIZE = SRC_SIZE
# DATA_SECTION
DATASET = load_dataset("glue", "mnli")
TRAIN_DATA = DATASET["train"]
VALIDATION_DATA = DATASET["validation_matched"]
TEST_DATA = DATASET["test_matched"]
EXIT_SIGNAL = queue.Queue()
# THE DATALOADER


def train_collate_fn(batch):
    i_texts = [example["premise"] + " [SEP] " + example["hypothesis"] for example in batch]
    t_texts = [example["label"] for example in batch]
    tk_data = TK(i_texts, return_tensors="pt", padding='longest')
    i_data = tk_data["input_ids"].clone().detach().to(device)
    t_data = torch.tensor(t_texts, dtype=torch.long).to(device)
    return i_data, t_data


def test_collate_fn(batch):
    i_texts = [example["premise"] + " [SEP] " + example["hypothesis"] for example in batch]
    tk_data = TK(i_texts, return_tensors="pt", padding='longest')
    i_data = tk_data["input_ids"].clone().detach().to(device)
    return i_data


TRAIN_DL = DataLoader(TRAIN_DATA, batch_size=B_SIZE, shuffle=True, collate_fn=train_collate_fn)
VALIDATION_DL = DataLoader(VALIDATION_DATA, batch_size=B_SIZE, shuffle=False, collate_fn=train_collate_fn)
TEST_DL = DataLoader(TEST_DATA, batch_size=B_SIZE, shuffle=True, collate_fn=test_collate_fn)


# KEYBOARD CATCH THREAD


def keyboard_listener():
    while True:
        time.sleep(0.1)
        if keyboard.is_pressed("/"):
            EXIT_SIGNAL.put(True)


K_THREAD = threading.Thread(target=keyboard_listener)
K_THREAD.daemon = True
K_THREAD.start()


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MultiHeadAttention(nn.Module):  # THE NEURON ATTENTION
    def __init__(self, ids, heads):
        super().__init__()
        assert ids % heads == 0, "d_model must be divisible by num_heads"

        self.ids = ids
        self.heads = heads
        self.d_k = ids // heads

        self.w_q = nn.Linear(ids, ids)
        self.w_k = nn.Linear(ids, ids)
        self.w_v = nn.Linear(ids, ids)
        self.w_o = nn.Linear(ids, ids)

    def scaled_dot_product_attn(self, q, k, v, mask):
        attn_sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_sc = attn_sc.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_sc, dim=-1)
        out = torch.matmul(attn_probs, v)
        return out

    def split_heads(self, x):
        batch_size, seq_length, ids = x.size()
        return x.view(batch_size, seq_length, self.heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.ids)

    def forward(self, q, k, v, mask):
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))

        attn_out = self.scaled_dot_product_attn(q, k, v, mask)
        out = self.w_o(self.combine_heads(attn_out))
        return out


class FeedForwardLayer(nn.Module):  # FORWARD PASS LAYER
    def __init__(self, ids):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(ids, 4 * ids),
            nn.GELU(),
            nn.Linear(4 * ids, ids)
        )

    def forward(self, x):
        return self.ff(x)


class PositionalEncoding(nn.Module):         # WORD POSITION
    def __init__(self, ids, max_s_length):
        super().__init__()
        pe = torch.zeros(max_s_length, ids)
        pos = torch.arange(0, max_s_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, ids, 2).float() * -(math.log(10000.0) / ids))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        if x.dim() == 3:
            return x + self.pe[:, :x.size(1)]
        if x.dim() == 2:
            return x.unsqueeze(1) + self.pe[:, :x.size(1)]
        else:
            raise ValueError("Input tensor must be 2D or 3D")


class EncoderLayer(nn.Module):  # THE HIDDEN LAYER(ENCODE)
    def __init__(self, ids, heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(ids, heads)
        self.ff = FeedForwardLayer(ids)
        self.norm1 = nn.LayerNorm(ids)
        self.norm2 = nn.LayerNorm(ids)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):    # THE HIDDEN LAYER(DECODE)
    def __init__(self, ids, heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(ids, heads)
        self.cross_attn = MultiHeadAttention(ids, heads)
        self.f_f = FeedForwardLayer(ids)
        self.norm1 = nn.LayerNorm(ids)
        self.norm2 = nn.LayerNorm(ids)
        self.norm3 = nn.LayerNorm(ids)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        ff_out = self.f_f(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class Transformer_Generate(nn.Module):           # ARCHITECTURE FOR TRAINING
    def __init__(self, src_size, tgt_size, ids, heads, layers, max_s_length, dropout):
        super().__init__()
        self.enc_emb = nn.Embedding(src_size, ids)
        self.dec_emb = nn.Embedding(tgt_size, ids)
        self.pos_enc = PositionalEncoding(ids, max_s_length)

        self.enc_l = nn.ModuleList([EncoderLayer(ids, heads, dropout) for _ in range(layers)])
        self.dec_l = nn.ModuleList([DecoderLayer(ids, heads, dropout) for _ in range(layers)])

        self.classifier = nn.Linear(ids, CLASSES)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def gen_mask(src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = MAX_LENGTH
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(device)
        return src_mask, tgt_mask

    def forward(self, src, tgt=None):
        src_mask, tgt_mask = self.gen_mask(src, tgt)
        src_emb = self.dropout(self.pos_enc(self.enc_emb(src))).to(device)
        tgt_emb = self.dropout(self.pos_enc(self.dec_emb(tgt))).to(device)

        enc_out = src_emb
        for enc_l in self.enc_l:
            enc_out = enc_l(enc_out, src_mask)

        dec_out = tgt_emb
        tgt_mask = tgt_mask.unsqueeze(1)
        for dec_l in self.dec_l:
            dec_out = dec_l(dec_out, enc_out, src_mask, tgt_mask)

        output = self.classifier(dec_out[:, 0, :])
        return output


class Transformer(nn.Module):
    def __init__(self, src_size, ids, heads, layers, dropout):
        super().__init__()
        self.enc_emb = nn.Embedding(src_size, ids)
        self.enc_l = nn.ModuleList([EncoderLayer(ids, heads, dropout) for _ in range(layers)])
        self.classifier = nn.Linear(ids, CLASSES)
        dropout = 0.1
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        src_emb = self.dropout(self.enc_emb(src)).to(device)

        enc_out = src_emb
        for enc_l in self.enc_l:
            enc_out = enc_l(enc_out, src_mask)

        return self.classifier(enc_out[:, 0, :])


def save_checkpoint(epoch, batch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'model_checkpoint.pth')


def load_checkpoint():
    try:
        checkpoint = torch.load('model_checkpoint.pth')
        return checkpoint
    except FileNotFoundError:
        return None
# LEARNING FUNCTION


total_batches = len(TRAIN_DL)


def train_model(model, train_dl, test_dl, criterion, optimizer, epochs):
    epoch = 0
    batch = 0
    model.to(device)
    model.train()

    # LOADING TEMP RESULTS (IF THEY EXIST)
    checkpoint = load_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        batch = checkpoint['batch'] + 1
        epoch = checkpoint['epoch']
        print(f"Temp results found. Starting from epoch {epoch + 1}, batch {batch}.")
    else:
        print("Can't find the temp results. Starting the new training")

    for epoch in range(epoch, epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_dl, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', leave=False, position=0)
        progress_bar.n = batch
        progress_bar.refresh()
        # PROGRESS BAR
        for inp_batch, target_batch in progress_bar:
            optimizer.zero_grad()
            out_batch = model(inp_batch)
            loss = criterion(out_batch, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch += 1

        avg_loss = total_loss / len(train_dl)
        progress_bar.n = 0
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inp_batch, target_batch in tqdm(test_dl, desc='Testing', unit='batch', leave=False):
                out_batch = model(inp_batch)
                _, predicted = torch.max(out_batch, 1)
                total_correct += (predicted == target_batch).sum().item()
                total_samples += target_batch.size(0)

        accuracy = total_correct / total_samples
        print(f'Epochs {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        batch = 0
        progress_bar.n = 0


def test_model(model, test_dl):
    model.load_state_dict(torch.load('adv_model.pth', map_location=torch.device('cpu')), strict=False)
    model.eval()

    with torch.no_grad():
        for inp_batch in tqdm(test_dl, desc='Testing', unit='batch', leave=False):
            out_batch = model(inp_batch)
            for i in range(len(inp_batch)):
                input_data = inp_batch[i]
                prediction = torch.argmax(out_batch[i]).item()
                print(f"Input Data: {input_data}")
                print(f"Prediction: {prediction}\n")
# MODEL INIT, OPTIMIZER, CRITERION


MODEL = Transformer(SRC_SIZE, IDS, HEADS, LAYERS, DROPOUT)
MODEL.apply(weights_init)
OPTIMIZER = AdaBound(MODEL.parameters(), lr=1e-5, final_lr=1e-3, weight_decay=0.01)
for state in OPTIMIZER.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.to(device)
CRITERION = nn.CrossEntropyLoss()

# USER INTERFACE
print("What we are doing?")
print("1. Training")
print("2. Testing")
print("3. Generate_Training")
choice = input("Choose option : ")
# TEST AND TRAINING
if choice == "1":
    EPOCHS = int(input("Set the number of epochs: "))
    train_model(MODEL, TRAIN_DL, VALIDATION_DL, CRITERION, OPTIMIZER, EPOCHS)
    torch.save(MODEL.state_dict(), 'adv_model.pth')
    print("The model saved successfully.")
elif choice == "2":
    test_model(MODEL, TEST_DL)
elif choice == "3":
    MODEL = Transformer_Generate(SRC_SIZE, TGT_SIZE, IDS, HEADS, LAYERS, MAX_LENGTH, DROPOUT)
else:
    print("The Wrong Input, Please choose the 1 or 2.")
