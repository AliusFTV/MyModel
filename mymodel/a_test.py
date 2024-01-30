from datasets import load_dataset
DATASET = load_dataset("glue", "mnli")
TRAIN_DATA = DATASET["train"]
batch_size = 8
def collate_fn(batch):
    i_texts = [example["premise"] + " [SEP] " + example["hypothesis"] for example in batch]
    chars = sorted(list(set(i_texts)))
    vocab_size = len(i_texts)
    print(vocab_size)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])
    print(encode)
    print(decode)
    return encode, decode

collate_fn(batch_size)
