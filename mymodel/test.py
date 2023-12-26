import torch
from transformers import BertTokenizer
from datasets import load_dataset

d_model = 512
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', return_tensors="pt")
dataset = load_dataset("glue", "mnli")
train_data = dataset["train"]
test_data = dataset["validation_matched"]
batch_size = 8
def collate_fn(batch):
    input_texts = [example["premise"] + " [SEP] " + example["hypothesis"] for example in batch]
    target_texts = [example["label"] for example in batch]

    input_data = tokenizer(input_texts, return_tensors="pt", padding=True)["input_ids"].clone().detach()
    print(f"Expected d_model: {d_model}, Actual d_model: {input_data.size(-1)}")
    target_data = torch.tensor(target_texts)
    return input_data, target_data
collate_fn()