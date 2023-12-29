from transformers import BertConfig

config = BertConfig.from_pretrained('bert-base-uncased')
print(config.max_position_embeddings)