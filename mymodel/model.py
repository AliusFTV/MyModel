from transformers import Transformer, TransformerConfig
import torch
import torch.nn as nn

# Определение конфигурации
config = TransformerConfig(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu'
)

# Создание модели
model = Transformer(config)

# Определение входных данных
input_data = torch.randn((10, 32, 512))  # Пример входных данных

# Прямой проход
output = model(input_data)