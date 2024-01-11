vocab_size = 30522
d_model = 1024
num_layers = 6
batch_size = 16
num_classes = 3
nhead = 8
dim_feedforward = 2048

# Параметры трансформера
num_params_transformer = (4 * d_model * d_model + 4 * d_model) * nhead * num_layers

# Параметры полносвязного слоя для классификации
num_params_classifier = (d_model + 1) * num_classes

# Параметры входного слоя (учитываем размер словаря)
num_params_input = d_model * vocab_size

# Параметры слоя прямого прохода (для каждого из 6 слоев)
num_params_feedforward = (2 * d_model * dim_feedforward + 2 * dim_feedforward) * num_layers

# Общее количество параметров
total_params = num_params_input + num_params_transformer + num_params_classifier + num_params_feedforward

print("Total Parameters:", total_params)