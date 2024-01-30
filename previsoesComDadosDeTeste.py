import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataSet import dataset, dados_teste, best_model, previsao_imagem

# Previsão
img, label = dados_teste[52]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Previsto:', previsao_imagem(img, best_model))

# Previsão
img, label = dados_teste[34]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Previsto:', previsao_imagem(img, best_model))

# Previsão
img, label = dados_teste[19]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Previsto:', previsao_imagem(img, best_model))