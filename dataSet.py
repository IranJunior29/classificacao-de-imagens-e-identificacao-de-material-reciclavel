# Imports
import io
import pickle
import torch
import torch.nn as nn
import time
import os
import torchvision.transforms as T
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision.datasets import ImageFolder
from funcoes import split, encontra_melhor_modelo, to_device, fit, evaluate
from classes import ImageClassificationBase
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.models import resnet18, ResNet18_Weights, resnet152, ResNet152_Weights



# Modelo ResNet18
class ResNet18(ImageClassificationBase):

    # Construtor
    def __init__(self):
        # Init do construtor da classe mãe
        super().__init__()

        # Carrega o modelo pré-treinado
        self.network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Extrai o número de atributos
        num_ftrs = self.network.fc.in_features

        # Replace da última camada
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))

        self.name = 'ResNet18'

    # Método forward
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


# Modelo ResNet152
class ResNet152(ImageClassificationBase):

    # Construtor
    def __init__(self):
        # Init do construtor da classe mãe
        super().__init__()

        # Carrega o modelo pré-treinado
        self.network = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

        # Extrai o número de atributos
        num_ftrs = self.network.fc.in_features

        # Replace da última camada
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))  # replace last layer

        self.name = 'ResNet152'

    # Método forward
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


# Classe usada para transferir os data loaders para o device
class DeviceDataLoader():

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)


# Classe para carregar o modelo
class DeviceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)


def busca_melhor_modelo(epochs, lr, models, train_loader, val_loader):
    # Lista para os resultados
    result_models = []

    # Intervalo entre cada execução para não aquecer muito o computador
    safety_sleep = 15

    # Loop
    for model in models:

        # Print
        print("\nIniciando o Treinamento do Modelo:", model.name)

        # Envia o modelo para o device
        model = to_device(model, device)

        # Define o otimizador
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Executa o método fit (treinamento)
        history = fit(epochs, model, train_loader, val_loader, optimizer)

        # Histórico de treinamento
        model.history = pd.DataFrame(history, columns=['train_loss', 'val_loss', 'val_acc'])

        # Valor máximo de acurácia em validação
        model.maxAcc = max([x['val_acc'] for x in history])

        # Valor máximo de acurácia em teste
        model.evalAcc = evaluate(model, dl_teste)['val_acc']

        # Salva o resultado
        result_models.append(model)

        # Não precisa de sleep se for o último modelo
        if model.name != nomes_modelos[-1]:
            print(f'Descansando {safety_sleep} segundos')
            time.sleep(safety_sleep)
            safety_sleep += 5

            # Encontra o melhor modelo
    best_model = encontra_melhor_modelo(result_models)

    print("\nTreinamento Concluído!")

    return best_model, result_models


# Função para as previsões
def previsao_imagem(img, model):
    # Batch de 1
    xb = to_device(img.unsqueeze(0), device)

    # Previsão
    yb = model(xb)

    # Índice da probabilidade mais alta
    prob, preds = torch.max(yb, dim=1)

    return dataset.classes[preds[0].item()]

# Função para as previsões
def classifica_imagem(img_path):
    # Carrega imagem
    image = Image.open(img_path)

    # Aplica mesma transformação aplicada antes do treino
    example_image = transformador(image)

    # Plot
    plt.imshow(example_image.permute(1, 2, 0))
    print("Previsão para a imagem:", previsao_imagem(example_image, best_model) + ".")

if __name__ == '__main__':

    # Define o device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    ''' Carregando o Conjunto de Dados '''

    # Pasta com as imagens
    data_dir = Path('dados')

    # Transformação
    transformador = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    # Carrega as imagens a aplica as transformações
    dataset = ImageFolder(data_dir, transform=transformador)

    ''' Preparação das Imagens '''

    # Split randômico das imagens
    dados_treino, dados_teste, dados_valid = random_split(dataset, split(len(dataset)),
                                                          generator=torch.Generator().manual_seed(42))

    # Tamanho do batch
    batch_size = 16

    # Data Loader de treino
    dl_treino = DataLoader(dataset=dados_treino,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True)

    # Data Loader de teste
    dl_teste = DataLoader(dataset=dados_teste,
                          batch_size=batch_size,
                          num_workers=4,
                          pin_memory=True)

    # Data Loader de validação
    dl_valid = DataLoader(dataset=dados_valid,
                          batch_size=batch_size,
                          num_workers=4,
                          pin_memory=True)

    ''' Transferindo os Data Loaders Para o Dispositivo (CPU ou GPU) '''

    # Move os data loaders para o device
    dl_treino = DeviceDataLoader(dl_treino, device)
    dl_teste = DeviceDataLoader(dl_teste, device)
    dl_valid = DeviceDataLoader(dl_valid, device)

    # Inicializa os modelos
    modelos = [ResNet18(), ResNet152()]

    # Nomes dos modelos
    nomes_modelos = [modelo.name for modelo in modelos]

    # Hiperparâmetros (número de épocas e taxa de aprendizado)
    num_epochs = 5
    taxa_aprendizado = 5.5e-5

    melhor_modelo, resultados = busca_melhor_modelo(num_epochs, taxa_aprendizado, modelos, dl_treino, dl_valid)

    ''' Salvando os Modelos em Disco '''

    # Extrai os nomes dos modelo
    nomes_modelos = [modelo.name for modelo in resultados]

    # Pasta para salvar os modelos
    pasta_modelos = Path('modelos')

    # Salva cada modelo com formato pickle
    for i in range(len(nomes_modelos)):
        file = open(os.path.join(pasta_modelos, f'{nomes_modelos[i]}.pkl'), 'wb')
        pickle.dump(resultados[i], file)
        file.close()

    ''' Carregando os Modelos do Disco '''

    # Carrega cada modelo
    loaded_models = []
    for i in range(len(nomes_modelos)):
        file = open(os.path.join(pasta_modelos, f'{nomes_modelos[i]}.pkl'), 'rb')
        loaded_models.append(DeviceUnpickler(file).load())
        file.close()

    # Melhor modelo
    best_model = encontra_melhor_modelo(loaded_models)

    print(best_model.name)
