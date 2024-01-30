import torch


# Função para calcular a acurácia do modelo
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Função de split (divisão 90/5/5)
def split(length) -> tuple:
    r = .05
    test_val = int(round(length*r,0))
    train = int(round(length-test_val*2,0))
    return (train,test_val,test_val)

# Função que move os tensores para o device
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Define nenhum decorador de gradiente para o método de avaliação e define função para avaliação do modelo
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()

    # Coleta saídas para cada lote
    outputs = [model.validation_step(batch) for batch in val_loader]

    # Envia a saída completa para a função de modelo de fim de época
    return model.validation_epoch_end(outputs)


# Função para o treinamento
def fit(epochs, model, train_loader, val_loader, optimizer):
    # Histórico de treinamento
    history = []

    # Loop
    for epoch in range(epochs):

        # Chama o método de treinamento do modelo
        model.train()

        # Erros de. treino
        train_losses = []

        # Loop pelos batches do data loader
        for batch in train_loader:
            # Calcula o erro
            loss = model.training_step(batch)

            # Adiciona à lista de error
            train_losses.append(loss)

            # Backpropagation
            loss.backward()

            # Otimiza o modelo
            optimizer.step()

            # Não calcula gradiente para o batch
            optimizer.zero_grad()

            # Validação do modelo
        result = evaluate(model, val_loader)

        # Média de erro em treino
        result['train_loss'] = torch.stack(train_losses).mean().item()

        # Print do andamento das épocas
        model.epoch_end(epoch, result)

        # Histórico de treino
        history.append(result)

    return history


# Função para encontrar o melhor modelo
def encontra_melhor_modelo(models):
    # Melhor modelo (maior acurácia)
    best_model = max([sum((model.maxAcc, model.evalAcc)) for model in models])

    return [model for model in models if sum((model.maxAcc, model.evalAcc)) == best_model][0]





