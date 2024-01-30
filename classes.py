
import torch
import torch.nn as nn
import torch.nn.functional as F
from funcoes import accuracy, to_device




''' Definindo o Modelo Base '''

# Modelo base
class ImageClassificationBase(nn.Module):

    # Construtor
    def __int__(self):
        super().__init__()
        self.history = None
        self.maxAcc = None
        self.evalAcc = None

    # Passo de treinamento
    def training_step(self, batch):

        # Batch
        images, labels = batch

        # Gera as previsões
        out = self(images)

        # Calcula o erro do modelo
        loss = F.cross_entropy(out, labels)

        return loss

    # Passo de validação
    def validation_step(self, batch):

        # Batch
        images, labels = batch

        # Gera as previsões
        out = self(images)

        # Calcula o erro
        loss = F.cross_entropy(out, labels)

        # Calcula a acurácia
        acc = accuracy(out, labels)

        return {'val_loss': loss.detach(), 'val_acc': acc}

    # Finaliza o passo de validação
    def validation_epoch_end(self, outputs):

        # Erro do batch
        batch_losses = [x['val_loss'] for x in outputs]

        # Erro da época
        epoch_loss = torch.stack(batch_losses).mean()

        # Acurácia de todos os batches e épocas
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # Print das métricas em cada batch
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch + 1, result['train_loss'], result['val_loss'], result['val_acc']))

''' Modelos ResNet '''






















