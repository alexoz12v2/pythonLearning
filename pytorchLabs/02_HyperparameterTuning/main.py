import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

n1: int = 300
n2: int = 150

class FstNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # funzione che converte l'immagine, tensore di floats, in un array piatto
        self.flatten = nn.Flatten()

        # poi definisci tutti i layers della rete, che se usi feedforward allora puoi usare la funzione nn.Sequential per semplificarti la vita
        self.feedforward = nn.Sequential(
            # layer 1: da immagine 28x28 a vettore 784 con funzione di attivazione ReLU
            nn.Linear(784, n1), nn.ReLU(),
            nn.Linear(n1, n2), nn.ReLU(),
            nn.Linear(n2, 10), nn.ReLU()
            # nn.Softmax(dim=1) -> feedforward deve ritornare logits, dopodiche applichi, dopo che chiami il modello, la funzione di softmax
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.feedforward(x)
        return logits


# Nel training loop, l'ottimizzazione avviene in 3 passi:
# 1. Chiama zero_grad() per resettare il gradiente (di default, ogni volta che chiami model() il gradiente viene calcolato e accumulato)
# 2. Backpropagation con loss.backwards()
# 3. optimizer.step() per aggiustare i parametri a seconda del gradiente calcolato e i parametri dell'ottimizzatore
def train(device: str, dataloader: DataLoader, model: nn.Module,
          loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optimizer: torch.optim.Optimizer, epoch: int,
          do_print=True, /) -> None:
    # you cannot always use len(dataloader.dataset) to get the number of samples, like you can do with len(dataset), because
    # the dataloader refers to the original dataset but might be using a sampler of it.
    # So the correct way is => size of the sampler = number of batches * batch size
    size = len(dataloader) * dataloader.batch_size

    # attiva la modalita di training al model
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # mi assicuro che tutti gli array con cui lavoro sono nello stesso device (magari lo sono gia)

        # errore di predizione
        logits = model(X)
        pred = logits

        # predProbabilities: torch.Tensor = nn.Softmax(dim=1)(logits) # calcola la funzione softmax lungo la colonna (perche ogni riga e' un input)
        # Nota: La cross entropy lavora con i logits, quindi non applicare softmax prima di calcolare la cross entropy
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print every 50 batches
        if batch % 50 == 0 and do_print:
            print(f'Training Epoch: {epoch} [{batch * len(X)}/{size} 100%] Loss: {loss.item():.6f}')


def test(device: str, dataloader: DataLoader, model: nn.Module,
         loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], do_print=True,
         /) -> (torch.float, torch.float):  # il dataloader preso qua e' quello di test
    size = len(dataloader) * dataloader.batch_size
    num_batches = len(dataloader)  # la uso per il calcolo della loss media

    # metti il modello in modalita di test
    model.eval()

    # inizializza a 0 loss e accuratezza
    test_loss, correct = 0, 0
    with torch.no_grad():  # disabilita il calcolo automatico del gradiente quando chiami model(input)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            pred = logits
            # pred = nn.Softmax(dim=1)(logits)
            # Nota: la cross entropy lavora con i logits, infatti dalla documentazione "nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss" (NLL e la negative log likelyhood)

            # somma il contributo della loss (.item() per 'spacchettarlo' dal tipo pytorch)
            test_loss += loss_fn(pred, y)

            # somma il contributo della accuratezza
            # 1. calcola un boolean, confrontando l'indice della probabilita piu grossa dal vettore della softmax (ovvero la classe scelta), con la classe vera (y)
            # 2. converti in un torch.float
            # 3. lo spacchetti e ottieni il tipo nativo python (.item())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    if do_print:
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct:.4f}')
    return test_loss, correct


# iperparametri:
# 1. Batch Size
# 2. Numero di Epoche
# 3. Learning Rate
def main(batch_size: int, epochs: int, /) -> None:
    # trasformazione che trasforma la matrice RGB in un tensore di float [0, 1], seguita da una normalizzazione del dominio
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.2806, std=0.3530)
    ])

    training_data = torchvision.datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )
    #mean, std = np.mean(training_data.data.numpy() / 255), np.std(training_data.data.numpy() / 255)

    # assegna 10% del training set al validation set
    training_data, validation_data = torch.utils.data.dataset.random_split(training_data, [0.9, 0.1])

    test_data = torchvision.datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )

    train_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_data_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'using {device} device')

    loss_fn = nn.CrossEntropyLoss()

    # per ora parto con learning rates fissi, poi vediamo come pytorch implementa la possibilita di ottimizzare il learning rate con gli scheduler
    learning_rates = [0.1, 0.01, 1e-3, 1e-4]
    train_err = torch.zeros(len(learning_rates))
    val_err = torch.zeros(len(learning_rates))
    models = []
    best_model, min_val_err = None, 100

    for i, learning_rate in enumerate(learning_rates):
        model = FstNeuralNetwork().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-3)

        # ogni epoca consiste di
        # 1. Train Loop
        # 2. Validation/Test Loop
        for t in range(epochs):
            print(f'Epoch {t}\n---------------------------------------------------')
            train(device, train_data_loader, model, loss_fn, optimizer, t)

        train_loss = 0.0
        model.eval()
        with torch.inference_mode(): # equivalente a no_grad
            for X, y in train_data_loader:
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                train_loss += loss_fn(y_hat, y).item()
            train_err[i] = train_loss / len(train_data_loader)

            val_loss = 0.0
            for X, y in validation_data_loader:
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                val_loss += loss_fn(y_hat, y).item()
            val_err[i] = val_loss / len(validation_data_loader)

        if val_err[i] < min_val_err:
            best_model = model
            min_val_err = val_err[i]

        models.append(model)

    print(f"best model is {best_model} with validation error of {min_val_err}.\nModel Summary: {summary(best_model, input_size=(1, 28, 28))}")

    test_errors = torch.zeros(len(learning_rates))
    test_accuracies = torch.zeros(len(learning_rates))
    for i, m in enumerate(models):
        test_errors[i], test_accuracies[i] = test(device, test_data_loader, m, loss_fn)

    # plotting dei learning rate e loss
    plt.semilogx(np.array(learning_rates), train_err.numpy(), label='total training loss')
    plt.semilogx(np.array(learning_rates), val_err.numpy(), label='total validation loss')
    plt.semilogx(np.array(learning_rates), test_errors.numpy(), label='total test loss')
    plt.xlabel('Learning Rate')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main(64, 5)
