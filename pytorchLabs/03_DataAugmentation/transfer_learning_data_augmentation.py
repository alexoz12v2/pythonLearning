# laboratorio 3:
# scaricare un modello da imagenet
# applicare transfer learning
# applicare data augmentation
import concurrent
import pathlib
import random
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, set_start_method
from tempfile import TemporaryDirectory
from typing import Tuple, Callable

import numpy as np
import pandas as pd
import torch.utils.data
import torchvision
from matplotlib.figure import Figure
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import os  # funzioni per il filesystem
from glob import glob  # permette di cercare dei filename con regular expression
from torchsummary import summary
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import transform
import matplotlib.pyplot as plt


# di solito la data augemntation che andiamo ad applicare e' online perche cosi non ci limitano il disco e potenza
# CPU GPU. Applicate le trasformazioni abbiamo dei dati nuovi.
# le trasformazioni sono applicate randomicamente (eg immagine ruotata, zoomata, colori leggermente diversi)
# lo svantaggio della data augmentation, acnhe se non lo osserveremo molto, e' che il caricamento delle immagini
# richiedono un disco veloci, e le trasformazioni sono applicate dalla CPU, e quindi devono essere abbastanza veloci affinche
# la GPU possa prendersi il prossimo batch e fare training

# on the fly data augmentation: Invece di preapplicare le trasformaizoni di data augmentation e salvarle su disco, le applico
# sulla CPU in fase di training mentre la GPU e' cccupata a fare un ciclo di training con il batch precedente

class CatsAndDogsDataset(Dataset):
    __labelmap = {'cats': 0, 'dogs': 1}

    # expected structure: ${root}/[cats, dogs]/<images>
    def __init__(self, root, *, device='cpu', train: bool, transform: torch.Tensor | dict[str, torch.Tensor] = None):
        non_none_t = transform if transform is not None else transforms.Resize((224, 224))
        self.__train_mode = train
        self.device = device
        self.root = pathlib.Path(root)  # decode_image prende un pathlib.Path
        self.transform = transform \
            if transform is not None and isinstance(transform, dict) \
            else {'train': non_none_t,
                  'val': non_none_t}
        self.image_files = []
        self.labels = []

        # controlla che root sia effettivamente una directory
        if not os.path.isdir(self.root):
            raise ValueError(f'Invalid dataset path, {self.root} does not exist')

        # controlla che ci siano le subdirectories che mi aspetto
        expected_classes = ['cats', 'dogs']

        # controlla che dentro il dataset ci siano sia cani e gatti
        for class_name in expected_classes:
            class_path = self.root / class_name
            if not class_path.is_dir():
                raise ValueError(f'Invalid dataset path, {class_path} does not exist')

            # prenditi tutti i filename dei file immagine immagine dalla directory
            self.image_files.extend(list(class_path.glob('*.jpg')))  # glob di path ritorna un generator, va convertito
            self.labels.extend([self.__labelmap[class_name]] * len(self.image_files))

            # mischia le cose
            elems = list(zip(self.image_files, self.labels))
            random.shuffle(elems)

            # ritornali sfusi (* e' l'operatore di spread)
            self.image_files, self.labels = zip(*elems)  # da lista di tuple a 2x tuple di immagini

            # converti in liste
            self.image_files = list(self.image_files)
            self.labels = list(self.labels)

    def train_mode(self):
        self.__train_mode = True

    def eval_mode(self):
        self.__train_mode = False

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            dataitem, label = self.open_single_image(idx)
            return dataitem, label
        elif isinstance(idx, slice):
            dataitems = []
            labels = []
            # ThreadPoolExecutor perche sono assai le immagini
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                indices = range(*idx.indices(len(self)))
                futures = {executor.submit(self.open_single_image, i) for i in indices}

                # accumula le immagini man mano che i threads le aprono
                for future in concurrent.futures.as_completed(futures):
                    try:
                        dataitem, label = future.result()
                        dataitems.append(dataitem)
                        labels.append(label)
                    except Exception as e:
                        raise ValueError('error opening image', e)
            return dataitems, labels  # torch.stack mi ritorna un tensore 4D, meglio di no
        else:
            raise TypeError('Index type not supported. Must be integer or slice')

    def open_single_image(self, idx: int):
        dataitem = torchvision.io.decode_image(self.image_files[idx]).to(self.device).to(torch.float32) / 255.0
        label = self.labels[idx]
        if self.transform:
            dataitem = self.transform['train' if self.__train_mode else 'val'](dataitem)
        return dataitem, label


def train_model(model: nn.Module, dataloaders: dict[str, DataLoader],
                criterion: Callable[[torch.Tensor, torch.Tensor], torch.float32],
                optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                num_epochs: int = 25) -> nn.Module:
    # fai benchmark della fase di training
    start = time.time()

    # uguale al comando mktemp, che mi crea una directory con nome univoco nella cwd
    # lo uso per fare un dump dei parametri iniziali e di quelli che mi danno errore di validation migliore
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html per piu' dettagli su salvataggio
    with TemporaryDirectory() as tempdir:
        best_model_params_path = pathlib.Path(tempdir) / 'best_model_params.pt'

        torch.save(model.state_dict(), best_model_params_path)
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # fai un giro di training con forward e backward pass, e poi valuta solo in forward la loss per validation
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                # per ogni roba nel dataloader
                for inputs, labels in dataloaders[phase]:
                    labels = labels.to(inputs.device)
                    # azzero i gradienti
                    optimizer.zero_grad()

                    # attiva il calcolo dei gradienti "storico" (funziona con model.train)
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward pass
                        outputs = model(inputs)
                        # calcolo la logit massima e prendo il suo indice fregandomene del valore
                        _, preds = torch.max(outputs, dim=1)
                        preds = preds.type(labels.dtype)
                        # calcolo la loss
                        loss = criterion(outputs, labels)

                        # backward pass + ottimizzazione iperparametri
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # calcolo delle statistiche
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)  # potrebbe crashare

                if phase == 'train':
                    scheduler.step()

                # calcolo la loss media per example
                epoch_loss = running_loss / (len(dataloaders[phase]) * dataloaders[phase].batch_size)
                epoch_acc = running_corrects.double() / (len(dataloaders[phase]) * dataloaders[phase].batch_size)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # se siamo in validazione ed e' il best so far, allora fai il dump su disco
                if phase == 'val' and epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        elapsed_time = time.time() - start
        print(f'Training Complete in {elapsed_time // 60:.0f} minutes {elapsed_time % 60:.0f} seconds')
        print(f'Best Accuracy: {best_accuracy:.4f}')

        # caricati il migliore modello
        model.load_state_dict(torch.load(best_model_params_path))
    return model


# funzione presa da tutorial pytorch, matplotlib non sono tanto sicuro
def imshow(inp, ax, title=None) -> None:
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    if title is not None:
        plt.title(title)

def visualize_model(model: nn.Module, dataloader: DataLoader, class_names: list[str], num_images=4) -> None:
    was_training = model.training # per ristabilirla dopo
    images_so_far = 0
    model.eval()
    fig = plt.figure()
    with torch.no_grad():
        i: int; inputs: torch.Tensor; labels: torch.Tensor
        for i, (inputs, labels) in enumerate(dataloader):
            labels = labels.to(inputs.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = fig.add_subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j], ax)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    fig.show()
                    return
        model.train(mode=was_training)
        fig.show()


def compute_mean_std(images: list[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.zeros(3).to(images[0].device)
    std = torch.zeros(3).to(images[0].device)
    num_images = len(images)

    for img in images:
        mean += img.mean(dim=[1, 2])  # 0 -> depth, 1 -> height, 2 -> width (se stack di immagini, 0 -> indice)
        std += img.std(dim=[1, 2])

    mean /= num_images
    std /= num_images

    return mean, std


def resize_if_smaller(image: torch.Tensor, size=(224, 224)) -> torch.Tensor:
    if image.size(1) < size[0] or image.size(2) < size[1]:
        image = transforms.Resize(size)(image)
    return image


def dataset_statistics(device):
    # Il file CSV nella path indicata, se esiste, contiene media e STD. se non esiste, allora apri momentaneamente
    # il dataset e calcola la media forkando piu processi e sfruttando cuda su piu' processes
    # probabile che se fai il tensor stacking e tensor.sum ti viene codice piu' facile, giusto per usare i processes
    csv_file_path = pathlib.Path(os.getcwd() + '/dataset/mean_std.csv')
    if not csv_file_path.exists():
        chunk_size = 64
        dataset = CatsAndDogsDataset(os.getcwd() + '/dataset/training_set', device=device, train=False)
        chunks = [dataset[i:i + chunk_size][0] for i in range(0, len(dataset), chunk_size)]
        with Pool(os.cpu_count()) as pool:
            results = pool.map(compute_mean_std, chunks)

        mean = torch.zeros(3).to(device)
        std = torch.zeros(3).to(device)
        for cur_mean, cur_std in results:
            mean += cur_mean
            std += cur_std
        mean /= len(results)
        std /= len(results)

        mean_std_df = pd.DataFrame(
            {'mean': mean.cpu(), 'std': std.cpu()})  # pandas vuole numpy. per conversione a numpy, servono nella cpu
        mean_std_df.to_csv(csv_file_path, index=False)
        print(f'saved CSV file with training dataset statistics in ${csv_file_path}')
    else:
        mean_std_df = pd.read_csv(csv_file_path)
        mean = mean_std_df['mean'].values
        std = mean_std_df['std'].values
    return mean, std


def main() -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    mean, std = dataset_statistics(device)

    # Importante: Prima di tutto voglio usare immagini a 224x224, perche' e' la size a cui lavora resnet50.
    # Per fare cio', applico in training un crop ad una porzione casuale dell'immagine, cosi da fare data augmentation
    # on the fly, aggiungendo stocasticita' al processo di training. Aggiungo anche un flip casuale.
    # In fase di valutazione invece, tale stocasticita' viene rimossa
    # manca la transforms.ToTensor perche' la applico io nella classe del mio dataset
    data_transforms = {
        'train': transforms.Compose([
            transforms.Lambda(lambda img: resize_if_smaller(img, size=(224, 224))), # crop esplode se img piu' piccola
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'val': transforms.Compose([
            transforms.Lambda(lambda img: resize_if_smaller(img, size=(224, 224))),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=mean, std=std)
        ])
    }

    # apri i dataset
    training_set = CatsAndDogsDataset(os.getcwd() + '/dataset/training_set', device=device, train=True,
                                      transform=data_transforms)
    test_set = CatsAndDogsDataset(os.getcwd() + '/dataset/test_set', device=device, train=False,
                                  transform=data_transforms)
    training_set, val_set = torch.utils.data.random_split(training_set, [0.8, 0.2])

    # configura i dataloader
    dataloaders = {
        'train': torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True, num_workers=0),
        'val': torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0),
        'test': torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
    }

    # seleziona un pretrained model (o pretrained=True o weights=quello che e)
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    print(summary(model, input_size=(3, 224, 224)))

    # Cambia 'ultimo layer fully connected in uno per classificazione binaria, e freeza tutti i layer tranne quello
    # appena aggiunto
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2).to(device)

    # definisci i parametri della funzione di training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) # non ho messo learning
    # ogni 7 epoche moltiplichi il learning rate di 0.1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, 1)
    print('siamo sopravvisuti!')
    visualize_model(model, dataloaders['train'], ['cat','dog'])
    torch.save(model, 'catsndogs.pt')


if __name__ == "__main__":
    # senza di questo CUDA esplode nei processi forkati
    set_start_method('spawn', force=True)
    main()
