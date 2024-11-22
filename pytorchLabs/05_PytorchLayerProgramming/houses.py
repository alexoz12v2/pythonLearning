from pathlib import Path

import PIL
import pandas as pd
import torch
from PIL.Image import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import transforms
from typing_extensions import NamedTuple


# appunti su python in generale perche sono ignorante
# Foo = type('Foo', (), { 'attr': 10, 'print_attr': lambda self: print(f"attr: {self.attr}") })
# e' equivalente a
# class Foo: { ... }
# perche le classi sono istanze delle metaclassi, e type e' una metaclasse, a sua volta istanza di type, cioe di se stessa

# houses dataset file line example (CSV format with whitespace as separator)
# numBathrooms numBedrooms houseAreaSqFt HousePriceUSD
# 4 4 3721 85255 500000
def read_txt_as_csv(file_path: Path, column_names: list[str], dtype: dict[str, any]) -> pd.DataFrame:
    """
    Parameters
    ----------
    file_path path al file di testo
    column_names nomi delle colonne
    dtype tipi per ogni colonna

    Returns dataframe contenente le tuple contenute nel file
    -------
    """
    df = pd.read_csv(file_path, sep='\s+', header=None,
                     names=column_names, dtype=dtype)
    return df

# L'input alla rete fully connected e' un tensore (batch_size x 4), 4 = [ numBathrooms, numBedrooms, houseAreaSqFt ]
# L'input alla rete convoluzionale e' un tensore (batch_size x 3 x width x height), dove width e height sono
#   variabili, quindi deve anche essere passata una transform in data augmentation-preprocessing per riportarmi a
#   (3 x 224 x 224) della resnet. Questo e' gestito vuori dal modello pero', quindi mi aspetto in input nella oforward
#   una tupla con due tensori
class HouseCostEstimationNet(nn.Module):
    room_types = ["bathroom", "bedroom", "frontal", "kitchen", "frontal"]
    num_room_types = 5

    class Input(NamedTuple):
        device: torch.device

    def __init__(self, cinput: Input) -> None:
        super(HouseCostEstimationNet, self).__init__()
        self.conf = cinput
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # vedi file resnet.py linea 292

        # il ramo della CNN mi sputa fuori (batch_size x 2048)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=2048, device=cinput.device),
            nn.ReLU(inplace=True),
        )

        # che si aggiungono ai (batch_size x 2048) sputati dal MLP per fare torch.cat
        self.mlp = nn.Sequential(
            nn.Linear(in_features=3, out_features=512, device=cinput.device),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=2048, device=cinput.device),
            nn.ReLU(inplace=True),
        )

        # layer fc finale segue la concatenazione
        self.fc = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048, device=cinput.device),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=512, device=cinput.device),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128, device=cinput.device),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=1, device=cinput.device),
        )

        # attiva soltanto l'ultimo gruppo di layers convolutivi, a 512 depth, e i layer FC, come layers che imparano
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        self.to(cinput.device)
        match self.conf.device.type:
            case 'cuda':
                self.stream0 = torch.cuda.Stream(device=cinput.device)
                self.stream1 = torch.cuda.Stream(device=cinput.device)
                self.rendezvous_event01 = torch.cuda.Event()
            case x if x != 'cpu':
                raise ValueError('unrecognized device')

    def forward(self, csv_data: torch.Tensor, img_data: torch.Tensor) -> torch.Tensor:
        if csv_data.ndim != 2 or img_data.ndim != 5: # img: B x N x C x H x W
            raise ValueError(f'unexpected ndims in input tensors csv_data {csv_data.ndim} 2 and img_data {img_data.ndim} 4')
        if csv_data.dtype != torch.float32 or img_data.dtype != torch.float32:
            raise ValueError(f'unexpected dtype: {csv_data.dtype} and {img_data.dtype}. They should be both float32')
        if csv_data.device != self.conf.device or img_data.device != self.conf.device:
            raise ValueError(f'unexpected device: {self.conf.device} csv_data {csv_data.device} img_data {img_data.device}')
        if csv_data.shape[1] != 3 or img_data.shape[1] > HouseCostEstimationNet.num_room_types \
                or img_data.shape[3] != 224 or img_data.shape[4] != 224:
            raise ValueError(f'unexpected shapes: 3 {csv_data.shape[1]}, 224 {img_data.shape[3]}, 224 {img_data.shape[4]}')

        match self.conf.device.type:
            case 'cuda':
                with torch.cuda.stream(self.stream0):
                    t_cnns0 = []
                    for i in range(0, HouseCostEstimationNet.num_room_types, 2):
                        t_cnns0.append(self.resnet.forward(img_data[:, i, :, :, :]))
                with torch.cuda.stream(self.stream1):
                    t_cnns1 = []
                    for i in range(1, HouseCostEstimationNet.num_room_types, 2):
                        t_cnns1.append(self.resnet.forward(img_data[:, i, :, :, :]))
                    t_mlp: torch.cuda.FloatTensor = self.mlp.forward(csv_data)
                    self.rendezvous_event01.record()
                with torch.cuda.stream(self.stream0):
                    self.stream0.wait_event(self.rendezvous_event01)
                    t_cnns0.extend(t_cnns1)
                    t_cnn = torch.stack(t_cnns0, dim=1).mean(dim=1)
                    t = torch.cat((t_cnn, t_mlp), dim=1)
                    t = self.fc.forward(t)
            case 'cpu':
                t_cnns = []
                for i in range(HouseCostEstimationNet.num_room_types):
                    t_cnns.append(self.resnet.forward(img_data[:, i, :, :, :]))
                t_cnn = torch.stack(t_cnns, dim=1).mean(dim=1)
                t_mlp: torch.Tensor = self.mlp.forward(csv_data)
                t = torch.cat((t_cnn, t_mlp), dim=1)
                t = self.fc.forward(t)
            case _:
                raise ValueError('unexpected device in forward')

        return t


# Define the dataset class
class HouseDataset(Dataset):
    def __init__(self, csv_file: Path, img_dir: Path, transform=None):
        """
        Args:
            csv_file (Path): Path to the HousesInfo.txt file.
            img_dir (Path): Directory with all the house images.
            transform: Image transformations to apply.
        """
        self.data = read_txt_as_csv(
            csv_file,
            column_names=["numBathrooms", "numBedrooms", "houseAreaSqFt", "HousePriceUSD"],
            dtype={"numBathrooms": float, "numBedrooms": float, "houseAreaSqFt": float, "HousePriceUSD": float},
        )
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slices
            csv_batch = []
            img_batch = []
            price_batch = []
            for i in range(*idx.indices(len(self.data))):
                csv_data, image, house_price = self[i]  # Recursive call for single index
                csv_batch.append(csv_data)
                img_batch.append(image)
                price_batch.append(house_price)
            return torch.stack(csv_batch), torch.stack(img_batch), torch.tensor(price_batch)

        # Handle single index
        images = []
        for room_type in HouseCostEstimationNet.room_types:
            img_path = self.img_dir / f"{idx}_{room_type}.jpg"  # Assuming filenames are 0.jpg, 1.jpg, etc.
            if img_path.exists():
                image = PIL.Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.PILToTensor()(transforms.Resize((224, 224))(image))  # Converts to (C x H x W)
                images.append(image)
            else:
                images.append(torch.zeros(3, 224, 224))
        row = self.data.iloc[idx]
        csv_data = torch.tensor([row["numBathrooms"], row["numBedrooms"], row["houseAreaSqFt"]], dtype=torch.float32)
        house_price = torch.tensor(row["HousePriceUSD"], dtype=torch.float32)

        return csv_data, torch.stack(images), house_price


# Test function
def house_model():
    # Paths
    houses_info_path = Path.cwd() / "houses/HousesInfo.txt"
    img_dir_path = Path.cwd() / "houses/"

    # Image transform
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader
    dataset = HouseDataset(csv_file=houses_info_path, img_dir=img_dir_path, transform=img_transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Model configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HouseCostEstimationNet(HouseCostEstimationNet.Input(device=device))
    model.eval()  # Set model to evaluation mode

    # Iterate through the DataLoader
    with torch.no_grad():
        for csv_data, img_data, house_price in data_loader:
            csv_data = csv_data.to(device)
            img_data = img_data.to(device)
            house_price = house_price.to(device)

            # Forward pass
            predictions = model.forward(csv_data, img_data)
            print(f"Predicted Prices: {predictions.squeeze().tolist()}")
            print(f"Actual Prices: {house_price.tolist()}")
            break  # Test one batch for demonstration


# Main entry point
if __name__ == '__main__':
    house_model()