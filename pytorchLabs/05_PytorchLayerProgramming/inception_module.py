import torch
from typing_extensions import NamedTuple

import torch.nn as nn

# cerchero di andare piu a fondo nella gestione con cuda per fare computazioni in parallelo, da documentazione
# https://pytorch.org/docs/stable/notes/cuda.html

# per creare degli inputs strutturati, puoi usare
# - structs mutabili:   @dataclass decorator
# - structs immutabili: NamedTuple base class

# puoi fare bed of nails se il max pool dell'embedding fa https://stackoverflow.com/questions/71025321/indices-in-maxpool2d-in-pytorch

# nota: nn.Parameter essenziamlmente e' un wrapper per un tensore. la unica differenza e' che  poi nn.Module ispeziona
# la sua struttura, raccoglie tutti gli nn.Parameters e li rende disponibile nell'iteratore model.parameters()
class InceptionModule(nn.Module):
    class Input(NamedTuple):
        width: int
        height: int
        depth: int
        conv1x1_depth: int
        conv3x3_depth: int
        conv5x5_depth: int
        maxpool1_depth: int
        device: torch.device
    """
    - spatial input size stessa di spatial output size
    - osservazioni su depth: 192 -> 256, 256 -> 480, 480 -> 512, 512 -> 528, 528 -> 832, 832 -> 1024
    - forse meglio che faccio specificare in input device e profondita di ciascun filtro
    - mi aspetto che i tensori in input abbiamo il formato (depth x width x height)
    """
    def __init__(self, cinput: Input) -> None:
        super(InceptionModule, self).__init__()
        self.conf = cinput
        self.relu = nn.ReLU(inplace=True)

        # each component is indexed by its stream of computation
        self.conv1x1_0 = nn.Conv2d(in_channels=cinput.depth,
                                   out_channels=cinput.conv1x1_depth,
                                   kernel_size=1, stride=1, padding=0, device=cinput.device)

        self.conv1x1_1 = nn.Conv2d(in_channels=cinput.depth,
                                   out_channels=cinput.conv3x3_depth,
                                   kernel_size=1, stride=1, padding=0, device=cinput.device)
        self.conv3x3_1 = nn.Conv2d(in_channels=cinput.conv3x3_depth,
                                   out_channels=cinput.conv3x3_depth,
                                   kernel_size=3, stride=1, padding=1, device=cinput.device)

        self.conv1x1_2 = nn.Conv2d(in_channels=cinput.depth,
                                   out_channels=cinput.conv5x5_depth,
                                   kernel_size=1, stride=1, padding=0, device=cinput.device)
        self.conv5x5_2 = nn.Conv2d(in_channels=cinput.conv5x5_depth,
                                   out_channels=cinput.conv5x5_depth,
                                   kernel_size=5, stride=1, padding=2, device=cinput.device)

        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1_3 = nn.Conv2d(in_channels=cinput.depth,
                                   out_channels=cinput.maxpool1_depth,
                                   kernel_size=1, stride=1, padding=0, device=cinput.device)
        # assicurati che tutto, anche robe di classe base, siano su device
        self.to(cinput.device)
        match self.conf.device.type:
            case "cuda":
                self.stream0 = torch.cuda.Stream(device=cinput.device)
                self.stream1 = torch.cuda.Stream(device=cinput.device)
                self.stream2 = torch.cuda.Stream(device=cinput.device)
                self.stream3 = torch.cuda.Stream(device=cinput.device)
                # 0 aspetta 1,2,3 e poi fa cat
                self.rendezvous_event01 = torch.cuda.Event()
                self.rendezvous_event02 = torch.cuda.Event()
                self.rendezvous_event03 = torch.cuda.Event()
            case "cpu":
                return
            case _:
                raise ValueError("Unsupported device type")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.device != self.conf.device:
            raise ValueError(f'input device {batch.device} is not the expected device {self.conf.device}')
        if batch.ndim != 4 or batch.dtype != torch.float32:
            raise ValueError('unexpected torch.Tensor format')
        if batch.shape[1] != self.conf.depth or batch.shape[2] != self.conf.width or batch.shape[3] != self.conf.height:
            raise ValueError('unexpected torch.Tensor shape')

        match self.conf.device.type:
            case 'cuda':
                with torch.cuda.stream(self.stream0):
                    y0 = self.relu(self.conv1x1_0.forward(batch))
                with torch.cuda.stream(self.stream1):
                    y1 = self.relu(self.conv3x3_1.forward(self.relu(self.conv1x1_1.forward(batch))))
                    self.rendezvous_event01.record(self.stream1)
                with torch.cuda.stream(self.stream2):
                    y2 = self.relu(self.conv5x5_2.forward(self.relu(self.conv1x1_2.forward(batch))))
                    self.rendezvous_event02.record(self.stream2)
                with torch.cuda.stream(self.stream3):
                    y3 = self.relu(self.conv1x1_3.forward(self.maxpool_3.forward(batch)))
                    self.rendezvous_event03.record(self.stream3)
                with torch.cuda.stream(self.stream0):
                    self.stream0.wait_event(self.rendezvous_event01)
                    self.stream0.wait_event(self.rendezvous_event02)
                    self.stream0.wait_event(self.rendezvous_event03)
                    y = torch.cat([y0, y1, y2, y3], dim=1)
            case 'cpu':
                y0 = self.relu(self.conv1x1_0.forward(batch))
                y1 = self.relu(self.conv3x3_1.forward(self.relu(self.conv1x1_1.forward(batch))))
                y2 = self.relu(self.conv5x5_2.forward(self.relu(self.conv1x1_2.forward(batch))))
                y3 = self.relu(self.conv1x1_3.forward(self.maxpool_3.forward(batch))) # maxPool2D ritorna Tensor se gli passi return_indices a False, altrimenti ti da Tuple[Tensor, Tensor]
                y = torch.cat([y0, y1, y2, y3], dim=1)
            case _:
                raise ValueError('unsupported device')

        return y


# codice di testing
if __name__ == "__main__":
    # Define a sample device and input configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the input dimensions and filter depths
    input_config = InceptionModule.Input(
        width=64,  # Width of the input
        height=64,  # Height of the input
        depth=192,  # Depth (channels) of the input tensor
        conv1x1_depth=256,  # Depth of the 1x1 convolution
        conv3x3_depth=480,  # Depth of the 3x3 convolution
        conv5x5_depth=512,  # Depth of the 5x5 convolution
        maxpool1_depth=832,  # Depth of the max pooling operation
        device=device  # Device (cuda or cpu)
    )

    # Create a dummy input tensor (depth x width x height)
    torch.random.manual_seed(42)
    x = torch.randn((input_config.depth, input_config.width, input_config.height), device=device, dtype=torch.float32)
    x.unsqueeze_(dim=0) # quelli con l'underscore sono i metodi mutable

    # Instantiate the InceptionModule
    model = InceptionModule(input_config)

    # Perform a forward pass
    output = model(x)

    # Print the shape of the output tensor to verify everything works
    expected_depth = input_config.conv1x1_depth + input_config.conv3x3_depth + input_config.conv5x5_depth + input_config.maxpool1_depth
    expected_shape = torch.Size((1, expected_depth, input_config.width, input_config.height))
    print(f"Output shape: {output.shape}, Expected shape: {expected_shape}")