from collections import OrderedDict

from typing_extensions import NamedTuple

import torch
from torch import nn

# - To subsample by a factor of 2:
#     First Convolution:  Kernel size = 16x16 Stride = 2 Padding = 7 (preserve the input-output size relationship)
#     Second Convolution: Kernel size = 16x16 Stride = 2 Padding = 7
# - To preserve the size: Use a stride of 1 and a padding of 8 with a 16 × 16 16×16 kernel

class ECGResBlock(nn.Module):
    FILTER_BASE_SIZE = 64

    class Input(NamedTuple):
        width: int
        height: int
        depth: int
        dropout_probability: float
        downsampling: bool
        first: bool # il primo layer e' speciale
        k: int # vedi il paper, e' un moltiplicatore della depth. i filtri usano 64k fogli
        device: torch.device

    def __init__(self, cinput: Input) -> None:
        super(ECGResBlock, self).__init__()
        self.conf = cinput
        conv_params =  {
            'stride': 2 if cinput.downsampling else 1,
        }

        self.branch0 = nn.Sequential(OrderedDict(([
            ('bn0', nn.BatchNorm2d(num_features=self.conf.depth, device=cinput.device)),
            ('relu0', nn.ReLU(inplace=True)),
            ('dropout0', nn.Dropout2d(p=self.conf.dropout_probability, inplace=True))
        ] if cinput.first else []) + [
            ('conv0', nn.Conv2d(in_channels=self.conf.depth, out_channels=ECGResBlock.FILTER_BASE_SIZE * cinput.k,
                                kernel_size=16, stride=1, padding=7, device=cinput.device)),
            ('bn1', nn.BatchNorm2d(num_features=ECGResBlock.FILTER_BASE_SIZE * cinput.k, device=cinput.device)),
            ('relu1', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout2d(p=self.conf.dropout_probability, inplace=True)),
            ('conv1', nn.Conv2d(in_channels=ECGResBlock.FILTER_BASE_SIZE * cinput.k,
                                out_channels=ECGResBlock.FILTER_BASE_SIZE * cinput.k,
                                kernel_size=16, stride=conv_params['stride'], padding=8, device=cinput.device))
        ]))

        self.branch1 = nn.Sequential(OrderedDict([
             ('maxpool', nn.MaxPool2d(kernel_size=2 if cinput.downsampling else 3, stride=conv_params['stride'], padding=0 if cinput.downsampling else 1)),
             ('bottleneck', nn.Conv2d(in_channels=cinput.depth, out_channels=ECGResBlock.FILTER_BASE_SIZE * cinput.k,
                                      kernel_size=1, stride=1, padding=0, device=cinput.device)),
        ]))

        self.to(cinput.device)
        match self.conf.device.type:
            case 'cuda':
                self.stream0 = torch.cuda.Stream(device=self.conf.device)
                self.stream1 = torch.cuda.Stream(device=self.conf.device)
                self.rendezvous_event01 = torch.cuda.Event()
            case dev if dev != 'cpu':
                raise ValueError('unsupported device type')

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
                    displacement: torch.Tensor = self.branch0(batch)
                with torch.cuda.stream(self.stream1):
                    residual: torch.Tensor = self.branch1(batch)
                    self.rendezvous_event01.record()
                with torch.cuda.stream(self.stream0):
                    self.stream0.wait_event(self.rendezvous_event01)
                    y = displacement + residual
            case 'cpu':
                displacement: torch.Tensor = self.branch0(batch)
                residual: torch.Tensor = self.branch1(batch)
                y = displacement + residual
            case _:
                raise ValueError(f'unexpected device {self.conf.device}')

        return y


class ECGNet(nn.Module):
    NUM_CLASSES = 14

    class Input(NamedTuple):
        width: int
        height: int
        depth: int
        device: torch.device

    def __init__(self, cinput: Input) -> None:
        super(ECGNet, self).__init__()
        self.conf = cinput
        inputs: list[ECGResBlock.Input] = [ECGResBlock.Input(
            width = max(cinput.width // (2 ** (i // 2)), 1),
            height = max(cinput.height // (2 ** (i // 2)), 1),
            depth = cinput.depth if i == 0 else ECGResBlock.FILTER_BASE_SIZE * (1 + i // 4),
            dropout_probability = 0.5,
            downsampling = i % 2 == 1,
            first = i == 0,
            k = 1 + i // 4,
            device = cinput.device,
        ) for i in range(0, 16)]
        last_num_features = inputs[len(inputs)-1].k * ECGResBlock.FILTER_BASE_SIZE
        last_numel = last_num_features * inputs[len(inputs)-1].width * inputs[len(inputs)-1].height \
                     // (4 if inputs[len(inputs)-1].downsampling else 1)

        self.warmup = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=cinput.depth, out_channels=cinput.depth,
                                kernel_size=3, stride=1, padding=1, device=cinput.device)),
            ('bn0', nn.BatchNorm2d(num_features=cinput.depth, device=cinput.device)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.residualnet = nn.Sequential(OrderedDict([
            ('res' + str(i), ECGResBlock(inputs[i])) for i in range(0, len(inputs) -1)
        ]))
        self.classnet = nn.Sequential(OrderedDict([
            ('bnc', nn.BatchNorm2d(num_features=last_num_features, device=cinput.device)),
            ('reluc', nn.ReLU(inplace=True)),
            ('flat', nn.Flatten()),
            ('dense', nn.Linear(last_numel, ECGNet.NUM_CLASSES, device=cinput.device)),
        ]))

        self.to(cinput.device)
        if self.conf.device.type != 'cpu' and self.conf.device.type != 'cuda':
            raise ValueError('unsupported device type')

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        y = self.classnet(self.residualnet(self.warmup(batch)))
        return y
    
# test spicciolo
if __name__ == '__main__':
    # Test configuration for a model
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Set up the input configuration
    input_config = ECGNet.Input(width=128, height=128, depth=3, device=device)

    # Create an instance of the ECGNet model
    model = ECGNet(cinput=input_config)

    # Create a dummy input tensor with the shape (batch_size, depth, height, width)
    # Example: batch size of 4, depth=3 (e.g., RGB channels), height=128, width=128
    batch_size = 4
    dummy_input = torch.randn(batch_size, input_config.depth, input_config.height, input_config.width, device=device)

    # Forward pass through the model
    output = model(dummy_input)

    # Print the output shape (it should be [batch_size, NUM_CLASSES])
    expected_shape = torch.Size((4, 14))
    print(f"Output shape: {output.shape}, expected shape: {expected_shape}")