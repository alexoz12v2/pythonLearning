import sys
import traceback

import torch
from torch import nn
from typing_extensions import NamedTuple, OrderedDict


class StreamConvNetBlock(nn.Module):
    EXPECTED_SPATIAL_SIZE = 224

    class Input(NamedTuple):
        device: torch.device
        depth: int # canali in input, size fissa a 224

    def __init__(self, cinput: Input) -> None:
        super(StreamConvNetBlock, self).__init__()
        self.conf = cinput

        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential( # 3x224x224 -> 96x55x55
                nn.Conv2d(in_channels=cinput.depth, out_channels=96, kernel_size=7,
                          stride=2, padding=1, device=cinput.device),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(num_features=96, device=cinput.device),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            )),
            ('conv2', nn.Sequential( # 96x55x55 -> 256x14x14
                nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=2, device=cinput.device),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(num_features=256, device=cinput.device),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            )),
            ('conv3', nn.Sequential( # 256x14x14 -> 512x14x14
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, device=cinput.device),
                nn.ReLU(inplace=True),
            )),
            ('conv4', nn.Sequential( # 512x14x14 -> 512x14x14
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, device=cinput.device),
                nn.ReLU(inplace=True),
            )),
            ('conv5', nn.Sequential( # 512x14x14 -> 256x4x2 (con paper e' 256x7x7) ...
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, device=cinput.device),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=2, stride=2, padding=0), ... perche non mi accucchio con le dimensioni
                nn.MaxPool2d(kernel_size=(4, 12), stride=4, padding=1),
            )),
            ('full6', nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=4096, out_features=2048, device=cinput.device),
                nn.Dropout1d(p=0.5, inplace=True),
                nn.ReLU(inplace=True),
            )),
            ('full7', nn.Sequential(
                nn.Linear(in_features=2048, out_features=100, device=cinput.device),
                nn.Dropout1d(p=0.5, inplace=True),
            )),
        ]))

        self.to(cinput.device)
        if self.conf.device.type != 'cpu' and self.conf.device.type != 'cuda':
            raise ValueError('unsupported device type')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.dtype != torch.float32:
            raise ValueError('Unexpected tensor format')
        if x.shape[2] != StreamConvNetBlock.EXPECTED_SPATIAL_SIZE \
            or x.shape[3] != StreamConvNetBlock.EXPECTED_SPATIAL_SIZE:
            raise ValueError('Unexpected tensor shape')

        y: torch.Tensor = x
        for layer in iter(self.net):
            for sublayer in iter(layer):
                y = sublayer.forward(y)

        #y: torch.Tensor = self.net.forward(x)
        return y

class StreamConvNet(nn.Module):
    class Input(NamedTuple):
        device: torch.device
        L: int # numero di frames nello stream temporale. nota che la depth del temporal stream viene 2L perche ogni
               # immagine dell'optical flow e' un vettore 2D
    def __init__(self, cinput: Input) -> None:
        super(StreamConvNet, self).__init__()
        self.conf = cinput

        self.spatial = StreamConvNetBlock(StreamConvNetBlock.Input(
            device = cinput.device,
            depth = 3
        ))

        self.temporal = StreamConvNetBlock(StreamConvNetBlock.Input(
            device = cinput.device,
            depth = 2 * cinput.L,
        ))

        self.to(cinput.device)
        match self.conf.device.type:
            case 'cuda':
                self.stream0 = torch.cuda.Stream(self.conf.device)
                self.stream1 = torch.cuda.Stream(self.conf.device)
                self.rendezvous_event01 = torch.cuda.Event()
            case x if x != 'cpu':
                raise ValueError('Unexpected device type')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.dtype != torch.float32:
            raise ValueError('Unexpected tensor format')

        # extract the first 3 from all
        s = x[:, 0:3, :, :]
        t = x[:, 3:, :, :]

        match self.conf.device.type:
            case 'cuda':
                with torch.cuda.stream(self.stream0):
                    y0 = self.spatial.forward(s)
                    p0 = torch.softmax(y0, dim=1)
                with torch.cuda.stream(self.stream1):
                    y1 = self.temporal.forward(t)
                    p1 = torch.softmax(y1, dim=1)
                    self.rendezvous_event01.record()
                with torch.cuda.stream(self.stream0):
                    self.stream0.wait_event(self.rendezvous_event01)
                    p0.add_(p1).div_(2)
            case 'cpu':
                y0 = self.spatial.forward(s)
                y1 = self.temporal.forward(t)
                p0 = torch.softmax(y0, dim=1)
                p1 = torch.softmax(y1, dim=1)
                p0.add_(p1).div_(2)
            case _:
                raise ValueError('Unexpected device type')

        return p0


if __name__ == "__main__":
    # Check for CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Parameters
    rgb = 3    # RGB channels
    L   = 10   # Number of frames in the temporal stream
    spatial_size = StreamConvNetBlock.EXPECTED_SPATIAL_SIZE

    # Initialize the StreamConvNet
    net_input = StreamConvNet.Input(device=device, L=L)
    model = StreamConvNet(net_input)

    # Create test input tensors
    # Note: Input shape for `StreamConvNet` should be (batch_size, depth, height, width)
    batch_size = 4  # Arbitrary batch size
    spatial_input = torch.randn(batch_size, rgb, spatial_size, spatial_size, device=device, dtype=torch.float32)
    temporal_input = torch.randn(batch_size, 2 * L, spatial_size, spatial_size, device=device, dtype=torch.float32)

    try:
        print("Testing spatial stream...")
        spatial_output = model.spatial.forward(spatial_input)
        print(f"Spatial stream output shape: {spatial_output.shape}")

        print("Testing temporal stream...")
        temporal_output = model.temporal.forward(temporal_input)
        print(f"Temporal stream output shape: {temporal_output.shape}")

        print("Testing combined forward pass...")
        combined_input = torch.cat([spatial_input, temporal_input], dim=1)  # Concatenate for testing
        combined_output = model.forward(combined_input)
        print(f"Combined output shape: {combined_output.shape}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()