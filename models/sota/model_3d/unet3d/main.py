import argparse
import torch
import torch.nn as nn
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent))

from model import UNet3D


# See: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/model.py
class Network3D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Network3D, self).__init__()

        self.model_name = 'unet3d'
        self.input_size = args.input_size

        self.model = UNet3D(
            in_channels=1,
            out_channels=1
        )

    def forward(self, x):
        output = self.model(x)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_size = (1, 32, 64, 64)
    model = Network3D(args)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = torch.rand(size=[1, *args.input_size]).to(device)
    output_data = model(input_data)
    print(output_data.shape)

# Loss: BCEDiceLoss
# Optimizer: Adam:
#   - learning_rate: 0.0002
#   - weight_decay: 0.00001
# Max Epochs: 1000
