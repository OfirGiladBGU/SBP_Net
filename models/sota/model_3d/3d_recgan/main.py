import argparse
import torch
import torch.nn as nn
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent))

from main_3D_RecGAN import AutoencoderUNet3D, Discriminator3D


# See: https://github.com/Yang7879/3D-RecGAN/blob/master/main_3D-RecGAN.py
class Network3D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Network3D, self).__init__()

        self.model_name = '3d_recgan'
        self.input_size = args.input_size

        self.ae_u = AutoencoderUNet3D()
        self.dis = Discriminator3D()

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

# Loss:
#   - (AutoencoderUNet3D) BCEWithLogitsLoss
#   - (Discriminator) WGAN-GP
# Optimizer: Adam:
#   - (AutoencoderUNet3D) weight_decay: 0.0005
#   - (Discriminator) learning_rate: 0.0001
# Max Epochs: 15
