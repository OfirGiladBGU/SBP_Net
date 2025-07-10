import argparse
import torch
import torch.nn as nn


class Network1D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.model_name = 'cnn_2d_to_1d'
        self.input_size = args.input_size  # (1, W, H)

        # Encoder with larger kernels
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),  # (batch_size, 64, H/2, W/2)
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # (batch_size, 128, H/4, W/4)
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # (batch_size, 256, H/8, W/8)
            nn.ReLU(True)
        )

        fc_in_features = 256 * (self.input_size[1] // 8) * (self.input_size[2] // 8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in_features, 1),  # Single neuron for binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Classification
        cls = self.fc(x3)
        return cls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_size = (1, 32, 32)
    model = Network1D(args)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = torch.rand(size=[1, *args.input_size]).to(device)
    output_data = model(input_data)
    print(output_data.shape)