import argparse
import torch
import torch.nn as nn


# Self-Attention Block
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.shape  # Extract batch, channels, height, width

        # Compute Query, Key, and Value
        query = self.query(x).view(batch_size, -1, H * W)  # (B, C/8, N)
        key = self.key(x).view(batch_size, -1, H * W)  # (B, C/8, N)
        value = self.value(x).view(batch_size, -1, H * W)  # (B, C, N)

        # Compute Attention Map (Correct order)
        attention = torch.bmm(query.permute(0, 2, 1), key)  # (B, N, N) spatial attention
        attention = torch.softmax(attention, dim=-1)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, N)

        # Reshape back to (B, C, H, W)
        out = out.view(batch_size, C, H, W)

        # Residual Connection
        out = self.gamma * out + x

        return out


class Network2D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Network2D, self).__init__()

        self.model_name = 'ae_2d_to_2d'
        self.input_size = args.input_size
        self.additional_tasks = [
            "confidence map"
        ]

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

        # Self-Attention after the encoder
        self.attention = SelfAttention(256)

        # Decoder for reconstruction
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

        # Decoder for confidence map
        self.confidence_decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True)
        )
        self.confidence_decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True)
        )
        self.confidence_decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # Confidence map, values between 0 and 1
        )

    def forward(self, x):
        # Encoding
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Attention block
        x3 = self.attention(x3)

        # Decoding for reconstruction with skip connections
        recon = self.decoder1(x3)
        recon = self.decoder2(recon + x2)  # Skip connection
        recon = self.decoder3(recon + x1)  # Skip connection

        # Decoding for confidence map
        confidence = self.confidence_decoder1(x3)
        confidence = self.confidence_decoder2(confidence + x2)  # Skip connection
        confidence = self.confidence_decoder3(confidence + x1)  # Skip connection

        return recon, confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_size = (1, 32, 32)
    model = Network2D(args)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = torch.rand(size=[1, *args.input_size]).to(device)
    output_data, conf_data = model(input_data)
    print(output_data.shape)
    print(conf_data.shape)
