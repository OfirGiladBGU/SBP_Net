import argparse
import torch
import torch.nn as nn
# import torch.nn.functional as F


##########
# Test 1 #
##########

# class Network3D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network3D, self).__init__()
#
#         self.model_name = 'ae_3d_to_3d'
#         self.input_size = args.input_size
#
#         # Encoder
#         self.encoder1 = nn.Sequential(
#             nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, D/2, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, D/4, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.encoder3 = nn.Sequential(
#             nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # (batch_size, 256, D/8, H/8, W/8)
#             nn.ReLU(True)
#         )
#
#         # Decoder
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             # (batch_size, 128, D/4, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             # (batch_size, 64, D/2, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 1, D, H, W)
#             nn.Sigmoid()  # To normalize the output to [0, 1]
#         )
#
#     def forward(self, x):
#         # Encoding
#         x1 = self.encoder1(x)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#
#         # Decoding with skip connections
#         x = self.decoder1(x3)
#         x = self.decoder2(x + x2)  # Skip connection
#         x = self.decoder3(x + x1)  # Skip connection
#
#         return x

##########
# Test 2 #
##########

# class SelfAttention3D(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention3D, self).__init__()
#         self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
#         self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
#         self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         batch_size, C, D, H, W = x.size()
#
#         # Compute query, key, and value
#         query = self.query(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)  # (batch_size, D*H*W, C//8)
#         key = self.key(x).view(batch_size, -1, D * H * W)  # (batch_size, C//8, D*H*W)
#         value = self.value(x).view(batch_size, -1, D * H * W)  # (batch_size, C, D*H*W)
#
#         # Compute attention map
#         attention = torch.softmax(torch.bmm(query, key), dim=-1)  # (batch_size, D*H*W, D*H*W)
#
#         # Apply attention to value
#         out = torch.bmm(value, attention.permute(0, 2, 1))  # (batch_size, C, D*H*W)
#         out = out.view(batch_size, C, D, H, W)
#
#         # Add residual connection
#         out = self.gamma * out + x
#
#         return out


# class Network3D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network3D, self).__init__()
#
#         self.model_name = 'ae_3d_to_3d'
#         self.input_size = args.input_size
#
#         # Encoder
#         self.encoder1 = nn.Sequential(
#             nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=2),  # (batch_size, 64, D/2, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=2),  # (batch_size, 128, D/4, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.encoder3 = nn.Sequential(
#             nn.Conv3d(128, 256, kernel_size=5, stride=2, padding=2),  # (batch_size, 256, D/8, H/8, W/8)
#             nn.ReLU(True)
#         )
#
#         # Self-Attention after the encoder
#         # self.self_attention = SelfAttention3D(256)
#
#         # Decoder
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose3d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose3d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose3d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.Sigmoid()  # Normalize output to [0, 1]
#         )
#
#     def forward(self, x):
#         # Encoding
#         x1 = self.encoder1(x)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#
#         # Apply self-attention
#         # x3 = self.self_attention(x3)
#
#         # Decoding with skip connections
#         recon = self.decoder1(x3)
#         recon = self.decoder2(recon + x2)  # Skip connection
#         recon = self.decoder3(recon + x1)  # Skip connection
#
#         return recon

##########
# Test 3 #
##########

# Creates Large Noise around thin connections

# class Network3D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network3D, self).__init__()
#
#         self.model_name = 'ae_3d_to_3d'
#         self.input_size = args.input_size
#
#         # Encoder with larger kernels
#         self.encoder1 = nn.Sequential(
#             nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),  # (batch_size, 64, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (batch_size, 128, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.encoder3 = nn.Sequential(
#             nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (batch_size, 256, H/8, W/8)
#             nn.ReLU(True)
#         )
#
#         # Self-Attention after the encoder
#         # self.attention = SelfAttention3D(256)
#
#         # Decoder for reconstruction
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0),
#             nn.ReLU(True)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0),
#             nn.ReLU(True)
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # Encoding
#         x1 = self.encoder1(x)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#
#         # Attention block
#         # x4 = self.attention(x3)
#
#         # Decoding for reconstruction with skip connections
#         recon = self.decoder1(x3)
#         recon = self.decoder2(recon + x2)  # Skip connection
#         recon = self.decoder3(recon + x1)  # Skip connection
#
#         return recon

##########
# Test 4 #
##########

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, D, H, W = x.size()

        # Compute query, key, and value
        query = self.query(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)  # (batch_size, D*H*W, C//8)
        key = self.key(x).view(batch_size, -1, D * H * W)  # (batch_size, C//8, D*H*W)
        value = self.value(x).view(batch_size, -1, D * H * W)  # (batch_size, C, D*H*W)

        # Compute attention map
        attention = torch.softmax(torch.bmm(query, key), dim=-1)  # (batch_size, D*H*W, D*H*W)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (batch_size, C, D*H*W)
        out = out.view(batch_size, C, D, H, W)

        # Add residual connection
        out = self.gamma * out + x

        return out

class Network3D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Network3D, self).__init__()

        self.model_name = 'ae_3d_to_3d'
        self.input_size = args.input_size

        # Single Encoder Layer
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2),  # Downsample
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # Optional: Latent bottleneck transformation
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Single Decoder Layer
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # For normalized outputs
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.bottleneck(encoded)
        decoded = self.decoder(latent)
        return decoded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_size = (1, 32, 32, 32)
    model = Network3D(args)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = torch.rand(size=[1, *args.input_size]).to(device)
    output_data = model(input_data)
    print(output_data.shape)
