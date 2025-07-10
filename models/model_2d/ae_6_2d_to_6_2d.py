import argparse
import torch.nn as nn
import torch
# import torch.nn.functional as F


##########
# Test 1 #
##########

# class Network2D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network2D, self).__init__()
#
#         self.model_name = 'ae_6_2d_to_6_2d'
#         self.input_size = args.input_size
#
#         # Encoder
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.encoder3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (batch_size, 256, H/8, W/8)
#             nn.ReLU(True)
#         )
#
#         # Decoder
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             # (batch_size, 128, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             # (batch_size, 64, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose2d(64, 6, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 6, H, W)
#             nn.Sigmoid()  # To normalize the output to [0, 1]
#         )
#
#     def forward(self, x):
#         z = x
#
#         # Encoding
#         z1 = self.encoder1(z)
#         z2 = self.encoder2(z1)
#         z3 = self.encoder3(z2)
#
#         # Decoding with skip connections
#         z = self.decoder1(z3)
#         z = self.decoder2(z + z2)  # Skip connection
#         z = self.decoder3(z + z1)  # Skip connection
#
#         return z

##########
# Test 2 #
##########

# class Network2D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network2D, self).__init__()
#
#         self.model_name = 'ae_6_2d_to_6_2d'
#         self.input_size = args.input_size
#
#         # Encoder
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.encoder3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (batch_size, 256, H/8, W/8)
#             nn.ReLU(True)
#         )
#
#         # Decoder with concatenation for skip connections
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128 + 128)
#             nn.ReLU(True)
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose2d(128, 6, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64 + 64)
#             nn.Sigmoid()  # Normalize to [0, 1]
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(self.input_size[0] * self.input_size[1] * self.input_size[2], self.input_size[0] * self.input_size[1] * self.input_size[2])
#         )
#
#     def forward(self, x):
#         # Encoding
#         z1 = self.encoder1(x)
#         z2 = self.encoder2(z1)
#         z3 = self.encoder3(z2)
#
#         # Decoding with concatenation skip connections
#         z = self.decoder1(z3)
#         z = self.decoder2(torch.cat((z, z2), dim=1))  # Concatenation instead of addition
#         z = self.decoder3(torch.cat((z, z1), dim=1))  # Concatenation instead of addition
#
#         # Residual learning to predict only the hole areas
#         reshaped_x = x.view(x.size(0), -1)
#         z = z + self.fc(reshaped_x).view(z.size())
#         return z


# # Masked loss function: focuses on areas with holes
# def masked_loss(output, target, mask):
#     # Mask should have 1s for missing areas (holes) and 0s for intact areas
#     return (mask * F.mse_loss(output, target, reduction='none')).mean()
#
# # Reconstruction loss (L1 loss for sharpness)
# def reconstruction_loss(output, target):
#     return F.l1_loss(output, target)

# Example usage with argument parsing
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_size', type=int, default=256)
#     args = parser.parse_args()
#
#     # Initialize the model
#     model = Network2D(args)
#
#     # Example input
#     input_data = torch.randn(1, 6, 256, 256)  # Batch of 1, 6-channel, 256x256 image
#     target_data = torch.randn(1, 6, 256, 256)  # Target completion
#     hole_mask = torch.randint(0, 2, (1, 6, 256, 256))  # Binary mask indicating holes
#
#     # Forward pass
#     output = model(input_data)
#
#     # Compute loss
#     loss = reconstruction_loss(output, target_data) + masked_loss(output, target_data, hole_mask)
#     print(f"Loss: {loss.item()}")

##########
# Test 3 #
##########

# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(6, 64, 4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1),
#             nn.ReLU(),
#         )
#
#     def forward(self, x):
#         return self.encoder(x)
#
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 6, 4, stride=2, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.decoder(x)
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.discriminator = nn.Sequential(
#             nn.Conv2d(6, 64, 4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 1, 4, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.discriminator(x)
#
#
# class Network2D(nn.Module):
#     def __init__(self, args):
#         super(Network2D, self).__init__()
#
#         self.model_name = 'ae_6_2d_to_6_2d'
#         self.input_size = args.input_size
#
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#
#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)

##########
# Test 4 #
##########

# TODO: check option that the original input is kept and only the holes are predicted and merged

# DIPNet
class Network2D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Network2D, self).__init__()

        self.model_name = 'ae_6_2d_to_6_2d'
        self.input_size = args.input_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 6, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        return x3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_size = (6, 32, 32)
    model = Network2D(args)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = torch.rand(size=[1, *args.input_size]).to(device)
    output_data = model(input_data)
    print(output_data.shape)
