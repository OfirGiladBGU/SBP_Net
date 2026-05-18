import argparse
import torch
import torch.nn as nn


#################
# Final Version #
#################

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

# Creates Large Noise around thin connections

class Network2D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Network2D, self).__init__()

        self.model_name = 'ae_2d_to_2d'
        self.input_size = args.input_size

        # Encoder with larger kernels
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),  # (batch_size, 64, H, W)
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),  # (batch_size, 64, H/2, W/2)
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (batch_size, 128, H/4, W/4)
            nn.ReLU(True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (batch_size, 256, H/8, W/8)
            nn.ReLU(True)
        )

        # Self-Attention after the encoder
        self.attention = SelfAttention(in_channels=256)

        # Decoder for reconstruction
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2, output_padding=0),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        # Encoding
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # Attention block
        x5 = self.attention(x4)

        # Decoding for reconstruction with skip connections
        recon = self.decoder1(x5)
        recon = self.decoder2(recon + x3)  # Skip connection
        recon = self.decoder3(recon + x2)  # Skip connection
        recon = self.decoder4(recon + x1)  # Skip connection

        return recon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_size = (1, 32, 32)
    model = Network2D(args)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = torch.rand(size=[1, *args.input_size]).to(device)
    output_data = model(input_data)
    print(output_data.shape)


##########
# Test 1 #
##########

# class Network2D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network2D, self).__init__()
#
#         self.model_name = 'ae_2d_to_2d'
#         self.input_size = args.input_size
#
#         # Encoder
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, H/2, W/2)
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
#             nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 1, H, W)
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

# # Self-Attention Block
# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         batch_size, C, width, height = x.size()
#         query = self.query(x).view(batch_size, -1, width * height)  # B x C/8 x N
#         key = self.key(x).view(batch_size, -1, width * height)      # B x C/8 x N
#         value = self.value(x).view(batch_size, -1, width * height)  # B x C x N
#
#         attention = torch.bmm(query.permute(0, 2, 1), key)  # B x N x N
#         attention = torch.softmax(attention, dim=-1)
#
#         out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
#         out = out.view(batch_size, C, width, height)
#
#         out = self.gamma * out + x
#         return out
#
# class Network2D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network2D, self).__init__()
#
#         self.model_name = 'ae_2d_to_2d'
#         self.input_size = args.input_size
#
#         # Encoder with larger kernels
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),  # (batch_size, 64, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # (batch_size, 128, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.encoder3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # (batch_size, 256, H/8, W/8)
#             nn.ReLU(True)
#         )
#
#         # Self-Attention after the encoder
#         self.attention = SelfAttention(256)
#
#         # Decoder
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.Sigmoid()  # Normalize output to [0, 1]
#         )
#
#     def forward(self, x):
#         # Encoding
#         x1 = self.encoder1(x)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#
#         # Attention block
#         x3 = self.attention(x3)
#
#         # Decoding with skip connections
#         x = self.decoder1(x3)
#         x = self.decoder2(x + x2)  # Skip connection
#         x = self.decoder3(x + x1)  # Skip connection
#
#         return x

##########
# Test 3 #
##########

# # Self-Attention Block
# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         batch_size, C, width, height = x.size()
#         query = self.query(x).view(batch_size, -1, width * height)  # B x C/8 x N
#         key = self.key(x).view(batch_size, -1, width * height)      # B x C/8 x N
#         value = self.value(x).view(batch_size, -1, width * height)  # B x C x N
#
#         attention = torch.bmm(query.permute(0, 2, 1), key)  # B x N x N
#         attention = torch.softmax(attention, dim=-1)
#
#         out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
#         out = out.view(batch_size, C, width, height)
#
#         out = self.gamma * out + x
#         return out
#
# class Network2D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network2D, self).__init__()
#
#         self.model_name = 'ae_2d_to_2d'
#         self.input_size = args.input_size
#         self.additional_tasks = [
#             "confidence map"
#         ]
#
#         # Encoder with larger kernels
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),  # (batch_size, 64, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # (batch_size, 128, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.encoder3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # (batch_size, 256, H/8, W/8)
#             nn.ReLU(True)
#         )
#
#         # Self-Attention after the encoder
#         self.attention = SelfAttention(256)
#
#         # Decoder
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.Sigmoid()  # Normalize output to [0, 1]
#         )
#
#         # Confidence layer with larger kernel size
#         self.confidence = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=5, padding=2),
#             nn.Sigmoid()  # Normalize output to [0, 1]
#         )
#
#     def forward(self, x):
#         # Encoding
#         x1 = self.encoder1(x)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#
#         # Attention block
#         x3 = self.attention(x3)
#
#         # Decoding with skip connections
#         x = self.decoder1(x3)
#         x = self.decoder2(x + x2)  # Skip connection
#         output_data = self.decoder3(x + x1)  # Skip connection
#
#         # Confidence map generation
#         output_confidence_data = self.confidence(output_data)
#
#         return output_data, output_confidence_data

##########
# Test 4 #
##########

# # Self-Attention Block
# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
#         self.key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
#         self.value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     # def forward(self, x):
#     #     batch_size, C, width, height = x.size()
#     #     query = self.query(x).view(batch_size, -1, width * height)  # B x C/8 x N
#     #     key = self.key(x).view(batch_size, -1, width * height)      # B x C/8 x N
#     #     value = self.value(x).view(batch_size, -1, width * height)  # B x C x N
#     #
#     #     attention = torch.bmm(query.permute(0, 2, 1), key)  # B x N x N
#     #     attention = torch.softmax(attention, dim=-1)
#     #
#     #     out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
#     #     out = out.view(batch_size, C, width, height)
#     #
#     #     out = self.gamma * out + x
#     #     return out
#
#     def forward(self, x):
#         batch_size, C, H, W = x.shape  # Extract batch, channels, height, width
#
#         # Compute Query, Key, and Value
#         query = self.query(x).view(batch_size, -1, H * W)  # (B, C/8, N)
#         key = self.key(x).view(batch_size, -1, H * W)  # (B, C/8, N)
#         value = self.value(x).view(batch_size, -1, H * W)  # (B, C, N)
#
#         # Compute Attention Map (Correct order)
#         attention = torch.bmm(query.permute(0, 2, 1), key)  # (B, N, N) spatial attention
#         attention = torch.softmax(attention, dim=-1)
#
#         # Apply attention to value
#         out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, N)
#
#         # Reshape back to (B, C, H, W)
#         out = out.view(batch_size, C, H, W)
#
#         # Residual Connection
#         out = self.gamma * out + x
#
#         return out


# class Network2D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network2D, self).__init__()
#
#         self.model_name = 'ae_2d_to_2d'
#         self.input_size = args.input_size
#         # self.additional_tasks = [
#         #     "confidence map"
#         # ]
#
#         # Encoder with larger kernels
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),  # (batch_size, 64, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # (batch_size, 128, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.encoder3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # (batch_size, 256, H/8, W/8)
#             nn.ReLU(True)
#         )
#
#         # Self-Attention after the encoder
#         self.attention = SelfAttention(256)
#
#         # Decoder for reconstruction
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.Sigmoid()  # Normalize output to [0, 1]
#         )
#
#         # Decoder for confidence map
#         self.confidence_decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.confidence_decoder2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.confidence_decoder3 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.Sigmoid()  # Confidence map, values between 0 and 1
#         )
#
#         # # Skip connection layers to align encoder features with decoder inputs
#         # self.skip_conv1 = nn.Conv2d(64, 64, kernel_size=1)
#         # self.skip_conv2 = nn.Conv2d(128, 128, kernel_size=1)
#
#     def forward(self, x):
#         # Encoding
#         x1 = self.encoder1(x)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#
#         # Attention block
#         x3 = self.attention(x3)
#
#         # Decoding for reconstruction with skip connections
#         recon = self.decoder1(x3)
#         recon = self.decoder2(recon + x2)  # Skip connection
#         recon = self.decoder3(recon + x1)  # Skip connection
#
#         # # Decoding for confidence map
#         # confidence = self.confidence_decoder1(x3)
#         # confidence = self.confidence_decoder2(confidence + x2)  # Skip connection
#         # confidence = self.confidence_decoder3(confidence + x1)  # Skip connection
#
#         # return recon, confidence
#
#         # # Decoder with skip connections
#         # d1 = self.decoder1(x3)  # Output: (B, 128, H/4, W/4)
#         # d1 = d1 + self.skip_conv2(x2)  # Skip connection from encoder2
#         #
#         # d2 = self.decoder2(d1)  # Output: (B, 64, H/2, W/2)
#         # d2 = d2 + self.skip_conv1(x1)  # Skip connection from encoder1
#         #
#         # recon = self.decoder3(d2)  # Output: (B, 1, H, W)
#         return recon

##########
# Test 5 #
##########
#
# import torch.nn.functional as F
#
# class Network2D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network2D, self).__init__()
#
#         self.model_name = 'ae_2d_to_2d'
#         self.input_size = args.input_size
#
#         # Encoder
#         self.enc1 = self.conv_block(1, 64)
#         self.enc2 = self.conv_block(64, 128)
#         self.enc3 = self.conv_block(128, 256)
#
#         # Bottleneck
#         self.bottleneck = self.conv_block(256, 512)
#
#         # Decoder
#         self.dec3 = self.conv_block(512 + 256, 256)
#         self.dec2 = self.conv_block(256 + 128, 128)
#         self.dec1 = self.conv_block(128 + 64, 64)
#
#         # Final output
#         self.final = nn.Conv2d(64, 1, kernel_size=1)
#
#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         # Encoder
#         c1 = self.enc1(x)
#         p1 = F.max_pool2d(c1, kernel_size=2)
#         c2 = self.enc2(p1)
#         p2 = F.max_pool2d(c2, kernel_size=2)
#         c3 = self.enc3(p2)
#         p3 = F.max_pool2d(c3, kernel_size=2)
#
#         # Bottleneck
#         bn = self.bottleneck(p3)
#
#         # Decoder
#         u3 = F.interpolate(bn, scale_factor=2, mode='bilinear', align_corners=True)
#         u3 = torch.cat([u3, c3], dim=1)
#         d3 = self.dec3(u3)
#
#         u2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
#         u2 = torch.cat([u2, c2], dim=1)
#         d2 = self.dec2(u2)
#
#         u1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
#         u1 = torch.cat([u1, c1], dim=1)
#         d1 = self.dec1(u1)
#
#         # Final output
#         output = self.final(d1)
#         return torch.sigmoid(output)

##########
# Test 6 #
##########

# class Network2D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network2D, self).__init__()
#
#         self.model_name = 'ae_2d_to_2d'
#         self.input_size = args.input_size
#
#         # Encoder with larger kernels
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),  # (batch_size, 64, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # (batch_size, 128, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.encoder3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (batch_size, 256, H/8, W/8)
#             nn.ReLU(True)
#         )
#
#         # Self-Attention after the encoder
#         self.attention = SelfAttention(256)
#
#         # Decoder for reconstruction
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, output_padding=0),
#             nn.ReLU(True)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(True)
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.Sigmoid()  # Normalize output to [0, 1]
#         )
#
#     def forward(self, x):
#         # Encoding
#         x1 = self.encoder1(x)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#
#         # Attention block
#         x3 = self.attention(x3)
#
#         # Decoding for reconstruction with skip connections
#         recon = self.decoder1(x3)
#         recon = self.decoder2(recon + x2)  # Skip connection
#         recon = self.decoder3(recon + x1)  # Skip connection
#         return recon
