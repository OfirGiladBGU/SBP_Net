import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------
# Part 1: Tools replacement
# ---------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.act(self.conv(x)))

class DeconvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.deconv(x)


# ---------------------
# Part 2: Autoencoder U-Net
# ---------------------
class AutoencoderUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([
            ConvBlock3D(1, 64),
            ConvBlock3D(64, 128),
            ConvBlock3D(128, 256),
            ConvBlock3D(256, 512)
        ])

        self.fc1 = nn.Linear(32768, 5000)
        self.fc2 = nn.Linear(5000, 32768)

        self.dec = nn.ModuleList([
            DeconvBlock3D(512 + 512, 256),
            DeconvBlock3D(256 + 256, 128),
            DeconvBlock3D(128 + 128, 64),
            DeconvBlock3D(64 + 64, 1)
        ])

    def forward(self, x):
        skips = []
        out = x
        for enc_layer in self.enc:
            out = enc_layer(out)
            skips.append(out)
        skips = skips[::-1]

        flat = out.view(out.size(0), -1)
        bottleneck = F.relu(self.fc1(flat))
        out = F.relu(self.fc2(bottleneck)).view(-1, 512, 4, 4, 4)

        for i, dec_layer in enumerate(self.dec):
            inp = torch.cat([out, skips[i]], dim=1)
            out = dec_layer(inp)
            if i < len(self.dec) - 1:
                out = F.relu(out, inplace=False)

        vox_sig = torch.sigmoid(out)
        vox_sig_modified = torch.clamp(vox_sig, min=0.01)
        return vox_sig, vox_sig_modified


# ---------------------
# Part 3: Discriminator
# ---------------------
class Discriminator3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
        ])
        self.activations = nn.ModuleList([
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        ])

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)  # Concatenate on channel axis
        for conv, act in zip(self.layers, self.activations):
            out = act(conv(out))
        return out.view(out.size(0), -1)  # Reshape to match expected output shape


# ---------------------
# Part 4: Forward Example
# ---------------------
if __name__ == '__main__':
    batch_size = 8
    resolution = 64
    ae_u = AutoencoderUNet3D()
    dummy_input = torch.rand(batch_size, 1, resolution, resolution, resolution)
    y_pred, y_pred_mod = ae_u(dummy_input)
    print('y_pred shape:', y_pred.shape)
    print('y_pred_mod shape:', y_pred_mod.shape)

    # Test discriminator
    dis = Discriminator3D()
    dis_out = dis(dummy_input, y_pred)
    print('discriminator output shape:', dis_out.shape)
