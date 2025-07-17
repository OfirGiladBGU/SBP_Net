import argparse
import torch
import torch.nn as nn
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent))

from main_3D_RecGAN_pytorch import AutoencoderUNet3D, Discriminator3D


# See: https://github.com/Yang7879/3D-RecGAN/blob/master/main_3D-RecGAN.py
class Network3D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Network3D, self).__init__()

        self.model_name = 'recgan_3d'
        self.input_size = args.input_size

        self.resolution = 64
        self.ae_u = AutoencoderUNet3D()
        self.dis = Discriminator3D()

        self.device = None
        self.ae_u_optimizer = None
        self.dis_optimizer = None

    def forward(self, x):
        output, _ = self.ae_u(x)
        return output

    #################################
    # Util function for training 3d #
    #################################

    def init_optimizers(self):
        self.ae_u_optimizer = torch.optim.Adam(self.ae_u.parameters(), lr=0.0005, betas=(0.9, 0.999))
        self.dis_optimizer = torch.optim.Adam(self.dis.parameters(), lr=0.0001, betas=(0.9, 0.999))

    def model_step(self, x, y, train=True):
        if not self.ae_u_optimizer or not self.dis_optimizer:
            raise ValueError("Optimizers must be initialized before training.")
        if self.device is None:
            self.device = next(self.parameters()).device

        torch.autograd.set_detect_anomaly(True)

        if train:
            self.ae_u_optimizer.zero_grad()
            self.dis_optimizer.zero_grad()

        batch_size = x.size(0)

        # === Autoencoder forward ===
        y_pred, y_pred_mod = self.ae_u(x)

        # === Discriminator forward ===
        real_pair = self.dis(x, y)
        fake_pair = self.dis(x, y_pred)

        # === Autoencoder loss ===
        y_flat = y.view(batch_size, -1)
        y_pred_mod_flat = y_pred_mod.view(batch_size, -1)
        w = 0.85
        ae_loss = -torch.mean(
            w * y_flat * torch.log(y_pred_mod_flat + 1e-8) +
            (1 - w) * (1 - y_flat) * torch.log(1 - y_pred_mod_flat + 1e-8)
        )

        # === WGAN loss ===
        gan_g_loss = -fake_pair.mean()
        gan_d_loss_no_gp = fake_pair.mean() - real_pair.mean()

        alpha = torch.rand(batch_size, self.resolution ** 3).to(self.device)
        y_pred_flat = y_pred.view(batch_size, -1)
        differences = y_pred_flat - y_flat
        interpolates = y_flat + alpha * differences
        interpolates = interpolates.view(batch_size, 1, self.resolution, self.resolution, self.resolution)
        interpolates.requires_grad_(True)
        fake_interpolates = self.dis(x, interpolates)
        gradients = torch.autograd.grad(
            outputs=fake_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(fake_interpolates),
            create_graph=True, retain_graph=True
        )[0]
        slopes = gradients.view(gradients.size(0), -1).norm(2, dim=1)
        gradient_penalty = torch.mean((slopes - 1.0) ** 2)
        gan_d_loss_gp = gan_d_loss_no_gp + 10 * gradient_penalty

        # === Autoencoder + GAN loss ===
        gan_g_w = 5
        ae_w = 100 - gan_g_w
        ae_gan_g_loss = ae_w * ae_loss + gan_g_w * gan_g_loss

        if train:
            ae_gan_g_loss.backward(retain_graph=True)
            self.ae_u_optimizer.step()

            gan_d_loss_gp.backward()
            self.dis_optimizer.step()

        total_loss = ae_gan_g_loss.item() + gan_d_loss_gp.item()
        print(
            "> [3D RecGAN] ae loss: {}, gan g loss: {}, gan d loss no gp: {}, gand d loss gp: {}".format(
                ae_loss.item(),
                gan_g_loss.item(),
                gan_d_loss_no_gp.item(),
                gan_d_loss_gp.item()
            )
        )
        return total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_size = (1, 64, 64, 64)
    model = Network3D(args)
    model.ae_u.eval()
    model.dis.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.ae_u.to(device)
    model.dis.to(device)
    input_data = torch.rand(size=[1, *args.input_size]).to(device)
    output_data1, output_data1_m = model.ae_u(input_data)
    output_data2 = model.dis(input_data, output_data1)
    print(output_data1.shape)
    print(output_data1_m.shape)
    print(output_data2.shape)

# Loss:
#   - (AutoencoderUNet3D) BCEWithLogitsLoss
#   - (Discriminator) WGAN-GP
# Optimizer: Adam:
#   - (AutoencoderUNet3D) learning_rate: 0.0005
#   - (Discriminator) learning_rate: 0.0001
# Max Epochs: 15
