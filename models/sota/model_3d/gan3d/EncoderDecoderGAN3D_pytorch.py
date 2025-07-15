from __future__ import print_function, division

import os
from mpl_toolkits.mplot3d import Axes3D  # you should keep the import
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import hamming_loss
import copy


class Generator3D(nn.Module):
    def __init__(self, in_channels=1):
        super(Generator3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(32, momentum=0.8),

            nn.Conv3d(32, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(64, momentum=0.8),

            nn.Conv3d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(128, momentum=0.8),

            nn.Conv3d(128, 512, 1, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.5)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose3d(512, 256, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm3d(256, momentum=0.8),

            nn.ConvTranspose3d(256, 128, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm3d(128, momentum=0.8),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose3d(128, 64, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm3d(64, momentum=0.8),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose3d(64, in_channels, 5, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator3D(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator3D, self).__init__()
        self.model = nn.Sequential(
            # 1 x 16 x 16 x 16
            nn.Conv3d(in_channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(64, momentum=0.8),

            # 1 x 64 x 8 x 8 x 8
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(128, momentum=0.8),

            # 1 x 128 x 4 x 4 x 4
            nn.Conv3d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(256, momentum=0.8),

            # 1 x 256 x 4 x 4 x 4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class EncoderDecoderGAN:
    def __init__(self):
        self.vol_rows = 32
        self.vol_cols = 32
        self.vol_height = 32
        self.mask_height = 16
        self.mask_width = 16
        self.mask_length = 16

        self.channels = 1
        self.num_classes = 2

        self.generator = Generator3D(in_channels=self.channels)
        self.discriminator = Discriminator3D(in_channels=self.channels)

    def generateWall(self):
        x, y, z = np.indices((32, 32, 32))
        voxel = (x < 28) & (x > 5) & (y > 5) & (y < 28) & (z > 10) & (z < 25)
        # add channel
        voxel = voxel[..., np.newaxis].astype(np.float16)
        # repeat 1000 times
        voxels = list()
        for i in range(1000):
            voxels.append(voxel)
        voxels = np.asarray(voxels)
        return voxels

    def mask_randomly(self, vols):
        y1 = np.random.randint(0, self.vol_rows - self.mask_height, vols.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.vol_cols - self.mask_width, vols.shape[0])
        x2 = x1 + self.mask_width
        z1 = np.random.randint(0, self.vol_height - self.mask_length, vols.shape[0])
        z2 = z1 + self.mask_length

        masked_vols = np.empty_like(vols)
        missing_parts = np.empty((vols.shape[0], self.mask_height, self.mask_width, self.mask_length, self.channels))
        for i, vol in enumerate(vols):
            masked_vol = vol.copy()
            _y1, _y2, _x1, _x2, _z1, _z2 = y1[i], y2[i], x1[i], x2[i], z1[i], z2[i]
            missing_parts[i] = masked_vol[_y1:_y2, _x1:_x2, _z1:_z2, :].copy()
            masked_vol[_y1:_y2, _x1:_x2, _z1:_z2, :] = 0
            masked_vols[i] = masked_vol

        return masked_vols, missing_parts, (y1, y2, x1, x2, z1, z2)

    def train(self, epochs, batch_size=16, sample_interval=50):
        # Preprocess
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        if os.path.exists(os.path.join(MODEL_DIR, 'generator.h5')):
            self.generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'generator.h5')))
            print("Loaded generator model")
        if os.path.exists(os.path.join(MODEL_DIR, 'discriminator.h5')):
            self.discriminator.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'discriminator.h5')))
            print("Loaded discriminator model")

        generator_optimizer = Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        weight_mse = 0.999
        weight_bce = 0.001
        loss_fn_mse = nn.MSELoss()
        loss_fn_bce = nn.BCELoss()

        X_train = self.generateWall()
        # # Adversarial ground truths
        # valid = np.ones((batch_size, 1))
        # fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            vols = X_train[idx]
            masked_vols, missing_parts, _ = self.mask_randomly(vols)

            # Adversarial ground truths
            masked_vols_tensor = torch.tensor(masked_vols.transpose(0, 4, 1, 2, 3), dtype=torch.float32).to(self.device)
            missing_parts_tensor = torch.tensor(missing_parts.transpose(0, 4, 1, 2, 3), dtype=torch.float32).to(self.device)
            valid_tensor = torch.ones((batch_size, 1), dtype=torch.float32).to(self.device)
            fake_tensor = torch.zeros((batch_size, 1), dtype=torch.float32).to(self.device)

            ### --- Train Discriminator ---
            self.discriminator.zero_grad()

            real_pred = self.discriminator(missing_parts_tensor)
            fake_missing = self.generator(masked_vols_tensor).detach()
            fake_pred = self.discriminator(fake_missing)

            loss_real = loss_fn_bce(real_pred, valid_tensor)
            loss_fake = loss_fn_bce(fake_pred, fake_tensor)
            d_loss = 0.5 * (loss_real + loss_fake)

            d_loss.backward()
            discriminator_optimizer.step()

            ### --- Train Generator ---
            self.generator.zero_grad()

            gen_missing = self.generator(masked_vols_tensor)
            validity = self.discriminator(gen_missing)

            g_loss_mse = loss_fn_mse(gen_missing, missing_parts_tensor)
            g_loss_bce = loss_fn_bce(validity, valid_tensor)
            g_loss = weight_mse * g_loss_mse + weight_bce * g_loss_bce

            g_loss.backward()
            generator_optimizer.step()

            print(
                f"{epoch} "
                f"[D loss: {d_loss.item():.6f}] "
                f"[G loss: {g_loss.item():.6f}, mse: {g_loss_mse.item():.6f}]"
            )

            # save generated samples
            if epoch % sample_interval == 0:
                idx = np.random.randint(0, X_train.shape[0], 2)
                vols = X_train[idx]
                self.sample_images(epoch, vols)
                self.save_model()

    def sample_images(self, epoch, vols):
        r, c = 2, 2

        masked_vols, missing_parts, (y1, y2, x1, x2, z1, z2) = self.mask_randomly(vols)
        # gen_missing = self.generator.predict(masked_vols)

        self.generator.eval()
        with torch.no_grad():
            masked_tensor = torch.tensor(masked_vols.transpose(0, 4, 1, 2, 3), dtype=torch.float32).to(self.device)
            gen_missing_tensor = self.generator(masked_tensor).cpu().numpy()
        gen_missing = np.where(gen_missing_tensor.transpose(0, 2, 3, 4, 1) > 0.5, 1, 0)
        self.generator.train()

        gen_missing = np.where(gen_missing > 0.5, 1, 0)
        # fig = plt.figure(figsize=plt.figaspect(0.5), dpi=300)

        vols = 0.5 * vols + 0.5

        for i in range(2):
            fig = plt.figure(figsize=plt.figaspect(0.5), dpi=300)
            masked_vol = masked_vols[i]
            masked_vol = masked_vol[:, :, :, 0].astype(np.bool_)
            colors1 = np.empty(masked_vol.shape, dtype=object)
            colors1[masked_vol] = 'red'
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.voxels(masked_vol, facecolors=colors1, edgecolor='black', linewidth=0.2)

            filled_in = np.zeros_like(masked_vol)
            # filled_in = vols[i].copy()
            one_gen_missing = gen_missing[i]
            one_gen_missing = one_gen_missing[:, :, :, 0].astype(np.bool_)

            # Compute hamming loss
            true_missing_part = missing_parts[i]
            true_missing_part = true_missing_part[:, :, :, 0].astype(np.bool_)
            ham_loss = hamming_loss(true_missing_part.ravel(), one_gen_missing.ravel())

            filled_in[y1[i]:y2[i], x1[i]:x2[i], z1[i]:z2[i]] = one_gen_missing
            fill = filled_in
            combine_voxels = masked_vol | fill

            colors2 = np.empty(combine_voxels.shape, dtype=object)
            colors2[masked_vol] = 'red'
            colors2[fill] = 'blue'

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.voxels(combine_voxels, facecolors=colors2, edgecolor='black', linewidth=0.2)
            # ax.voxels(masked_vol, facecolors=colors1, edgecolor='k')
            ax.set_title("Hamming Loss: %f" % ham_loss)
            # plt.show()
            fig.savefig(os.path.join(IMAGE_DIR, "%d_%d.png" % (epoch, i)))
            print("saved sample images")
            plt.close()

    def save_model(self):
        def save(model, model_name):
            model_path = os.path.join(MODEL_DIR, "%s.h5" % model_name)
            # model.save(model_path)
            model_parameters = copy.deepcopy(model.state_dict())
            torch.save(model_parameters, model_path)

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    IMAGE_DIR = './32_cube/images'
    MODEL_DIR = './32_cube/saved_model'

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    context_encoder = EncoderDecoderGAN()
    context_encoder.train(epochs=3000, batch_size=5, sample_interval=200)
