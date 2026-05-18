import argparse

from configs.configs_parser import ModelType
from main_base import run_main


################
# Custom Edit: #
################

# debug configs #
# MODEL = 'ae_6_2d_to_3d'
# BATCH_SIZE = 16
# DATASET = 'Trees3DV1'
# EPOCHS = 10

# MODEL = 'ae_3d_to_3d'
# BATCH_SIZE = 16
# # DATASET = 'Trees3DV2'
# # DATASET = 'Trees3DV2D'
# MODEL = 'Trees3DV2F'
# EPOCHS = 10


# Paper config #
# NOT USED in Ours


# 3D RecGAN config #
MODEL = 'recgan_3d'
BATCH_SIZE = 8
DATASET = 'Trees3DV2D'
EPOCHS = 15


# UNet3D config #
# MODEL = 'unet3d'
# # BATCH_SIZE = 1  # Full Input (Requires strong GPU)
# # DATASET = 'Trees3DV3'  # Full Input (Requires strong GPU)
# BATCH_SIZE = 2
# DATASET = 'Trees3DV2D'
# EPOCHS = 50


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to run 3D models')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default=DATASET, metavar='N',
                        help='Which dataset to use')
    # parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
    #                     help='Which weights to use')  # Moved to YAML config
    parser.add_argument('--model', type=str, default=MODEL, metavar='N',
                        help='Which model to use')
    parser.add_argument('--wandb', type=bool, default=True,
                        help='Connect to Weights & Biases')
    parser.add_argument('--train', type=bool, default=True,
                        help='Perform model training')
    parser.add_argument('--predict', type=bool, default=True,
                        help='Perform model prediction')
    parser.add_argument('--max_batches_to_plot', type=int, default=2,
                        help='Perform model prediction')
    parser.add_argument('--use_weights', type=bool, default=False,
                        help='Use weights for training')

    args = parser.parse_args()
    run_main(args=args, model_type=ModelType.Model_3D)
