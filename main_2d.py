import argparse

from configs.configs_parser import ModelType
from main_base import run_main


################
# Custom Edit: #
################

# debug configs #
# MODEL = 'ae_2d_to_2d'
# BATCH_SIZE = 128
# # DATASET = 'MNIST'  # (Sanity Check)
# # DATASET = 'CIFAR10'  # (Sanity Check)
# # DATASET = 'Trees2DV1S'  # (Sanity Check)
# DATASET = 'Trees2DV1'
# EPOCHS = 20

# MODEL = 'ae_6_2d_to_6_2d'
# BATCH_SIZE = 128
# DATASET = 'Trees2DV2'
# EPOCHS = 20


# Paper config #
MODEL = 'ae_2d_to_2d'
BATCH_SIZE = 128
DATASET = 'Trees2DV1'
EPOCHS = 20


if __name__ == "__main__":
    # NOTE: SELF NOTES:

    # TODO: In pipes dataset - since the pipes are circular, holes in the pipes are not necessarily holes in the image (because the other side of the pipe might be visible)

    parser = argparse.ArgumentParser(description='Main function to run 2D models')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                        help='number of epochs to train (default: 20)')
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
    parser.add_argument('--max_batches_to_plot', type=int, default=20,
                        help='Perform model prediction')
    parser.add_argument('--use_weights', type=bool, default=False,
                        help='Use weights for training')

    args = parser.parse_args()
    run_main(args=args, model_type=ModelType.Model_2D)
