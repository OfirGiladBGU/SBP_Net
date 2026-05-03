import argparse

from configs.configs_parser import ModelType
from main_base import run_main


################
# Custom Edit: #
################

# debug configs #
# MODEL = 'vit_2d_to_1d'
MODEL = 'cnn_2d_to_1d'
BATCH_SIZE = 128
DATASET = 'Trees1DV1'
EPOCHS = 10


# Paper config #
# NOT USED in Ours


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to run 1D models')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                        help='number of epochs to train (default: 10)')
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
    run_main(args=args, model_type=ModelType.Model_1D)
