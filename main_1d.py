import argparse

from configs.configs_parser import ModelType
from main_base import run_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to run 1D models')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                        help='embedding size for the model')
    parser.add_argument('--dataset', type=str, default='Trees1DV1', metavar='N',
                        help='Which dataset to use')
    # parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
    #                     help='Which weights to use')  # Moved to YAML config
    parser.add_argument('--model', type=str, default='vit_2d_to_1d', metavar='N',
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

    # Custom Edit:

    # args.model = 'vit_2d_to_1d'
    args.model = 'cnn_2d_to_1d'
    args.dataset = 'Trees1DV1'
    args.epochs = 10

    run_main(args=args, model_type=ModelType.Model_1D)
