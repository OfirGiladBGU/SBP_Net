import argparse

from configs.configs_parser import ModelType
from main_base import run_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to run 2D models')
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
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='Which dataset to use')
    # parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
    #                     help='Which weights to use')  # Moved to YAML config
    parser.add_argument('--model', type=str, default='ae_2d_to_2d', metavar='N',
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

    args.model = 'ae_2d_to_2d'
    # args.dataset = 'MNIST'
    # args.dataset = 'CIFAR10'
    # args.dataset = 'Trees2DV1S'
    args.dataset = 'Trees2DV1'
    args.epochs = 20

    # args.model = 'ae_6_2d_to_6_2d'
    # args.dataset = 'Trees2DV2'
    # args.epochs = 20

    run_main(args=args, model_type=ModelType.Model_2D)

    # Notes:

    # TODO: check option that the original input is kept and only the holes are predicted and merged - V
    # TODO: filter noise
    # TODO: check option to add confidence/classification head for the model to threshold the pixels - V
    # TODO: remove punishment on input white area

    # TODO: In pipes dataset - since the pipes are circular, holes in the pipes are not necessarily holes in the image (because the other side of the pipe might be visible)