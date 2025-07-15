import argparse

from configs.configs_parser import ModelType
from main_base import run_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main function to run 3D models')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='Trees3DV2', metavar='N',
                        help='Which dataset to use')
    # parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
    #                     help='Which weights to use')  # Moved to YAML config
    parser.add_argument('--model', type=str, default='ae_3d_to_3d', metavar='N',
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

    # Custom Edit:

    # args.model = 'ae_6_2d_to_3d'
    # args.dataset = 'Trees3DV1'
    # args.epochs = 10

    # args.model = 'ae_3d_to_3d'
    # # args.dataset = 'Trees3DV2'
    # # args.dataset = 'Trees3DV2D'
    # args.dataset = 'Trees3DV2F'
    # args.epochs = 10

    args.model = 'unet3d'
    # args.batch_size = 1
    # args.dataset = 'Trees3DV3'
    args.batch_size = 2
    args.dataset = 'Trees3DV2D'
    args.epochs = 50
    args.predict = True  # Skip predict for visualization

    run_main(args=args, model_type=ModelType.Model_3D)
