import argparse
import torch
import os
import datetime

from configs.configs_parser import *
from datasets.dataset_list import init_dataset
from models.model_list import init_model


def configure_wandb(args, model, init=True):
    import wandb
    if init:
        dot_env_path = os.path.join(ROOT_PATH, ".env")
        with open(dot_env_path, 'r') as f:
            for line in f:
                if line.startswith("WANDB_API_KEY"):
                    os.environ["WANDB_API_KEY"] = line.replace('WANDB_API_KEY=','').strip(" \"")
        API_KEY = os.environ.get("WANDB_API_KEY")
        wandb.login(key=API_KEY)

        wandb_project = "TreesAutoEncoder"
        init_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        wandb_name = f"{model.model_name}>{init_timestamp}"
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            tags=[args.dataset, DATASET_INPUT_FOLDER, f"{args.epochs} epochs", f"size {args.input_size}"]
        )
    else:
        # wandb.finish()
        pass


def run_main(args: argparse.Namespace, model_type: ModelType):
    # Set device and random seed
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    # Initialize dataset
    dataset = init_dataset(args=args)
    args.input_size = dataset.input_size

    # TODO: Requires dataset
    model = init_model(args=args)

    # Import relevant Trainer
    if model_type == ModelType.Model_2D:
        from trainer.train_2d import Trainer
        weights_filepath = WEIGHTS_2D_PATH
    elif model_type == ModelType.Model_3D:
        from trainer.train_3d import Trainer
        weights_filepath = WEIGHTS_3D_PATH
    else:
        raise ValueError(f"Model Type '{args.model_type}' is not supported.")

    # Set default weights filepath
    if weights_filepath is None:
        weights_name = f"Network_{DATASET_OUTPUT_FOLDER}_{model.model_name}.pth"
        weights_filepath = os.path.join(ROOT_PATH, "weights", weights_name)

    # Ensure the directory exists
    os.makedirs(name=os.path.dirname(weights_filepath), exist_ok=True)

    # Update args with paths information
    args.weights_filepath = weights_filepath
    args.results_path = os.path.join(MODELS_RESULTS_PATH, model.model_name)

    if args.wandb:
        configure_wandb(args=args, model=model, init=True)

    # Initialize Trainer
    trainer = Trainer(args=args, dataset=dataset, model=model)

    use_weights = args.use_weights
    if args.train:
        trainer.train(use_weights=use_weights)
    if args.predict:
        trainer.predict(max_batches_to_plot=args.max_batches_to_plot)

    if args.wandb:
        configure_wandb(args=args, model=model, init=False)
