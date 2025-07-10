import argparse
import torch
import torch.utils.data
from torch import optim
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm
import datetime
from matplotlib import gridspec

from configs.configs_parser import V1_2D_DATASETS, V2_2D_DATASETS, RANDOM_HOLES_DATASETS, IMAGES_6_VIEWS
from datasets.dataset_utils import apply_threshold
from trainer import loss_functions
from trainer import train_utils


class Trainer(object):
    def __init__(self, args: argparse.Namespace, dataset, model):
        self.args = args

        self.device = self.args.device
        self.dataset = dataset
        self.model = model
        self.model.to(self.device)

        self.start_time = datetime.datetime.now()

        # Get loaders
        self.train_loader = self.dataset.train_loader
        self.test_loader = self.dataset.test_loader

        self.datasets_for_holes = RANDOM_HOLES_DATASETS

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # self.optimizer = optim.Adadelta(self.model.parameters())  # For gap_cnn

    def loss_function(self, output_data, target_data, input_data=None):
        """
        :param output_data: model output on the 'original' input
        :param target_data: the target data that the model should output
        :param input_data: the original input data for the model
        :return:
        """
        # Handle additional_tasks
        if "confidence map" in getattr(self.model, "additional_tasks", list()):
            output_data, output_confidence_data = output_data
        else:
            output_confidence_data = None

        if self.model.model_name == "gap_cnn":
            LOSS = loss_functions.mse_loss(output=output_data, target=target_data)
            return LOSS
        elif self.model.model_name == "vgg_ae_demo":
            LOSS = loss_functions.perceptual_loss(output=output_data, target=target_data, channels=1, device=self.device)
            return LOSS

        ##########
        # Test 1 #
        ##########
        # LOSS = loss_functions.bce_loss(output=output_data, target=target_data, reduction='sum')

        ##########
        # Test 2 #
        ##########
        # LOSS = loss_functions.l1_loss(output=output_data, target=target_data, reduction='sum')

        ##########
        # Test 3 #
        ##########
        # LOSS = loss_functions.perceptual_loss(output=output_data, target=target_data, channels=1, device=self.args.device)

        ##########
        # Test 4 #
        ##########
        # lambda_value = 100.0
        # LOSS = loss_functions.weighted_mask_loss(output=output_data, target=target_data, input=input_data, lambda_value=lambda_value, reduction='mean')
        # LOSS += loss_functions.perceptual_loss(output=output_data, target=target_data, channels=1, device=self.args.device)
        # LOSS += loss_functions.bce_dice_loss(output=output_data, target=target_data)

        ##########
        # Test 5 #
        ##########
        # lambda_value = 100.0
        # p = 1
        # LOSS = loss_functions.weighted_mask_loss(output=output_data, target=target_data, input=input_data, lambda_value=lambda_value, reduction='mean')
        # LOSS += loss_functions.reconstruction_loss(output=output_data, target=target_data)
        # LOSS += loss_functions.total_variation_loss(output=output_data, target=target_data, p=p, device=self.args.device)

        ##########
        # Test 6 #
        ##########
        # lambda_value = 100.0
        # p = 1
        # LOSS = loss_functions.weighted_mask_loss(output=output_data, target=target_data, input=input_data, lambda_value=lambda_value, reduction='mean')
        # LOSS += loss_functions.perceptual_loss(output=output_data, target=target_data, channels=1, device=self.args.device)
        # LOSS += loss_functions.total_variation_loss(output=output_data, target=target_data, p=p, device=self.args.device)

        ##########
        # Test 7 #
        ##########
        # lambda_value = 100.0
        # p = 1
        # LOSS = loss_functions.weighted_mask_loss(output=output_data, target=target_data, input=input_data, lambda_value=lambda_value, reduction='mean')
        # LOSS += loss_functions.reconstruction_loss(output=output_data, target=target_data)
        # LOSS += loss_functions.edge_loss(output=output_data, target=target_data, device=self.args.device)

        ##########
        # Test 8 #
        ##########
        lambda_value = 100.0
        LOSS = loss_functions.weighted_mask_loss(output=output_data, target=target_data, input=input_data, lambda_value=lambda_value, reduction='sum')

        if output_confidence_data is not None:
            target_confidence_data = (target_data > 0).float()
            input_confidence_data = (input_data > 0).float()

            ############################
            # Confidence loss - Test 1 #
            ############################
            # Normalize by the number of pixels in the mask
            # LOSS = loss_functions.weighted_mask_loss(output=output_confidence_data, input=input_confidence_data, target=target_confidence_data, lambda_value=100.0, reduction='mean')
            # LOSS += loss_functions.perceptual_loss(output=output_confidence_data, target=target_confidence_data, channels=1, device=self.args.device)
            # LOSS += loss_functions.bce_dice_loss(output=output_confidence_data, target=target_confidence_data)

            ############################
            # Confidence loss - Test 2 #
            ############################
            LOSS = loss_functions.weighted_mask_loss(output=output_confidence_data, input=input_confidence_data, target=target_confidence_data, lambda_value=100.0, reduction='sum')

        return LOSS

    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        batch_idx = -1
        for batch_data in self.train_loader:
            batch_idx += 1
            (input_data, target_data) = batch_data

            # Notice: Faster on CPU
            if self.args.dataset in self.datasets_for_holes:
                target_data = input_data.clone()

                # TODO: Threshold
                # apply_threshold(input_data, 0.5)
                # apply_threshold(target_data, 0.5)

                # Fix for Trees dataset - Fixed problem
                # if input_data.dtype != torch.float32:
                #     input_data = input_data.float()
                # if target_data.dtype != torch.float32:
                #     target_data = target_data.float()

                # # For Equality Check
                # res = torch.eq(input_data, target_data)
                # print(res.max())
                # print(res.min())

                train_utils.create_2d_holes(input_data=input_data)

            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            # # For Equality Check
            # res = torch.eq(input_data, target_data)
            # print(res.max())
            # print(res.min())

            self.optimizer.zero_grad()
            output_data = self.model(input_data)
            loss = self.loss_function(
                output_data=output_data,
                target_data=target_data,
                input_data=input_data
            )
            loss.backward()

            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print(
                    '[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                        epoch,
                        batch_idx * len(input_data),
                        len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        loss.item() / len(input_data)
                    )
                )

        train_avg_loss = train_loss / len(self.train_loader.dataset)
        print(
            '> [Train] Epoch: {} Average loss: {}'.format(
                epoch,
                train_avg_loss
            )
        )
        return train_avg_loss

    def _test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            batch_idx = -1
            for batch_data in self.test_loader:
                batch_idx += 1
                (input_data, target_data) = batch_data

                if self.args.dataset in self.datasets_for_holes:
                    target_data = input_data.clone()

                    # TODO: Threshold
                    # apply_threshold(input_data, 0.5)
                    # apply_threshold(target_data, 0.5)

                    # if input_data.dtype != torch.float32:
                    #     input_data = input_data.float()
                    # if target_data.dtype != torch.float32:
                    #     target_data = target_data.float()

                    train_utils.create_2d_holes(input_data=input_data)

                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                output_data = self.model(input_data)
                test_loss += self.loss_function(
                    output_data=output_data,
                    target_data=target_data,
                    input_data=input_data
                ).item()

        test_avg_loss = test_loss / len(self.test_loader.dataset)
        print(
            '> [Test] Epoch: {}, Average Loss: {} (Train Time Elapsed: {})'.format(
                epoch,
                test_avg_loss,
                datetime.datetime.now() - self.start_time
            )
        )
        return test_avg_loss

    def train(self, use_weights=False):
        self.args.index_data = False
        if use_weights is True:
            print("Loading Model Weights")
            self.model.load_state_dict(torch.load(self.args.weights_filepath))

        self.start_time = datetime.datetime.now()
        start_timestamp = self.start_time.strftime('%Y-%m-%d_%H-%M-%S')
        print(
            f"[Model: '{self.model.model_name}'] Training... "
            f"(Timestamp: {start_timestamp})"
        )
        try:
            for epoch in range(1, self.args.epochs + 1):
                train_avg_loss = self._train(epoch=epoch)
                test_avg_loss = self._test(epoch=epoch)
                if self.args.wandb:
                    wandb.log(
                        data={"Train Loss": train_avg_loss, "Test Loss": test_avg_loss},
                        step=epoch
                    )
        except (KeyboardInterrupt, SystemExit):
            print(f"[Model: '{self.model.model_name}'] Manual Interruption!")

        end_time = datetime.datetime.now()
        end_timestamp = end_time.strftime('%Y-%m-%d_%H-%M-%S')
        print(
            f"[Model: '{self.model.model_name}'] Saving Model Weights... "
            f"(Timestamp: {end_timestamp}, Train Time Elapsed: {end_time - self.start_time})"
        )
        model_parameters = copy.deepcopy(self.model.state_dict())
        torch.save(model_parameters, self.args.weights_filepath)

    # Only supports the custom datasets
    # Add to the other 2 custom dataset files
    @staticmethod
    def _print_batch_data_files(subset: torch.utils.data.Subset, data_indices):
        # subset_files = subset.dataset.data_files1
        subset_files = subset.dataset.data_files2
        subset_indices = subset.indices
        idx = -1
        batch_data_files = []
        for data_idx in data_indices:
            idx += 1
            batch_data_file = subset_files[subset_indices[data_idx]]
            print(f"File Index: {idx}, Filepath: {batch_data_file}")
            batch_data_files.append(batch_data_file)
        return batch_data_files

    # TODO: Handle V1 and V2 cases
    def predict(self, max_batches_to_plot=20):
        self.args.index_data = True
        os.makedirs(name=self.args.results_path, exist_ok=True)

        # Load model weights
        if os.path.exists(self.args.weights_filepath):
            self.model.load_state_dict(torch.load(self.args.weights_filepath))
        self.model.eval()

        with torch.no_grad():
            batches_to_plot = min(len(self.test_loader), max_batches_to_plot)
            z_fill_count = len(str(batches_to_plot))
            batch_idx = -1
            for batch_data in self.test_loader:
                batch_idx += 1
                batch_num = batch_idx + 1

                self.start_time = datetime.datetime.now()
                start_timestamp = self.start_time.strftime('%Y-%m-%d_%H-%M-%S')
                print(
                    f"[Model: '{self.model.model_name}'] Predicting Batch: {batch_num}/{batches_to_plot} "
                    f"(Time: {start_timestamp})"
                )

                # Get the images from the test loader
                data_indices, (input_data, target_data) = batch_data

                if self.args.dataset in self.datasets_for_holes:
                    target_data = input_data.clone()

                    # TODO: Threshold
                    # apply_threshold(input_data, 0.5)
                    # apply_threshold(target_data, 0.5)

                    # Fix for Trees dataset - Fixed problem
                    # if input_data.dtype != torch.float32:
                    #     input_data = input_data.float()
                    # if target_data.dtype != torch.float32:
                    #     target_data = target_data.float()

                    train_utils.create_2d_holes(input_data=input_data)

                input_data = input_data.to(self.device)
                # target_data = target_data.to(self.device)
                output_data = self.model(input_data)

                end_time = datetime.datetime.now()
                end_timestamp = end_time.strftime('%Y-%m-%d_%H-%M-%S')
                print(
                    f"[Model: '{self.model.model_name}'] Parsing Output... "
                    f"(Timestamp: {end_timestamp}, Predict Time Elapsed: {end_time - self.start_time})"
                )

                batch_data_files = self._print_batch_data_files(
                    subset=self.test_loader.dataset.dataset,
                    data_indices=data_indices
                )

                #################
                # Parse Results #
                #################

                if "confidence map" in getattr(self.model, "additional_tasks", list()):
                    output_data, output_confidence_data = output_data
                    map_merge_data = torch.where(output_confidence_data > 0.5, output_data, 0)
                else:
                    output_confidence_data = None
                    map_merge_data = output_data.clone()

                # TODO: Threshold
                # apply_threshold(output_images, 0.5)

                merge_data = torch.where(input_data > 0, input_data, map_merge_data)

                # Detach the images from the cuda and move them to CPU
                if self.args.cuda is True:
                    input_data = input_data.cpu()
                    # target_data = target_data.cpu()
                    output_data = output_data.cpu()
                    merge_data = merge_data.cpu()

                    if output_confidence_data is not None:
                        output_confidence_data = output_confidence_data.cpu()
                        map_merge_data = map_merge_data.cpu()

                # Convert (b, 6, w, h) to (6*b, 1, w, h) - Trees2DV2
                if input_data.shape[1] == 6:
                    x, y = self.args.input_size[1:]
                    input_data = input_data.view(-1, 1, x, y)
                    target_data = target_data.view(-1, 1, x, y)
                    output_data = output_data.view(-1, 1, x, y)
                    fusion_data = fusion_data.view(-1, 1, x, y)

                    if output_confidence_data is not None:
                        output_confidence_data = output_confidence_data.view(-1, 1, x, y)
                        map_merge_data = map_merge_data.view(-1, 1, x, y)

                #################
                # Visualization #
                #################

                plotting_data_list = [
                    {"Title": "Input", "Data": input_data},
                    {"Title": "Target", "Data": target_data},
                    {"Title": "Output", "Data": output_data}
                ]

                # Handle additional_tasks
                if "confidence map" in getattr(self.model, "additional_tasks", list()):
                    additional_plotting_data_list = [
                        {"Title": "Confidence", "Data": output_confidence_data},
                        {"Title": "Map Merge", "Data": map_merge_data}
                    ]
                    plotting_data_list += additional_plotting_data_list

                plotting_data_list.append({"Title": "Merge", "Data": merge_data})

                ##########################
                # V1 - without filepaths #
                ##########################

                # # Create a grid of images
                # columns = len(plotting_data_list)
                # rows = input_data.size(0)
                # fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
                # ax = []
                # for row_idx in tqdm(range(rows)):
                #     for col_idx in range(columns):
                #         data: torch.Tensor = plotting_data_list[col_idx]["Data"]
                #         ax.append(fig.add_subplot(rows, columns, row_idx * columns + col_idx + 1))
                #         numpy_image = data[row_idx].numpy()
                #         plt.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')
                #
                # for col_idx in range(columns):
                #     title: str = plotting_data_list[col_idx]["Title"]
                #     ax[col_idx].set_title(f"{title}:")

                #######################
                # V2 - with filepaths #
                #######################

                # Create a grid of images
                columns = len(plotting_data_list)
                rows = input_data.size(0)
                total_rows = rows * 2
                fig = plt.figure(figsize=(columns + 0.5, rows + (rows / 2) + 0.5))
                row_heights = [1.0 if i % 2 == 0 else 0.3 for i in range(total_rows)]  # Image: 1.0, Text: 0.3
                gs = gridspec.GridSpec(total_rows, columns, figure=fig, height_ratios=row_heights)
                title_axes = []
                for row_idx in tqdm(range(rows)):
                    for col_idx in range(columns):
                        ax = fig.add_subplot(gs[row_idx * 2, col_idx])
                        data: torch.Tensor = plotting_data_list[col_idx]["Data"]
                        numpy_image = data[row_idx].numpy()
                        ax.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')

                        # Store first row axes for setting titles
                        if row_idx == 0:
                            title_axes.append(ax)

                    # Add text below this row of images
                    ax_text = fig.add_subplot(gs[row_idx * 2 + 1, :])  # span all columns
                    ax_text.axis('off')
                    relative_path = os.path.basename(batch_data_files[row_idx])
                    ax_text.text(0.5, 0.5, f"Filepath: {relative_path}", fontsize=12, ha='center', va='center')

                # Add titles to the top row of image columns
                for col_idx, ax in enumerate(title_axes):
                    title: str = plotting_data_list[col_idx]["Title"]
                    ax.set_title(f"{title}:")

                fig.tight_layout()
                save_filename = os.path.join(self.args.results_path, f"{self.args.dataset}_{batch_num}.png")
                plt.savefig(save_filename)

                # Log the image to wandb
                if self.args.wandb:
                    batch_num_str = str(batch_num).zfill(z_fill_count)
                    wandb.log(
                        data={f"Batch {batch_num_str} - Predict Plots": wandb.Image(plt)}
                    )

                if batch_num == max_batches_to_plot:
                    break
