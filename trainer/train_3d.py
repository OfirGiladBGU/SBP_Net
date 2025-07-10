import argparse
import torch
import torch.utils.data
from torch import optim
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import datetime

from configs.configs_parser import V1_3D_DATASETS, V2_3D_DATASETS, IMAGES_6_VIEWS
from datasets.dataset_utils import apply_threshold, convert_numpy_to_data_file
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

        if self.model.model_name == "unet3d":
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=0.00001)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def loss_function(self, output_data, target_data, input_data=None):
        """
        :param output_data: model output on the 'original' input
        :param target_data: the target data that the model should output
        :param input_data: the original input data for the model
        :return:
        """
        if self.model.model_name == "unet3d":
            LOSS = loss_functions.weighted_bce_dice_loss(output=output_data, target=target_data)
            return LOSS

        ##########
        # Test 1 #
        ##########
        # LOSS = loss_functions.mse_loss(output=output_data, target=target_data, reduction='sum')

        ##########
        # Test 2 #
        ##########
        # LOSS = loss_functions.bce_dice_loss(output=output_data, target=target_data)

        ##########
        # Test 3 #
        ##########
        # LOSS = loss_functions.weighted_bce_dice_loss(output=output_data, target=target_data)

        ##########
        # Test 4 #
        ##########
        lambda_value = 10.0
        LOSS = loss_functions.weighted_mask_loss(output=output_data, target=target_data, input=input_data, lambda_value=lambda_value, reduction='sum')

        return LOSS

    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        batch_idx = -1
        for batch_data in self.train_loader:
            batch_idx += 1
            (input_data, target_data) = batch_data

            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

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
            '> [Train] Epoch: {}, Average Loss: {}'.format(
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
        for data_idx in data_indices:
            idx += 1
            print(f"File Index: {idx}, Filepath: {subset_files[subset_indices[data_idx]]}")

    # TODO: Handle V1 and V2 cases
    def predict(self, max_batches_to_plot=2):
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

                input_data = input_data.to(self.device)
                # target_data = target_data.to(self.device)
                output_data = self.model(input_data)

                end_time = datetime.datetime.now()
                end_timestamp = end_time.strftime('%Y-%m-%d_%H-%M-%S')
                print(
                    f"[Model: '{self.model.model_name}'] Parsing Output... "
                    f"(Timestamp: {end_timestamp}, Predict Time Elapsed: {end_time - self.start_time})"
                )

                self._print_batch_data_files(
                    subset=self.test_loader.dataset.dataset,
                    data_indices=data_indices
                )

                #################
                # Parse Results #
                #################

                # TODO: Threshold
                threshold_data = output_data.clone()
                apply_threshold(threshold_data, threshold=0.1, keep_values=False)

                occluded_data = ((threshold_data - input_data) > 0.5).float()
                apply_threshold(occluded_data, threshold=0.5, keep_values=False)

                merge_data = torch.where(input_data > 0, input_data, threshold_data)

                # Detach the images from the cuda and move them to CPU
                if self.args.cuda:
                    input_data = input_data.cpu()
                    # target_data = target_data.cpu()
                    # output_data = output_data.cpu()  # Not relevant
                    threshold_data = threshold_data.cpu()
                    occluded_data = occluded_data.cpu()
                    merge_data = merge_data.cpu()

                #################
                # Visualization #
                #################

                plotting_data_list = [
                    {"Title": "Input", "Data": input_data},
                    {"Title": "Target", "Data": target_data},
                    # {"Title": "Output", "Data": output_data},  # Not relevant
                    {"Title": "Threshold", "Data": threshold_data},
                    {"Title": "Occluded", "Data": occluded_data},
                    {"Title": "Merge", "Data": merge_data}
                ]

                # Save 3d results and 2d results that will be used for the grid output
                images_info = list()
                data_3d_path = os.path.join(self.args.results_path, "data_3d")
                data_2d_path = os.path.join(self.args.results_path, "data_2d")
                os.makedirs(name=data_3d_path, exist_ok=True)
                os.makedirs(name=data_2d_path, exist_ok=True)

                columns = len(plotting_data_list)
                rows = input_data.size(0)
                for row_idx in tqdm(range(rows)):
                    images_info_idx = dict()

                    for col_idx in range(columns):
                        data: torch.Tensor = plotting_data_list[col_idx]["Data"]
                        title: str = plotting_data_list[col_idx]["Title"]

                        save_name = f"{self.args.dataset}_{batch_num}_{row_idx}_{title.lower()}"
                        save_filename_3d = os.path.join(data_3d_path, save_name)
                        save_filename_2d = os.path.join(data_2d_path, save_name)

                        # Handle Input
                        if title.lower() == "input" and self.args.dataset in V1_3D_DATASETS:
                            # Create a grid of images
                            input_columns = len(IMAGES_6_VIEWS)
                            input_rows = 1
                            fig = plt.figure(figsize=(input_columns + 0.5, input_rows + 0.5))
                            ax = []
                            for view_idx, view_name in enumerate(IMAGES_6_VIEWS):
                                ax.append(fig.add_subplot(input_rows, input_columns, view_idx + 1))
                                numpy_image = data[row_idx][view_idx].numpy()
                                plt.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')
                                ax[view_idx].set_title(f"{view_name} view:")

                            fig.tight_layout()
                            plt.savefig(save_filename_2d)
                            save_status = True
                        else:
                            data_idx = data[row_idx].squeeze().numpy()
                            # np.save(file=save_filename_3d, arr=data_idx)
                            convert_numpy_to_data_file(
                                numpy_data=data_idx,
                                source_data_filepath="dummy.npy",
                                save_filename=save_filename_3d
                            )
                            save_status = train_utils.data_3d_to_2d_plot(
                                data_3d=data_idx,
                                save_filename=save_filename_2d
                            )

                        # True - Plot was saved, False - Plot was not saved (Empty Image)
                        if save_status:
                            images_info_idx[title.lower()] = save_filename_2d
                        else:
                            images_info_idx[title.lower()] = None

                    images_info.append(images_info_idx)

                ###########################
                # Create a grid of images #
                ###########################

                img_width = 0
                img_height = 0
                font_size = 20  # Define a font size (can adjust based on preferences)
                headers = []  # Define the text for each column header

                for col_idx in range(columns):
                    title: str = plotting_data_list[col_idx]["Title"]
                    image_filepath = images_info[0][title.lower()]
                    if image_filepath is not None:
                        image_sample = Image.open(fp=f"{image_filepath}.png")
                        img_width = max(img_width, image_sample.size[0])
                        img_height = max(img_height, image_sample.size[1])
                        image_sample.close()
                    headers.append(f"{title}:")

                # Create a font object (You may need to specify the path to a TTF font on your system)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()

                # Create a blank image for the grid, add extra space at the top for the headers
                header_height = font_size + 10  # Extra space for the header
                grid_img_width = columns * img_width
                grid_img_height = rows * img_height + header_height  # Add space for header
                grid_img = Image.new(
                    mode='RGB',
                    size=(grid_img_width, grid_img_height),
                    color=(255, 255, 255)
                )  # White background

                # Create a drawing context for the header
                draw = ImageDraw.Draw(grid_img)

                # Add the headers
                for col_idx in range(columns):
                    header_text = headers[col_idx]

                    # Calculate the bounding box of the text using textbbox
                    text_bbox = draw.textbbox((0, 0), header_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]

                    # Center text in the column
                    x_position = col_idx * img_width + (img_width - text_width) // 2
                    draw.text((x_position, 5), header_text, font=font, fill="black")  # 5px padding from the top

                # Paste images into the grid (below the header)
                for row_idx in tqdm(range(rows)):
                    for col_idx in range(columns):
                        # Load image and paste into grid
                        title: str = plotting_data_list[col_idx]["Title"]
                        image_filepath = images_info[row_idx][title.lower()]
                        if image_filepath is not None:
                            data_image = Image.open(fp=f"{image_filepath}.png")
                            grid_img.paste(data_image, (col_idx * img_width, header_height + row_idx * img_height))

                # Save the combined image
                save_name = f"{self.args.dataset}_{batch_num}"
                save_filename = os.path.join(self.args.results_path, f"{save_name}.png")
                grid_img.save(save_filename)

                # Log the image to wandb
                if self.args.wandb:
                    batch_num_str = str(batch_num).zfill(z_fill_count)
                    wandb.log(
                        data={f"Batch {batch_num_str} - Predict Plots": wandb.Image(grid_img)}
                    )

                if batch_num == max_batches_to_plot:
                    break
