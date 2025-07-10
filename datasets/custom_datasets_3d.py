import argparse
import os
import pathlib
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import pandas as pd
import numpy as np

from configs.configs_parser import TRAIN_LOG_PATH, V1_3D_DATASETS, V2_3D_DATASETS, V3_3D_DATASETS, IMAGES_6_VIEWS
from datasets.dataset_utils import get_data_file_stem, convert_data_file_to_numpy, validate_data_paths


# 6 2D inputs + 1 3D target
class TreesCustomDatasetV1(Dataset):
    def __init__(self, args: argparse.Namespace, data_paths: list, transform2d=None, transform3d=None):
        self.args = args
        self.data_paths = data_paths
        self.transform2d = transform2d
        self.transform3d = transform3d
        self.to_tensor = transforms.ToTensor()

        if not os.path.exists(TRAIN_LOG_PATH):
            raise FileNotFoundError(f"File not found: {TRAIN_LOG_PATH}")
        # elif self.args.include_regression is True:
        self.log_data = pd.read_csv(TRAIN_LOG_PATH)

        self.paths_count = len(data_paths)
        if not (1 <= self.paths_count <= 2):
            raise ValueError("Invalid number of data paths")
        current_count = self.paths_count

        # current_count > 0:
        # self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
        # self.data_files1 = sorted(self.data_files1)

        self.data_files1 = []
        current_count -= 1

        if current_count > 0:
            # self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.*")
            # self.data_files2 = sorted(self.data_files2)

            self.data_files2 = []
            current_count -= 1
        else:
            self.data_files2 = None

        col_0 = self.log_data.columns[0]
        for row_idx, row in self.log_data.iterrows():
            for image_view in IMAGES_6_VIEWS:
                data_file1 = os.path.join(data_paths[0], f"{row[col_0]}_{image_view}.png")
                self.data_files1.append(data_file1)

            if self.data_files2 is not None:
                data_file2 = list(pathlib.Path(data_paths[1]).glob(f"{row[col_0]}.*"))[0]
                self.data_files2.append(data_file2)

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        item = tuple()

        batch = list()
        target = None

        data_idx = idx * 6
        for i in range(6):
            data_file1 = str(self.data_files1[data_idx + i])
            numpy_2d_data1 = convert_data_file_to_numpy(data_filepath=data_file1)

            numpy_2d_data1 = self.to_tensor(numpy_2d_data1)
            if self.transform2d is not None:
                numpy_2d_data1 = self.transform2d(numpy_2d_data1)
            batch.append(numpy_2d_data1)
        batch = torch.stack(batch)

        # 6 images + 1 3D data
        if self.data_files2 is not None:
            data_file2 =  str(self.data_files2[idx])
            numpy_3d_data2 = convert_data_file_to_numpy(data_filepath=data_file2)
            numpy_3d_data2 = numpy_3d_data2.astype(np.float32)

            numpy_3d_data2 = torch.Tensor(numpy_3d_data2)
            if self.transform3d is not None:
                numpy_3d_data2 = self.transform3d(numpy_3d_data2)
            target = torch.stack([numpy_3d_data2])

        if self.paths_count == 1:
            item += (batch, -1)
        elif self.paths_count == 2:
            item += (batch, target)
        else:
            pass

        if self.args.include_regression is True:
            label_local_components = self.log_data["label_local_components"][idx]
            item += (label_local_components,)

        return item


# 1 3D input + 1 3D target
class TreesCustomDatasetV2(Dataset):
    def __init__(self, args: argparse.Namespace, data_paths: list, transform3d=None):
        self.args = args
        self.data_paths = data_paths
        self.transform3d = transform3d

        if not os.path.exists(TRAIN_LOG_PATH):
            raise FileNotFoundError(f"File not found: {TRAIN_LOG_PATH}")
        # elif self.args.include_regression is True:
        self.log_data = pd.read_csv(TRAIN_LOG_PATH)

        self.paths_count = len(data_paths)
        if not (1 <= self.paths_count <= 2):
            raise ValueError("Invalid number of data paths")
        current_count = self.paths_count

        # current_count > 0:
        # self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.*")
        # self.data_files1 = sorted(self.data_files1)

        self.data_files1 = []
        current_count -= 1

        if current_count > 0:
            # self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.*")
            # self.data_files2 = sorted(self.data_files2)

            self.data_files2 = []
            current_count -= 1
        else:
            self.data_files2 = None

        col_0 = self.log_data.columns[0]
        for row_idx, row in self.log_data.iterrows():
            data_file1 = list(pathlib.Path(data_paths[0]).glob(f"{row[col_0]}.*"))[0]
            self.data_files1.append(data_file1)

            if self.data_files2 is not None:
                data_file2 = list(pathlib.Path(data_paths[1]).glob(f"{row[col_0]}.*"))[0]
                self.data_files2.append(data_file2)

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        item = tuple()

        numpy_3d_data2 = None

        data_file1 = str(self.data_files1[idx])
        numpy_3d_data1 = convert_data_file_to_numpy(data_filepath=data_file1)
        numpy_3d_data1 = numpy_3d_data1.astype(np.float32)

        numpy_3d_data1 = torch.Tensor(numpy_3d_data1)
        if self.transform3d is not None:
            numpy_3d_data1 = self.transform3d(numpy_3d_data1)
        numpy_3d_data1 = numpy_3d_data1.unsqueeze(0)

        # 1 3D input + 1 3D target
        if self.data_files2 is not None:
            data_file2 = str(self.data_files2[idx])
            numpy_3d_data2 = convert_data_file_to_numpy(data_filepath=data_file2)
            numpy_3d_data2 = numpy_3d_data2.astype(np.float32)

            numpy_3d_data2 = torch.Tensor(numpy_3d_data2)
            if self.transform3d is not None:
                numpy_3d_data2 = self.transform3d(numpy_3d_data2)
            numpy_3d_data2 = numpy_3d_data2.unsqueeze(0)

        if self.paths_count == 1:
            item += (numpy_3d_data1, -1)
        elif self.paths_count == 2:
            item += (numpy_3d_data1, numpy_3d_data2)
        else:
            pass

        if self.args.include_regression is True:
            label_local_components = self.log_data["label_local_components"][idx]
            item += (label_local_components,)

        return item


# 1 3D input + 1 3D target (Full)
class TreesCustomDatasetV3(Dataset):
    def __init__(self, args: argparse.Namespace, data_paths: list, transform3d=None):
        self.args = args
        self.data_paths = data_paths
        self.transform3d = transform3d

        self.log_data = None

        self.paths_count = len(data_paths)
        if not (1 <= self.paths_count <= 2):
            raise ValueError("Invalid number of data paths")
        current_count = self.paths_count

        # current_count > 0:
        self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.*")
        self.data_files1 = sorted(self.data_files1)

        current_count -= 1

        if current_count > 0:
            self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.*")
            self.data_files2 = sorted(self.data_files2)

            current_count -= 1
        else:
            self.data_files2 = None

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        item = tuple()

        numpy_3d_data2 = None

        data_file1 = str(self.data_files1[idx])
        numpy_3d_data1 = convert_data_file_to_numpy(data_filepath=data_file1)
        numpy_3d_data1 = numpy_3d_data1.astype(np.float32)

        numpy_3d_data1 = torch.Tensor(numpy_3d_data1)
        if self.transform3d is not None:
            numpy_3d_data1 = self.transform3d(numpy_3d_data1)
        numpy_3d_data1 = numpy_3d_data1.unsqueeze(0)

        # 1 3D input + 1 3D target
        if self.data_files2 is not None:
            data_file2 = str(self.data_files2[idx])
            numpy_3d_data2 = convert_data_file_to_numpy(data_filepath=data_file2)
            numpy_3d_data2 = numpy_3d_data2.astype(np.float32)

            numpy_3d_data2 = torch.Tensor(numpy_3d_data2)
            if self.transform3d is not None:
                numpy_3d_data2 = self.transform3d(numpy_3d_data2)
            numpy_3d_data2 = numpy_3d_data2.unsqueeze(0)

        if self.paths_count == 1:
            item += (numpy_3d_data1, -1)
        elif self.paths_count == 2:
            item += (numpy_3d_data1, numpy_3d_data2)
        else:
            pass

        if self.args.include_regression is True:
            raise ValueError("'include_regression' Not Supported")

        return item


class TreesCustomDataset3D:
    def __init__(self, args: argparse.Namespace, data_paths, transform2d=None, transform3d=None):
        validate_data_paths(data_paths=data_paths)
        self.args = args
        self.data_paths = data_paths
        self.transform2d = transform2d
        self.transform3d = transform3d
        self._init_subsets()

    def _init_subsets(self):
        if self.args.dataset in V1_3D_DATASETS:
            trees_dataset = TreesCustomDatasetV1(
                args=self.args,
                data_paths=self.data_paths,
                transform2d=self.transform2d,
                transform3d=self.transform3d
            )
        elif self.args.dataset in V2_3D_DATASETS:
            trees_dataset = TreesCustomDatasetV2(
                args=self.args,
                data_paths=self.data_paths,
                transform3d=self.transform3d
            )
        elif self.args.dataset in V3_3D_DATASETS:
            trees_dataset = TreesCustomDatasetV3(
                args=self.args,
                data_paths=self.data_paths,
                transform3d=self.transform3d
            )
        else:
            raise Exception("Dataset not available in 'Custom Dataset 3D'")

        # TODO: Perform split based on the 3D files count

        # Special case: Full data
        if trees_dataset.log_data is None:
            dataset_size = len(trees_dataset)
            train_size = int(dataset_size * 0.9)
            test_size = dataset_size - train_size
            train_indices = range(0, train_size)
            test_indices = range(train_size, train_size + test_size)

            # self.train_subset, self.test_subset = torch.utils.data.random_split(trees_dataset, [train_size, test_size])

            # Non random split
            self.train_subset = Subset(dataset=trees_dataset, indices=train_indices)
            self.test_subset = Subset(dataset=trees_dataset, indices=test_indices)
            return


        # # Option 1: Split by files percentage
        # dataset_size = len(trees_dataset)
        # train_size = int(dataset_size * 0.9)
        # test_size = dataset_size - train_size
        # train_indices = range(0, train_size)
        # test_indices = range(train_size, train_size + test_size)

        # # self.train_subset, self.test_subset = torch.utils.data.random_split(trees_dataset, [train_size, test_size])

        # Option 2: Split on based 3D files
        index_3d_uniques = trees_dataset.log_data["index_3d"].unique()
        split_percentage = 0.9
        index_3d_split_index = min(round(len(index_3d_uniques) * split_percentage), len(index_3d_uniques) - 1)
        data_3d_stem_list = [data_3d_stem[1:] for data_3d_stem in index_3d_uniques]

        train_stems = data_3d_stem_list[:index_3d_split_index]
        # test_stems = data_3d_stem_list[index_3d_split_index:]

        train_indices = []
        test_indices = []
        for idx in range(len(trees_dataset)):
            data_files2_stem = get_data_file_stem(
                data_filepath=trees_dataset.data_files2[idx],
                relative_to=trees_dataset.data_paths[1]
            )
            # Format: {output_idx}_{cube_idx}{ext}
            data_files2_output_idx = data_files2_stem.rsplit(sep='_', maxsplit=1)[0]
            if data_files2_output_idx in train_stems:
                train_indices.append(idx)
            else:
                test_indices.append(idx)

        # Non random split
        self.train_subset = Subset(dataset=trees_dataset, indices=train_indices)
        self.test_subset = Subset(dataset=trees_dataset, indices=test_indices)
