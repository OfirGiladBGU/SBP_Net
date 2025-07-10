import argparse
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from configs.configs_parser import *
from datasets.custom_datasets_1d import TreesCustomDataset1D
from datasets.custom_datasets_2d import TreesCustomDataset2D
from datasets.custom_datasets_3d import TreesCustomDataset3D


#########################
# Generic Dataset Class #
#########################
class IndexableSubset(Dataset):
    def __init__(self, args: argparse.Namespace, dataset):
        self.args = args
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_idx = self.dataset[idx]
        if self.args.index_data:
            return idx, data_idx
        else:
            return data_idx


class IndexableDataset(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}
        self.kwargs['batch_size'] = self.args.batch_size

        self.train_loader = None
        self.test_loader = None

    def init_dataloaders(self, train_subset, test_subset):
        self.train_loader = DataLoader(
            dataset=IndexableSubset(args=self.args, dataset=train_subset),
            shuffle=True,
            **self.kwargs
        )
        self.test_loader = DataLoader(
            dataset=IndexableSubset(args=self.args, dataset=test_subset),
            shuffle=False,
            **self.kwargs
        )

###################
# Public Datasets #
###################
class MNIST(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(MNIST, self).__init__(args=args)

        self.input_size = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        root = os.path.join(DATA_PATH, "mnist")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size[1])
        ])

        self.init_dataloaders(
            train_subset=datasets.MNIST(root=root, train=True, download=True, transform=transform),
            test_subset=datasets.MNIST(root=root, train=False, transform=transform)
        )


class EMNIST(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(EMNIST, self).__init__(args=args)

        self.input_size = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        root = os.path.join(DATA_PATH, "emnist")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size[1])
        ])

        self.init_dataloaders(
            train_subset=datasets.EMNIST(root=root, train=True, download=True, split='byclass', transform=transform),
            test_subset=datasets.EMNIST(root=root, train=False, split='byclass', transform=transform)
        )


class FashionMNIST(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(FashionMNIST, self).__init__(args=args)

        self.input_size = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        root = os.path.join(DATA_PATH, "fmnist")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size[1])
        ])

        self.init_dataloaders(
            train_subset=datasets.FashionMNIST(root=root, train=True, download=True, transform=transform),
            test_subset=datasets.FashionMNIST(root=root, train=False, transform=transform)
        )


class CIFAR10(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(CIFAR10, self).__init__(args=args)

        self.input_size = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        root = os.path.join(DATA_PATH, "cifar10")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size[1]),
            transforms.Grayscale(num_output_channels=1)
        ])

        self.init_dataloaders(
            train_subset=datasets.CIFAR10(root=root, train=True, download=True, transform=transform),
            test_subset=datasets.CIFAR10(root=root, train=False, transform=transform)
        )


##################
# Local Datasets #
##################
class TreesDataset1DV1(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset1DV1, self).__init__(args=args)

        self.input_size = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        # src_path = PREDS_FIXED_2D
        src_path = PREDS_ADVANCED_FIXED_2D
        args.include_regression = False

        data_paths = [src_path]
        trees_dataset = TreesCustomDataset1D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 2D labels (with random holes) to predict 2D labels
class TreesDataset2DV1S(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset2DV1S, self).__init__(args=args)

        self.input_size = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        src_path = LABELS_2D
        args.include_regression = False

        data_paths = [src_path]
        trees_dataset = TreesCustomDataset2D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 2D preds fixed to predict 2D labels
class TreesDataset2DV1(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset2DV1, self).__init__(args=args)

        self.input_size = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        # src_path = PREDS_FIXED_2D
        src_path = PREDS_ADVANCED_FIXED_2D
        dst_path = LABELS_2D
        args.include_regression = False

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset2D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 2D preds fixed to predict 2D labels (with regression)
class TreesDataset2DV1R(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset2DV1R, self).__init__(args=args)

        self.input_size = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        # src_path = PREDS_FIXED_2D
        src_path = PREDS_ADVANCED_FIXED_2D
        dst_path = LABELS_2D
        args.include_regression = True

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset2D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 6 2D labels (with random holes) to predict 2D labels
class TreesDataset2DV2S(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset2DV2S, self).__init__(args=args)

        self.input_size = (6, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        src_path = LABELS_2D
        args.include_regression = False

        data_paths = [src_path]
        trees_dataset = TreesCustomDataset2D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 6 2D preds fixed to predict 6 2D labels
class TreesDataset2DV2(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset2DV2, self).__init__(args=args)

        self.input_size = (6, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        # src_path = PREDS_FIXED_2D
        src_path = PREDS_ADVANCED_FIXED_2D
        dst_path = LABELS_2D
        args.include_regression = False

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset2D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 6 2D preds fixed to predict 6 2D labels (with regression)
class TreesDataset2DV2R(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset2DV2R, self).__init__(args=args)

        self.input_size = (6, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        # src_path = PREDS_FIXED_2D
        src_path = PREDS_ADVANCED_FIXED_2D
        dst_path = LABELS_2D
        args.include_regression = True

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset2D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 6 2D labels to predict 3D labels
class TreesDataset3DV1(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset3DV1, self).__init__(args=args)

        self.input_size = (6, 1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        src_path = LABELS_2D
        dst_path = LABELS_3D
        args.include_regression = False

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset3D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )

# Train with 3D reconstructed labels to predict 3D labels (with regression)
class TreesDataset3DV1R(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset3DV1R, self).__init__(args=args)

        self.input_size = (6, 1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

        src_path = LABELS_2D
        dst_path = LABELS_3D
        args.include_regression = True

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset3D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 3D reconstructed labels to predict 3D labels
class TreesDataset3DV2(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset3DV2, self).__init__(args=args)

        self.input_size = (1, DATA_3D_SIZE[0], DATA_3D_SIZE[1], DATA_3D_SIZE[2])

        src_path = LABELS_3D_RECONSTRUCT
        dst_path = LABELS_3D
        args.include_regression = False

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset3D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 3D reconstructed labels to predict 3D labels (with regression)
class TreesDataset3DV2R(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset3DV2R, self).__init__(args=args)

        self.input_size = (1, DATA_3D_SIZE[0], DATA_3D_SIZE[1], DATA_3D_SIZE[2])

        src_path = LABELS_3D_RECONSTRUCT
        dst_path = LABELS_3D
        args.include_regression = True

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset3D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 3D preds fixed to predict 3D labels (Direct Repair)
class TreesDataset3DV2D(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset3DV2D, self).__init__(args=args)

        self.input_size = (1, DATA_3D_SIZE[0], DATA_3D_SIZE[1], DATA_3D_SIZE[2])

        src_path = PREDS_ADVANCED_FIXED_3D
        dst_path = LABELS_3D
        args.include_regression = False

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset3D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with 3D preds fixed fusion to predict 3D labels
class TreesDataset3DV2F(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset3DV2F, self).__init__(args=args)

        self.input_size = (1, DATA_3D_SIZE[0], DATA_3D_SIZE[1], DATA_3D_SIZE[2])

        src_path = PREDS_FIXED_3D_FUSION
        dst_path = LABELS_3D
        args.include_regression = False

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset3D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Train with FULL 3D preds fixed to predict FULL 3D labels
class TreesDataset3DV3(IndexableDataset):
    def __init__(self, args: argparse.Namespace):
        super(TreesDataset3DV3, self).__init__(args=args)
        if args.batch_size != 1:
            raise ValueError("Batch size must be 1")

        # Dynamic Size
        self.input_size = (1, -1, -1, -1)

        src_path = PREDS_FIXED
        dst_path = LABELS
        args.include_regression = False

        data_paths = [src_path, dst_path]
        trees_dataset = TreesCustomDataset3D(args=args, data_paths=data_paths)

        self.init_dataloaders(
            train_subset=trees_dataset.train_subset,
            test_subset=trees_dataset.test_subset
        )


# Init Method
def init_dataset(args: argparse.Namespace):
    print(f"[Dataset: '{args.dataset}'] Initializing...")
    dataset_map = {
        # Public Datasets
        "MNIST": MNIST,
        "EMNIST": EMNIST,
        "FashionMNIST": FashionMNIST,
        "CIFAR10": CIFAR10,

        # 1D Datasets
        "Trees1DV1": TreesDataset1DV1,

        # 2D Datasets
        "Trees2DV1S": TreesDataset2DV1S,
        "Trees2DV1": TreesDataset2DV1,
        "Trees2DV1R": TreesDataset2DV1R,
        "Trees2DV2S": TreesDataset2DV2S,
        "Trees2DV2": TreesDataset2DV2,
        "Trees2DV2R": TreesDataset2DV2R,

        # 3D Datasets
        "Trees3DV1": TreesDataset3DV1,
        "Trees3DV1R": TreesDataset3DV1R,
        "Trees3DV2": TreesDataset3DV2,
        "Trees3DV2R": TreesDataset3DV2R,
        "Trees3DV2D": TreesDataset3DV2D,
        "Trees3DV2F": TreesDataset3DV2F,
        "Trees3DV3": TreesDataset3DV3,
    }
    if args.dataset in list(dataset_map.keys()):
        return dataset_map[args.dataset](args=args)
    else:
        raise Exception("Dataset not available in 'Dataset List'")
