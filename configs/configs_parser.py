import pathlib
import yaml
from enum import Enum


# TODO: Setup Configurations File
CONFIG_FILENAME = "parse2022_SC_32.yaml"


# TODO: Use for examples:

# CONFIG_FILENAME = "examples/parse2022_LC_32.yaml"
# CONFIG_FILENAME = "examples/parse2022_LC_48.yaml"

# CONFIG_FILENAME = "examples/PipeForge3DMesh_LC_32.yaml"
# CONFIG_FILENAME = "examples/PipeForge3DMesh_LC_48.yaml"

# CONFIG_FILENAME = "examples/PipeForge3DPCD_LC_32.yaml"
# CONFIG_FILENAME = "examples/PipeForge3DPCD_LC_48.yaml"


# TODO: Use for Ex1:

# CONFIG_FILENAME = "experiment1/parse2022_LC_32_50.yaml"
# CONFIG_FILENAME = "experiment1/PipeForge3DMesh_Best_LC_32.yaml"
# CONFIG_FILENAME = "experiment1/PipeForge3DPCD_Best_LC_32.yaml"
# CONFIG_FILENAME = "experiment1/HospitalCUP_LC_32.yaml"


# TODO: Use for Ex2:

# V1 - 20 samples

# CONFIG_FILENAME = "experiment2/parse2022_LC_32.yaml"
# CONFIG_FILENAME = "experiment2/parse2022_LC_48.yaml"
# CONFIG_FILENAME = "experiment2/parse2022_LC_64.yaml"


# V2 - 50 samples

# CONFIG_FILENAME = "experiment2/parse2022_LC_32_50.yaml"
# CONFIG_FILENAME = "experiment2/parse2022_LC_48_50.yaml"
# CONFIG_FILENAME = "experiment2/parse2022_LC_64_50.yaml"


# TODO: Use for Ex3:

# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_Hole_125_LC_32.yaml"
# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_Hole_250_LC_32.yaml"
# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_Hole_325_LC_32.yaml"
# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_Hole_450_LC_32.yaml"
# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_Hole_700_LC_32.yaml"


# TODO: Use for Ex4:

# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_Base_LC_32.yaml"

# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_150_LC_32.yaml"
# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_200_LC_32.yaml"
# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_300_LC_32.yaml"
# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_400_LC_32.yaml"


# TODO: Use for Ex_Sota:

# CONFIG_FILENAME = "experiment_sota/parse2022_LC_128_50_unet3d.yaml"


# TODO: Use for Parse2022 test:

# CONFIG_FILENAME = "parse2022_test/parse2022_LC_32_test.yaml"


# TODO: Use for hyperparams tests:

# CONFIG_FILENAME = "tests/parse2022_LC_32_v1.yaml"
# CONFIG_FILENAME = "tests/parse2022_LC_32_v2.yaml"
# CONFIG_FILENAME = "tests/parse2022_LC_32_v3.yaml"
# CONFIG_FILENAME = "tests/parse2022_LC_32_v4.yaml"
# CONFIG_FILENAME = "tests/parse2022_LC_32_v5.yaml"


#####################
# Automatic Parsing #
#####################
print(f"[Init] Configuration Filename: {CONFIG_FILENAME}")

# Util classes
class ModelType(Enum):
    Model_1D = 1
    Model_2D = 2
    Model_3D = 3

class DataType(Enum):
    TRAIN = 1
    EVAL = 2

class TaskType(Enum):
    SINGLE_COMPONENT = 1  # Assumption: Need to achieve a single component
    LOCAL_CONNECTIVITY = 2  # Assumption: Need to connect components on focused scope
    PATCH_HOLES = 3  # Assumption: Need to fix any type of holes

class ProjectionMode(Enum):  # TODO: Not yet used
    VISUAL_MODE = 1  # Same as looking on a Rubiks from 6 view direction (With rotation matrices)
    MATHEMATICAL_MODE = 2  # Direct projection on each of the 6 planes

# YAML text to Enum mapping
task_type_map = {
    "SINGLE_COMPONENT": TaskType.SINGLE_COMPONENT,
    "LOCAL_CONNECTIVITY": TaskType.LOCAL_CONNECTIVITY,
    "PATCH_HOLES": TaskType.PATCH_HOLES
}

# Root Path
ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent

# Read configurations from config file
CONFIG_FILEPATH = ROOT_PATH.joinpath("configs", CONFIG_FILENAME)
with open(CONFIG_FILEPATH, 'r') as stream:
    config_data: dict = yaml.safe_load(stream)

# Read Data Configurations
DATASET_INPUT_FOLDER = config_data.get("DATASET_INPUT_FOLDER", None)
DATASET_OUTPUT_FOLDER = config_data.get("DATASET_OUTPUT_FOLDER", None)
DATA_CROP_STRIDE = config_data.get("DATA_CROP_STRIDE", 16)
DATA_CROP_SIZE = config_data.get("DATA_CROP_SIZE", 32)
LOWER_THRESHOLD_2D = config_data.get("LOWER_THRESHOLD_2D", 0.1)
UPPER_THRESHOLD_2D = config_data.get("UPPER_THRESHOLD_2D", 0.9)
TASK_TYPE = task_type_map.get(config_data.get("TASK_TYPE", "PATCH_HOLES"))
START_INDEX = config_data.get("START_INDEX", -1)
STOP_INDEX = config_data.get("STOP_INDEX", -1)
INCLUDE_2D_PROJECTIONS = config_data.get("INCLUDE_2D_PROJECTIONS", True)  # Info: If True, the 2D depth projections will be created from the 3D data.

# Read Weights Configurations
WEIGHTS_1D_PATH = config_data.get("WEIGHTS_1D_PATH", None)
WEIGHTS_2D_PATH = config_data.get("WEIGHTS_2D_PATH", None)
WEIGHTS_3D_PATH = config_data.get("WEIGHTS_3D_PATH", None)

# Read Predict Pipeline Configurations
APPLY_FUSION = config_data.get("APPLY_FUSION", True)

APPLY_INPUT_MERGE_2D = config_data.get("APPLY_INPUT_MERGE_2D", False)  # Notice: Doesn't work well with revealed occluded objects
APPLY_THRESHOLD_2D = config_data.get("APPLY_THRESHOLD_2D", True)
THRESHOLD_2D = config_data.get("THRESHOLD_2D", 0.2)  # Info: Threshold for 2D images, used to remove noise
APPLY_NOISE_FILTER_2D = config_data.get("APPLY_NOISE_FILTER_2D", False)  # Notice: Doesn't work well with revealed occluded objects
HARD_NOISE_FILTER_2D = config_data.get("HARD_NOISE_FILTER_2D", True)  # Info: True - for components_before >= components_after, False - for components_before > components_after
PREDICT_CONNECTIVITY_TYPE_2D = config_data.get("PREDICT_CONNECTIVITY_TYPE_2D", 4)

APPLY_INPUT_MERGE_3D = config_data.get("APPLY_INPUT_MERGE_3D", True)
APPLY_THRESHOLD_3D = config_data.get("APPLY_THRESHOLD_3D", True)
THRESHOLD_3D = config_data.get("THRESHOLD_3D", 0.5)  # Info: Threshold for 3D volumes, used to remove noise
APPLY_NOISE_FILTER_3D = config_data.get("APPLY_NOISE_FILTER_3D", True)
HARD_NOISE_FILTER_3D = config_data.get("HARD_NOISE_FILTER_3D", True)  # Info: True - for components_before >= components_after, False - for components_before > components_after
PREDICT_CONNECTIVITY_TYPE_3D = config_data.get("PREDICT_CONNECTIVITY_TYPE_3D", 6)

# Parse Data Configurations
if DATASET_INPUT_FOLDER is None:
    raise ValueError("DATASET_INPUT_FOLDER is not defined in the configuration file")
DATASET_OUTPUT_FOLDER = DATASET_INPUT_FOLDER if DATASET_OUTPUT_FOLDER is None else DATASET_OUTPUT_FOLDER

DATA_3D_STRIDE = (DATA_CROP_STRIDE, DATA_CROP_STRIDE, DATA_CROP_STRIDE)
DATA_3D_SIZE = (DATA_CROP_SIZE, DATA_CROP_SIZE, DATA_CROP_SIZE)
DATA_2D_SIZE = (DATA_CROP_SIZE, DATA_CROP_SIZE)
LOWER_THRESHOLD_2D *= (DATA_CROP_SIZE ** 2)
UPPER_THRESHOLD_2D *= (DATA_CROP_SIZE ** 2)

# Data Folder Paths
DATA_PATH = ROOT_PATH.joinpath("data")
DATA_CROPS_PATH = ROOT_PATH.joinpath("data_crops")
DATA_RESULTS_PATH = ROOT_PATH.joinpath("data_results")

# Dataset Folder Paths
DATASET_PATH = DATA_PATH.joinpath(DATASET_INPUT_FOLDER)
CROPS_PATH = DATA_CROPS_PATH.joinpath(DATASET_OUTPUT_FOLDER)
RESULTS_PATH = DATA_RESULTS_PATH.joinpath(DATASET_OUTPUT_FOLDER)

#########
# FLAGS #
#########

# Preparation
APPLY_LOG_FILTER = True  # Helpful to reject projections with too dense or too sparse pixels from the training loader
APPLY_MEDIAN_FILTER = False  # Sometimes helpful for PCDs to fill missing inner black pixels
APPLY_CONTINUITY_FIX_3D = True  # The continuity filter for true holes detection in 3D
APPLY_CONTINUITY_FIX_2D = False  # The continuity filter for true holes detection in 2D
BINARY_DILATION = True  # Enable the usage local scope binary dilation mask (2D - 4 directions, 3D - 6 directions)
TRAIN_CONNECTIVITY_TYPE_3D = 6
TRAIN_CONNECTIVITY_TYPE_2D = 4

# Dataset 1D
V1_1D_DATASETS = ['Trees1DV1']

# Dataset 2D
V1_2D_DATASETS = ['Trees2DV1S', 'Trees2DV1', 'Trees2DV1R']
V2_2D_DATASETS = ['Trees2DV2S', 'Trees2DV2', 'Trees2DV2R']

RANDOM_HOLES_DATASETS = ['MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'Trees2DV1S', 'Trees2DV2S']

# Dataset 3D
V1_3D_DATASETS = ['Trees3DV1', 'Trees3DV1R']
V2_3D_DATASETS = ['Trees3DV2', 'Trees3DV2R', 'Trees3DV2D', 'Trees3DV2F']
V3_3D_DATASETS = ['Trees3DV2Full']

# Views Configurations
IMAGES_6_VIEWS = ["top", "bottom", "front", "back", "left", "right"]
PROJECTION_MODE = ProjectionMode.VISUAL_MODE

###################
# DATASET - PATHS #
###################

LABELS = DATASET_PATH.joinpath("labels")  # TARGET
LABELS_COMPONENTS = DATASET_PATH.joinpath("labels_components")

PREDS = DATASET_PATH.joinpath("preds")  # INPUT
PREDS_COMPONENTS = DATASET_PATH.joinpath("preds_components")

PREDS_FIXED = DATASET_PATH.joinpath("preds_fixed")  # INPUT (Outliers removed)
PREDS_FIXED_COMPONENTS = DATASET_PATH.joinpath("preds_fixed_components")

EVALS = DATASET_PATH.joinpath("evals")
EVALS_COMPONENTS = DATASET_PATH.joinpath("evals_components")

#########################
# DATASET CROPS - PATHS #
#########################

# Data 2D - Paths

# TRAINING
LABELS_2D = CROPS_PATH.joinpath("labels_2d")  # TARGET
LABELS_COMPONENTS_2D = CROPS_PATH.joinpath("labels_components_2d")

PREDS_2D = CROPS_PATH.joinpath("preds_2d")  # INPUT
PREDS_COMPONENTS_2D = CROPS_PATH.joinpath("preds_components_2d")

PREDS_FIXED_2D = CROPS_PATH.joinpath("preds_fixed_2d")  # INPUT (Outliers removed)
PREDS_FIXED_COMPONENTS_2D = CROPS_PATH.joinpath("preds_fixed_components_2d")

PREDS_ADVANCED_FIXED_2D = CROPS_PATH.joinpath("preds_advanced_fixed_2d")  # INPUT (Continuity fixed)
PREDS_ADVANCED_FIXED_COMPONENTS_2D = CROPS_PATH.joinpath("preds_advanced_fixed_components_2d")

# EVALUATION
EVALS_2D = CROPS_PATH.joinpath("evals_2d")
EVALS_COMPONENTS_2D = CROPS_PATH.joinpath("evals_components_2d")


# Data 3D - Paths

# TRAINING
LABELS_3D = CROPS_PATH.joinpath("labels_3d")  # TARGET
LABELS_COMPONENTS_3D = CROPS_PATH.joinpath("labels_components_3d")

PREDS_3D = CROPS_PATH.joinpath("preds_3d")  # INPUT
PREDS_COMPONENTS_3D = CROPS_PATH.joinpath("preds_components_3d")

PREDS_FIXED_3D = CROPS_PATH.joinpath("preds_fixed_3d")  # INPUT (Outliers removed)
PREDS_FIXED_COMPONENTS_3D = CROPS_PATH.joinpath("preds_fixed_components_3d")

PREDS_ADVANCED_FIXED_3D = CROPS_PATH.joinpath("preds_advanced_fixed_3d")  # INPUT (Continuity fixed)
PREDS_ADVANCED_FIXED_COMPONENTS_3D = CROPS_PATH.joinpath("preds_advanced_fixed_components_3d")

# RECONSTRUCT PATHS (Info: Reconstruct 3D data by merging 6 2D depth images reprojected into 3D space)
LABELS_3D_RECONSTRUCT = CROPS_PATH.joinpath("labels_3d_reconstruct")  # INPUT (Direct repair)
PREDS_3D_RECONSTRUCT = CROPS_PATH.joinpath("preds_3d_reconstruct")  # INPUT (Direct repair)
PREDS_FIXED_3D_RECONSTRUCT = CROPS_PATH.joinpath("preds_fixed_3d_reconstruct")  # INPUT (Direct repair)
PREDS_ADVANCED_FIXED_3D_RECONSTRUCT = CROPS_PATH.joinpath("preds_advanced_fixed_3d_reconstruct")  # INPUT (Direct repair)

# FUSION PATHS (Info: Fusion of the 3D reconstructions from the 2D reprojections, with the base 3D data)
PREDS_3D_FUSION = CROPS_PATH.joinpath("preds_3d_fusion")  # INPUT (Fusion data)
PREDS_FIXED_3D_FUSION = CROPS_PATH.joinpath("preds_fixed_3d_fusion")  # INPUT (Fusion data)
PREDS_ADVANCED_FIXED_3D_FUSION = CROPS_PATH.joinpath("preds_advanced_fixed_3d_fusion")  # INPUT (Fusion data)

# EVALUATION
EVALS_3D = CROPS_PATH.joinpath("evals_3d")
EVALS_COMPONENTS_3D = CROPS_PATH.joinpath("evals_components_3d")

# LOGS
TRAIN_LOG_PATH = CROPS_PATH.joinpath("train_log.csv")
EVAL_LOG_PATH = CROPS_PATH.joinpath("eval_log.csv")

###########################
# DATASET RESULTS - PATHS #
###########################

MODELS_RESULTS_PATH = RESULTS_PATH.joinpath("models")
PREDICT_PIPELINE_RESULTS_PATH = RESULTS_PATH.joinpath("predict_pipeline")
MERGE_PIPELINE_RESULTS_PATH = RESULTS_PATH.joinpath("merge_pipeline")
VISUALIZATION_RESULTS_PATH = RESULTS_PATH.joinpath("visualization")

PREDICT_PIPELINE_DICE_CSV_FILES_PATH = PREDICT_PIPELINE_RESULTS_PATH.joinpath("csv_files")
