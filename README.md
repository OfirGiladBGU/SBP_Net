# Trees Auto Encoder


## Description:

The repository contains all the scripts that were used to get the results available on the paper: \
`Learning Thin Structure Reconstruction from Sliding-Box 2D Projections`


## Requirements:

See the following file for full guidance: [manual_requirements.txt](manual_requirements.txt)

---

## Data Setup

### How to set up a dataset (Example for Parse2022):

In case that you have a dataset of 3D input files with holes and 3D ground truth files. \
(Example: for [Parse2022](https://grand-challenge.org/forums/forum/parse2022-623/) data, we took SOTA model results 
(i.e. [MEDPSeg](https://github.com/MICLab-Unicamp/medpseg)) as our input data, along with `Parse2022` labels as our ground truth data).

1. Put the `parse2022` dataset with the `labels` and `preds` data on the path: `./data/parse2022`
   - `labels` - The 3D ground truth
   - `preds` - The 3D input
2. Create config file in `.yaml` format and put it in `configs` folder (see example: [parse2022_SC_32.yaml](configs/parse2022_SC_32.yaml)).
3. Build the `dataset_2d` (also `dataset_1d`) by running `dataset_2d_creator` script: [dataset_2d_creator.py](datasets_forge/dataset_2d_creator.py)
4. Build the `dataset_3d` by running  `dataset_3d_creator` script: [dataset_3d_creator.py](datasets_forge/dataset_3d_creator.py)
5. Notice that the following structure will be created in the `DATASET_INPUT_FOLDER`:
   - A new folder named `preds_fixed` will be created with the data from `preds` after outliers removal.
   - For each `<folder>` in [`labels`, `preds`, `preds_fixed`]:
     ```
     ./data/parse2022
     └── <folder>_components      // (Components of the 3D data)
     ```
6. Notice that the following structure will be created in the `DATASET_OUTPUT_FOLDER`:
   - For each `<folder>` in [`labels`, `preds`, `preds_fixed`, `preds_advanced_fixed`]:
     ```
     ./data_crops/parse2022
     # From 2D script:
     ├── <folder>_3d              // (3D Box)
     ├── <folder>_components_3d   // (Components of the 3D Box)
     ├── <folder>_2d              // (2D orthographic projections of the 3D Box)
     ├── <folder>_components_2d   // (Components of the 2D projections)
     # From 3D script:
     ├── <folder>_3d_reconstruct  // (2D projections reprojected back to 3D as Reprojected 3D Box)
     └── <folder>_3d_fusion       // (`labels_3d_reconstruct` Reprojected 3D Box fused with the `<folder>` 3D Box)
     ```
     **Notice:** `labels_3d_fusion` is excluded as it has no meaning (it will be equal to `labels`).

### How to generate synthetic dataset (Example for PipeForge3D):

In case you have only a dataset of 3D files, and you want to create random holes in them \
(Example: We used our own [PipeForge3D](https://github.com/OfirGiladBGU/PipeForge3D) dataset generator to create 3D models, and apply random holes on them).

1. Put the `PipeForge3D` dataset with the `originals` data on the path: `./data/PipeForge3D`
   - `originals` - The raw data, before voxelization
2. Run either of the following scripts depending on your data type:
   - [generate_3d_preds_from_mesh.py](datasets_forge/generate_3d_preds_from_mesh.py) - For mesh like data types: `_mesh.ply`, `.obj`, `.nii.gz` (annotations)
   - [generate_3d_preds_from_pcd.py](datasets_forge/generate_3d_preds_from_pcd.py) - For point cloud like data types: `_pcd.ply`, `.pcd`
3. Repeat the steps in `How to set up a dataset` (Section Above).

---

## Training:

There are 3 types of model supported in this repo:

- `1D Model` - Predicts a binary label of `True/False` if the given input contains interesting holes. 
   The model purpose is to filter out the samples that will be sent to the `2D Model`. 
- `2D Model` - Detect and fills holes on the given 2D orthographic depth projections.
- `3D Model` - Detect and fill occluded holes that couldn't be detected by the `2D Model`.


### How to train the models:

- To train the `model_1d` run `main_1d` script: [main_1d.py](main_1d.py) (Currently `NOT USED` for Full Pipeline)
- To train the `model_2d` run `main_2d` script: [main_2d.py](main_2d.py)
- To train the `model_3d` run `main_3d` script: [main_3d.py](main_3d.py) (Currently `DISABLED` for Full Pipeline)

**Notice:** all scripts are calling the generic [main_base.py](main_base.py) scripts that call the matching training 
scripts in the files, based on the `model_type` parameter in the relevant `main_{i}` script:

- [train_1d.py](trainer/train_1d.py)
- [train_2d.py](trainer/train_2d.py)
- [train_3d.py](trainer/train_3d.py)

---

## Prediction Pipeline:

The prediction pipeline is designed to process 3D data and generate predictions using the trained models. \
It consists of two main scripts: 

- [offline_pipeline.py](offline_pipeline.py) - On training/evaluation local crops data.
- [TBD] [online_pipeline.py](online_pipeline.py) - Directly on new full 3D data.

---

## Evaluation:

We evaluate the performance of the trained models on the following metrics:
- `MAE (masked)` - Mean Absolute Error
- `RMSE (masked)` - Root Mean Square Error
- `SSIM` - Structural Similarity Index
- `δ_1 (maked)` - Delta 1 (See: [Pulling Things out of Perspective](https://ieeexplore.ieee.org/document/6909413))
- `δ_2 (maked)` - Delta 2
- `δ_3 (maked)` - Delta 3
- `Dice` - Dice Score

The evaluation works only on training/evaluation local crops data, and can be run from the following script:

- [evaluation_metrics.py](evaluation_metrics.py)

#### 

---

## Predict Pipeline Flow:

The results of the prediction pipeline are saved in `./data_results/{DATASET_OUTPUT_FOLDER}/predict_pipeline`.

The flow is as follows for `preds` data (The follow for `evals` data is the same , with the relevant changes):

1. `[Disabled]` Use `model 1D` on the `preds_fixed_2d` data to filter out the samples that will be sent to the `model 2D`.
2. Use `model 2D` on the `preds_fixed_2d` data.
3. Save the results in `output_2d`.
4. Perform `fusion` on the `output_2d` data to get `input_3d` data.
5. `[Disabled]` Use model 3D on the `input_3d`.
6. Save the results in `output_3d`.
7. Run steps 1-6 for all the cubes available at `preds`.
8. Integrate all the results in `output_3d` and save them in: `./data_results/{DATASET_OUTPUT_FOLDER}/merge_pipeline`.

---

## Data dimensions info:


### Source data in `DATASET_INPUT_FOLDER`:

For `<folder>` in [`labels`, `preds`, `preds_fixed`, `evals`]: 

1. `<folder>` -> Values: binary {0, 1}, Dim: 3
2. `<folder>_components` -> Values: grayscale (0-255), Dim: 3


### Generated data in `DATASET_OUTPUT_FOLDER`:

For `<folder>` in [`labels`, `preds`, `preds_fixed`, `evals`]:

1. `<folder>_2d` -> Values: grayscale (0-255), Dim: 2
2. `<folder>_components_2d` -> Values: RGB (0-255, 0-255, 0-255), Dim: 2
3. `<folder>_3d` -> Values: binary {0, 1}, Dim: 3
4. `<folder>_components_3d` -> Values: grayscale (0-255), Dim: 3
5. `<folder>_3d_reconstruct` -> Values: binary {0, 1}, Dim: 3
6. `<folder>_3d_fusion`* -> Values: binary {0, 1}, Dim: 3

* `labels_3d_fusion` is excluded, as it has no meaning.

---

## Data generation info:

Parameters used for the data generation on the scripts: 
- [generate_3d_preds_from_mesh.py](datasets_forge/generate_3d_preds_from_mesh.py)
- [generate_3d_preds_from_pcd.py](datasets_forge/generate_3d_preds_from_pcd.py)

```
# Parse2022:
None

# PipeForge3D - OBJ:
mesh_scale = 0.25
voxel_size = 1.0

# PipeForge3D - PCD:
points_scale = 0.25
voxel_size = 1.0

# Hospital CUP:
points_scale = 25.0
voxel_size = 1.0
```

---

# Chosen Approach:

- Flow: 
   1. Crop and Project both the `3d ground truth` and `3d predicted labels`:
      1. 6 views of `2d ground truth`
      2. 6 views of `2d predicted labels`
   2. Use `model 2D` as follows (2D to 2D):
      1. Train with the `2d predicted labels` to **repair** and get `2d ground truth`
      2. Predict with the `2d predicted labels` to get the `2d fixed labels`
   3. Use direct fusion (6-2D to 3D):
      1. Fuse the 6 `2d fixed labels` with the `3d predicted labels` to get the `3d fixed labels`

---

# Available Approaches:

Given the `3d ground truth` and the `3d predicted labels`:

1. (Main Core Flow) Option 1 Flows: 
   1. Crop and Project both the `3d ground truth` and `3d predicted labels`:
      1. 6 views of `2d ground truth`
      2. 6 views of `2d predicted labels`
   2. Use `model 2D` as follows (2D to 2D):
      1. Train with the 6 `2d predicted labels` to **repair** and get 6 `2d ground truth`
      2. Predict with the 6 `2d predicted labels` to get the 6 `2d fixed labels`
   3. Use `model 3D` as follows (6-2D to 3D):
      1. Train with the 6 `2d ground truth` to **reconstruct** and get the `3d ground truth`
      2. Predict with the 6 `2d fixed labels` to get the `3d fixed labels`
      
   4. Use all the `3d fixed label` to fix the `3d predicted labels`

2. (Secondary Core Flow) Option 2 Flows: 
   1. Crop and Project both the `3d ground truth` and `3d predicted labels` to `cropped`:
      1. 6 views of `2d ground truth`
      2. 6 views of `2d predicted labels`
   2. Use `model 2D` as follows (2D to 2D):
      1. Train with the 6 `2d predicted labels` to **repair** and get 6 `2d ground truth`
      2. Predict with the 6 `2d predicted labels` to get the 6 `2d fixed labels`
      * **2 approaches available with the same data:**
         1. Option 1: work with batches of 1 view (size: (b, 1, w, h))
         2. Option 2: work with batches of 6 views (size: (b, 6, 1, w, h))
   3. Perform direct projection (using `logical or`) for all the data:
      1. Option 1:
         1. From `2d ground truth` to get the `pre-3d ground truth`
         2. From `2d predicted labels` to get the `pre-3d predicted labels`
         3. From `2d fixed labels` to get the `pre-3d fixed labels`
      2. Option 2:
         1. From `2d ground truth` to get the `pre-3d ground truth`
         2. From `2d predicted labels` to get the `pre-3d predicted labels`
         3. From `2d fixed labels` to get the `pre-3d fixed labels`
         4. Merge `3d predicted labels` and `pre-3d ground truth` to get the `fused pre-3d ground truth`
         5. Merge `3d predicted labels` and `pre-3d fixed labels` to get the `fused pre-3d fixed labels`
   4. Use `model 3D` as follows (3D to 3D):
      1. Option 1 (Fill the whole internal space):
         1. Train with the `pre-3d ground truth` to **reconstruct** and get the `3d ground truth`
         2. Predict with the `pre-3d fixed labels` to get the `3d fixed labels`
      2. Option 2 (Fill only the predicted labels):
         1. Train with the `fused pre-3d ground truth` to **reconstruct** and get the `3d ground truth`
         2. Predict with the `fused pre-3d fixed labels` to get the `3d fixed labels`
   5. Use all the `3d fixed label` to fix the `3d predicted labels`

3. (Direct Repair Flow) Option 3 Flows:
   1. Crop both the `3d ground truth` and `3d predicted labels`:
      1. `cropped 3d ground truth`
      2. `cropped 3d predicted labels`
   2. Use `model 3D` as follows (3D to 3D):
      1. Train with the `cropped 3d predicted labels` to the `cropped 3d predicted labels` to **reconstruct** and get the `cropped 3d ground truth`
      2. Predict with the `cropped 3d predicted labels` to get the `cropped 3d fixed labels`
   3. Use all the `cropped 3d fixed label` to fix the `3d predicted labels`

**Notice:** `Model 1D` doesn't appear in flows above, as it can either be used to filter out the samples that will be sent to the `model 2D` or not used at all.

---

# Sources:

## AE (As base model skeleton):

https://github.com/dariocazzani/pytorch-AE

## VGG loss:

https://github.com/crowsonkb/vgg_loss/tree/master

## Tool to visualize projections on 3D cube:

https://3dthis.com/photocube.htm

## Configuring Matplotlib Plots to Display in a Window in PyCharm

See the following question on [Stack Overflow](https://stackoverflow.com/questions/57015206/how-to-show-matplotlib-plots-in-a-window-instead-of-sciview-toolbar-in-pycharm-p).

---

# Log to myself:

Commit with big change:
11.04.2025 - configured new noise filters based on the Continuity fix 3D filters

# Fixing broken `pip` on python env:

```
python -m pip install -U --force pip
```

---

## [NOT USED] Graph Generators

https://github.com/networkx/grave

https://github.com/deyuan/random-graph-generator

https://github.com/mlimbuu/random-graph-generator

https://github.com/mlimbuu/TCGRE-graph-generator

https://github.com/connectedcompany/alph
