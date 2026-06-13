# SBP-Net: Learning Thin Structure Reconstruction with Sliding-Box Projections
<!--
[![Conference](https://img.shields.io/badge/IEEE-ICIP_2026-blue.svg)](https://2026.ieeeicip.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
-->

This is the official PyTorch implementation for the paper **"SBP-Net: Learning Thin Structure Reconstruction with Sliding-Box Projections"** (IEEE ICIP 2026).

SBP-Net is designed for the localized completion of highly imperfect, thin 3D structures (such as vascular networks and industrial pipelines). By utilizing a sliding-box traversal and 2D orthographic depth projections, our method avoids the complexities of global 3D reconstruction, effectively detecting and repairing topological gaps.

<p align="center">
   <img src="assets/combined_3d_results_new.png" alt="SBP-Net 3D teaser" /><br>
   <em>Representative 3D reconstruction results showing the SBP-Net output on thin structures from medical CT scans and industrial point clouds.</em>
</p>

---

## Method Overview

The SBP-Net pipeline operates by localizing the reconstruction task. It extracts 3D sub-volumes, projects them into 6 orthogonal 2D depth views, repairs the structures using an Attention U-Net, and perfectly fuses the fixed geometry back into the global 3D space using a logical OR operation.

<p align="center">
   <img src="assets/Flow_new.png" alt="SBP-Net method overview" />
</p>

---

## Installation

Installation includes cloning the repository, creating a virtual environment, and installing the required dependencies. Please note that this project requires **Python 3.10** and uses a manual instruction file for dependencies rather than an automated `requirements.txt`.

```bash
git clone https://github.com/OfirGiladBGU/SBP-Net.git
cd SBP-Net

# Create and activate a Python 3.10 virtual environment
python3.10 -m venv venv
source venv/bin/activate
```

Then follow the [manual_requirements.txt](manual_requirements.txt) instructions.

---

## Quick Start

Follow these steps to run our pre-trained model on the [PipeForge3D](https://github.com/OfirGiladBGU/PipeForge3D) dataset:

### 1. Download Pre-trained Weights

Access the model weights via our [Google Drive](https://drive.google.com/drive/folders/1byYa2RnqDiDiQevLBdsWS4xSF5643C_m?usp=drive_link). Navigate to the `Weights/Ex1` folder and download the checkpoint that fits your data representation:

* **Mesh:** `Network_PipeForge3DMesh_Best_LC_32_ae_2d_to_2d.pth`
* **Point Cloud (PCD):** `Network_PipeForge3DPCD_Best_LC_32_ae_2d_to_2d.pth`

Create a weights directory in the root of the repository and place the downloaded `.pth` file inside:

```bash
mkdir -p weights
# Move your downloaded .pth file into this folder

```

### 2. Prepare the Dataset

From the `Datasets` folder in the Google Drive, download the dataset corresponding to your chosen weights:

* `PipeForge3DMesh_Best.zip`
* `PipeForge3DPCD_Best.zip`

Create a data directory, place the zip file inside, and extract its contents:

```bash
mkdir -p data
# Extract the zip file into the ./data folder

```

### 3. Set Up Evaluation Data

Navigate to your extracted dataset folder (inside `./data`) and create an `evals` directory.

Place your input `.npy` files (representing the incomplete structures or holes) into this `evals` directory. If you just want to test the pipeline, you can use one of the pre-existing examples provided in the dataset's `preds_fixed` folder.

> **Note:** The model was trained on the first 90% of the examples in this dataset.

### 4. Update Configuration

Open `configs/configs_parser.py` in your text editor. Under the `Ex1` section, uncomment the configuration line that matches your downloaded weights. For example:

```python
CONFIG_FILENAME = "experiment1/PipeForge3DPCD_Best_LC_32.yaml"

```

### 5. Run the Pipeline

Execute the pipeline script to generate the predictions:

```bash
python online_pipeline.py

```

The final model results will be saved in the following directory:
`./data_results/<dataset-name>/merge_pipeline`

### 6. Visualization (Optional)

To inspect the resulting `.npy` files, you can use the included visualization tool:

```bash
python datasets_visualize/dataset_visulalization.py

```

This tool supports an **interactive mode** for live viewing, or you can configure it to save the rendered visualizations locally to:
`./data_results/<dataset-name>/visualization`

---

## Data Setup

Our pipeline supports multiple data representations (Voxel Grids, Meshes, and Point Clouds). Below is an example of setting up a medical dataset (e.g., [Parse2022](https://parse2022.grand-challenge.org/)).

1. Place your dataset in `./data/parse2022` with the following structure:
   - `labels`: The 3D ground truth (e.g., Parse2022 Label Segmentation).
   - `preds`: The incomplete/damaged 3D input (e.g., SOTA Predictions - [MEDPSeg](https://github.com/MICLab-Unicamp/medpseg)).

   ![data_dir_example](/assets/data_dir_example.png)

2. Select a configuration file in the `configs/` folder (e.g., `parse2022_SC_32.yaml`) and update the `CONFIG_FILENAME` field inside `configs/configs_parser.py`.
3. Build the localized training crops by running:
   ```bash
   python datasets_forge/dataset_2d_creator.py
   ```
   *(Optional)* If you are running 3D-to-3D baseline experiments, also run `dataset_3d_creator.py`.

### Synthetic Generation (PipeForge3D)
If you are generating synthetic holes in complete structures (e.g., using [PipeForge3D](https://github.com/OfirGiladBGU/PipeForge3D)), place the raw data in `./data/PipeForge3D/originals` and run the appropriate generator before proceeding to the crop creation:
- For Meshes/NIfTI: `python datasets_forge/generate_3d_preds_from_mesh.py`
- For Point Clouds: `python datasets_forge/generate_3d_preds_from_pcd.py`

<p align="center">
   <img src="assets/combined_2d_results_new.png" alt="SBP-Net 2D teaser" /><br>
   <em>Examples of 2D orthographic projections generated from the sliding-box crops.</em>
</p>

---

## Training

The repository supports training for 2 model variations. Training configurations are dynamically loaded from your selected `.yaml` file.

- **2D Model (2D Projection Completion):** Detects and fills holes within the 2D orthographic depth projections.
  ```bash
  python main_2d.py
  ```
- **3D Model (3D Volume Completion):** Performs direct 3D-to-3D volumetric repair.
  ```bash
  python main_3d.py
  ```

> **Note:** The core **SBP-Net** approach relies exclusively on the **2D Model**. However, the **3D Model** is used for baseline comparisons and future research directions.

---

## Inference & Evaluation

The prediction pipeline integrates the localized repairs back into full 3D models. Outputs are saved to `./data_results/<DATASET_NAME>/predict_pipeline`.

**To run predictions on training/evaluation crops:**
```bash
python offline_pipeline.py
```

**To run predictions directly on new, full 3D volumes:**
```bash
python online_pipeline.py
```

**Evaluation:**
To compute quantitative results (MAE, RMSE, SSIM, Dice, and Chamfer/Hausdorff distances) against ground-truth crops:
```bash
python evaluation_metrics.py
```

---

## Acknowledgments

This code builds upon and compares against several excellent works in the 3D vision community. We thank the authors of the following repositories:
- [Unet3D](https://github.com/wolny/pytorch-3dunet)
- [3D-RecGAN](https://github.com/Yang7879/3D-RecGAN)
- [Conv ONet](https://github.com/autonomousvision/convolutional_occupancy_networks)
- [OReX](https://github.com/haimsaw/OReX)
- [DeepCA](https://github.com/WangStephen/DeepCA)

---

## Citation

If you find this code or our methodology useful in your research, please consider citing our paper:
```bibtex
@inproceedings{gilad2026SBPNet,
    title        = {SBP-Net: Learning Thin Structure Reconstruction with Sliding-Box Projections},
    author       = {Gilad, Ofir and Sharf, Andrei},
    booktitle    = {ICIP},
    year         = {2026},
    organization = {IEEE},
}
```
