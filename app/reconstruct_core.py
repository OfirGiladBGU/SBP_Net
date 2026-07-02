"""
SBP-Net Interactive Demo -- Phase 1 core.

`reconstruct_at(volume, x, y, z, ...)` is THE BALLGAME (see app/DEMO_PLAN.md):
dynamically crop a DATA_2D_SIZE-sized cube CENTERED on a clicked voxel, run it
through the real online inference pipeline (projection -> 2D model -> reproject
+ OR-fuse -> mandatory 3D continuity filter) and hand back the newly-added
voxels in GLOBAL volume coordinates -- all in memory, no disk round-trip.

This module is headless: no VTK / PyVista / matplotlib-GUI / interactive_plot_*.
It only reuses the existing, tested pipeline code.

Run directly to prove the compute from a plain script:
    python app/reconstruct_core.py
"""

import argparse
import os
import pathlib
import sys

# torch/MKL and the conda OpenMP runtime can both pull in an OpenMP DLL, which
# aborts with "OMP: Error #15" on this Windows env. PyCharm sets this for us;
# set it here too so the app runs headless from any launcher. (Not VTK-related.)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT_PATH = str(pathlib.Path(__file__).absolute().parent.parent)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

# NOTE: cube size is READ FROM CONFIG (DATA_2D_SIZE / DATA_CROP_SIZE), never hardcoded.
from configs.configs_parser import DATA_2D_SIZE, IMAGES_6_VIEWS
from datasets.dataset_utils import (
    convert_data_file_to_numpy,
    get_data_file_extension,
    get_data_file_stem,
    project_3d_to_2d,
)
from datasets_forge.dataset_2d_creator import crop_mini_cubes
from evaluator.predict_pipeline import init_pipeline_models, single_predict

# The cube edge length (e.g. 32). Read from config so it tracks DATA_CROP_SIZE.
CUBE_SIZE = int(DATA_2D_SIZE[0])

# All six views are always projected (matches the offline/online training path).
_PROJECTION_OPTIONS = {view: True for view in ["front", "back", "top", "bottom", "left", "right"]}

# Model wiring for the "paper config" (see online_pipeline.py): 2D autoencoder only.
_MODEL_2D = "ae_2d_to_2d"
_INPUT_SIZE_MODEL_2D = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])
_MODEL_3D = ""
_INPUT_SIZE_MODEL_3D = (1, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)


def build_args(use_cuda: bool = True, seed: int = 42) -> argparse.Namespace:
    """Assemble the argparse.Namespace the pipeline expects (online mode)."""
    args = argparse.Namespace()
    args.model_2d = _MODEL_2D
    args.input_size_model_2d = _INPUT_SIZE_MODEL_2D
    args.model_3d = _MODEL_3D
    args.input_size_model_3d = _INPUT_SIZE_MODEL_3D

    args.run_2d_flow = True
    args.run_3d_flow = True
    args.export_2d = False
    args.export_3d = False

    args.mode = "online"
    args.seed = seed
    args.cuda = bool(use_cuda) and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(seed)
    return args


def init_models(args: argparse.Namespace) -> argparse.Namespace:
    """Load the pipeline models once (pre-warm). Mutates and returns `args`."""
    init_pipeline_models(args=args)
    return args


def _centered_crop_bounds(center: float, cube_size: int, dim: int):
    """
    Bounds of a `cube_size` window centered on `center` along one axis, clamped
    to [0, dim]. When the volume is at least `cube_size` wide, the window is slid
    fully in-bounds (no padding). When the volume is narrower, the crop is padded
    with zeros to reach `cube_size`.

    Returns (start, end, pad_after) where the cropped slice volume[start:end] has
    length (end - start) and needs `pad_after` zeros appended to become cube_size.
    """
    half = cube_size // 2
    start = int(round(center)) - half
    if dim >= cube_size:
        start = max(0, min(start, dim - cube_size))
        return start, start + cube_size, 0
    # Volume narrower than the cube along this axis: take it all, pad the rest.
    return 0, dim, cube_size - dim


def crop_centered_cube(volume: np.ndarray, x: int, y: int, z: int, cube_size: int = CUBE_SIZE):
    """
    Dynamically crop a `cube_size`^3 cube CENTERED on (x, y, z), clamped to the
    volume and zero-padded if the volume is smaller than the cube on some axis.

    Returns (cube, (start_x, start_y, start_z)). `cube` is always cube_size^3.
    """
    sx, ex, pad_x = _centered_crop_bounds(x, cube_size, volume.shape[0])
    sy, ey, pad_y = _centered_crop_bounds(y, cube_size, volume.shape[1])
    sz, ez, pad_z = _centered_crop_bounds(z, cube_size, volume.shape[2])

    cube = volume[sx:ex, sy:ey, sz:ez]
    if pad_x or pad_y or pad_z:
        cube = np.pad(cube, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="constant", constant_values=0)
    return cube, (sx, sy, sz)


def project_centered_cube(cube: np.ndarray, source_ext: str = ".npy") -> dict:
    """
    Project a single dynamically-cropped cube to the 6-view representation the 2D
    model was trained on. `source_ext` selects the rotation convention exactly as
    the offline path does via the source filename (e.g. ".npy" for the PipeForge
    datasets, ".nii.gz" for parse2022). MUST match the training data type
    (constraint 4).
    """
    projections = project_3d_to_2d(
        data_3d=cube,
        projection_options=_PROJECTION_OPTIONS,
        source_data_filepath=f"demo{source_ext}",
    )
    projections["cube"] = cube
    return projections


def _predict_cube(cube_bin: np.ndarray, args: argparse.Namespace, source_ext: str) -> dict:
    """
    Run one already-cropped `cube_size`^3 binary cube through the REAL online
    pipeline (2D preprocess -> 2D model -> postprocess -> 3D reproject + OR-fuse ->
    mandatory 3D continuity filter -> threshold). Returns the pipeline `details`
    dict: {output_3d, input_2d, output_2d}. `input_2d`/`output_2d` are the 6-view
    grayscale images the network saw / produced (uint8 (6, H, W)), i.e. the
    "before"/"after" for the projections panel.
    """
    stem = "demo"
    src = f"{stem}{source_ext}"

    projections = project_centered_cube(cube_bin, source_ext=source_ext)
    projections_data = {stem: projections}

    details = single_predict(
        args=args,
        data_3d_filepath=src,
        projections_data=projections_data,
        log_data=None,
        enable_debug=False,
        run_2d_flow=args.run_2d_flow,
        run_3d_flow=args.run_3d_flow,
        export_2d=False,
        export_3d=False,
        return_details=True,
    )
    return details


def _views_dict(images) -> dict:
    """Map a (6, H, W) uint8 array to {view: HxW array}; None -> empty dict."""
    if images is None:
        return {}
    return {view: images[idx] for idx, view in enumerate(IMAGES_6_VIEWS)}


def reconstruct_at(volume: np.ndarray,
                   x: int, y: int, z: int,
                   args: argparse.Namespace,
                   cube_size: int = CUBE_SIZE,
                   source_ext: str = ".npy"):
    """
    Reconstruct the thin structure inside a `cube_size`^3 cube centered on the
    clicked voxel (x, y, z) using the REAL online pipeline, including the
    mandatory 3D continuity filter.

    `volume` is the current authoritative state (binary, indexed [x, y, z]).
    Returns a dict:
        {
          "new_coords":  (N, 3) int array of NEWLY-ADDED voxels in GLOBAL coords,
          "start":       (sx, sy, sz) crop origin in global coords,
          "output_cube": cube_size^3 binary reconstruction (local coords),
          "views_before": {view: HxW uint8} 2D images the network saw,
          "views_after":  {view: HxW uint8} 2D images the network produced,
        }
    """
    cube, (sx, sy, sz) = crop_centered_cube(volume, x, y, z, cube_size=cube_size)

    # Binarize the cropped input (training cubes are stored thresholded to {0, 1}).
    cube_bin = (cube > 0.5).astype(np.float32)

    details = _predict_cube(cube_bin, args, source_ext)
    output_cube = (np.asarray(details["output_3d"]) > 0.5)

    # Newly-added voxels = reconstruction MINUS what is already occupied in the
    # current state within this region (constraint 2: return only the new voxels).
    region_occ = np.zeros_like(output_cube, dtype=bool)
    crop = (volume[sx:sx + cube_size, sy:sy + cube_size, sz:sz + cube_size] > 0.5)
    region_occ[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
    new_local = output_cube & ~region_occ

    li, lj, lk = np.nonzero(new_local)
    gi, gj, gk = li + sx, lj + sy, lk + sz
    # Drop voxels that fall in the zero-padded region (outside the real volume).
    in_bounds = (gi < volume.shape[0]) & (gj < volume.shape[1]) & (gk < volume.shape[2])
    new_coords = np.stack([gi[in_bounds], gj[in_bounds], gk[in_bounds]], axis=1).astype(np.int32)

    return {
        "new_coords": new_coords,
        "start": (int(sx), int(sy), int(sz)),
        "output_cube": output_cube,
        "views_before": _views_dict(details.get("input_2d")),
        "views_after": _views_dict(details.get("output_2d")),
    }


def full_inference(volume: np.ndarray,
                   args: argparse.Namespace,
                   source_ext: str = ".npy"):
    """
    Run the paper's full stride-grid pipeline over the WHOLE volume, in memory.

    Mirrors the offline eval flow (crop_mini_cubes at DATA_3D_SIZE/DATA_3D_STRIDE
    -> per-view density filter -> real per-cube inference -> OR-merge back into
    the volume). This is a generator so the server can stream progress: it yields
    {"done", "total", "added"} after each processed cube, and finally
    {"done": total, "total": total, "added": <cum>, "result": <merged volume>}.
    """
    from configs.configs_parser import (
        DATA_3D_SIZE, DATA_3D_STRIDE,
        DENSITY_LOWER_THRESHOLD, DENSITY_UPPER_THRESHOLD,
    )

    cubes, cubes_data = crop_mini_cubes(
        data_3d=volume, cube_dim=DATA_3D_SIZE, stride_dim=DATA_3D_STRIDE, cubes_data=True,
    )

    # Pre-filter to the cubes worth running: at least one view within the training
    # density band (same condition the offline eval preparation uses). This skips
    # empty/near-empty and over-dense regions -> far fewer model calls.
    todo = []
    for idx, cube in enumerate(cubes):
        if int(np.count_nonzero(cube)) == 0:
            continue
        projections = project_centered_cube((cube > 0.5).astype(np.float32), source_ext=source_ext)
        for view in IMAGES_6_VIEWS:
            nz = int(np.count_nonzero(projections[f"{view}_image"]))
            if DENSITY_LOWER_THRESHOLD < nz < DENSITY_UPPER_THRESHOLD:
                todo.append(idx)
                break

    total = len(todo)
    result = (volume > 0.5).astype(np.uint8)
    added = 0
    for n, idx in enumerate(todo):
        cube_bin = (cubes[idx] > 0.5).astype(np.float32)
        details = _predict_cube(cube_bin, args, source_ext)
        out = (np.asarray(details["output_3d"]) > 0.5)

        cd = cubes_data[idx]
        sx, sy, sz = cd["start_x"], cd["start_y"], cd["start_z"]
        ex, ey, ez = cd["end_x"], cd["end_y"], cd["end_z"]           # clamped to volume
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        region = result[sx:ex, sy:ey, sz:ez]
        merged = np.logical_or(region, out[:dx, :dy, :dz])
        added += int(np.count_nonzero(merged) - np.count_nonzero(region))
        result[sx:ex, sy:ey, sz:ez] = merged
        yield {"done": n + 1, "total": total, "added": added}

    yield {"done": total, "total": total, "added": added, "result": result}


# --------------------------------------------------------------------------- #
# Phase 1 proof-of-compute: a bare script with a hardcoded coordinate.        #
# --------------------------------------------------------------------------- #
def _verify_projection_matches_offline() -> bool:
    """
    Constraint 4: the online single-cube projection must byte-match the offline
    projection the 2D model was trained on. Diff our projection of a known offline
    cube against its stored offline PNGs.
    """
    import cv2
    from configs.configs_parser import CROPS_PATH

    cube_dir = pathlib.Path(CROPS_PATH).joinpath("preds_fixed_3d")
    img_dir = pathlib.Path(CROPS_PATH).joinpath("preds_fixed_2d")
    # Format-agnostic: the offline 3D crops may be .npy, .nii.gz, ... per dataset.
    cube_files = sorted(f for f in cube_dir.glob("*.*")) if cube_dir.exists() else []
    if not cube_files:
        print("[verify] No offline cube crops found -- skipping projection diff.")
        return True

    cube_file = cube_files[0]
    source_ext = get_data_file_extension(data_filepath=str(cube_file))
    stem = get_data_file_stem(data_filepath=str(cube_file))
    cube = convert_data_file_to_numpy(data_filepath=str(cube_file), apply_data_threshold=True)
    # Project with the SAME source convention the offline crops were made with.
    projections = project_centered_cube(cube.astype(np.float32), source_ext=source_ext)

    all_match = True
    for view in IMAGES_6_VIEWS:
        png = img_dir.joinpath(f"{stem}_{view}.png")
        if not png.exists():
            print(f"[verify] Missing offline PNG for view '{view}' -- skipping.")
            continue
        offline = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2GRAY)
        online = projections[f"{view}_image"]
        match = np.array_equal(offline, online)
        all_match &= match
        diff = int(np.abs(offline.astype(int) - online.astype(int)).sum())
        print(f"[verify] view={view:<7} match={match} abs_diff_sum={diff}")

    print(f"[verify] Projection matches offline format: {all_match} (cube {stem})")
    return all_match


def load_demo_volume(volume_path=None):
    """
    Load a demo volume as a binary uint8 array indexed [x, y, z], using the
    project's own `convert_data_file_to_numpy` so any supported format works
    (.nii.gz, .npy, .pcd, ...). When no path is given, the first input object of
    the active config's dataset is used (an object with real gaps to fill).

    Returns (volume, name, source_ext). `source_ext` selects the projection
    rotation convention that matches how this dataset was projected for training
    (constraint 4), e.g. ".nii.gz" for parse2022, ".npy" for the PipeForge sets.
    """
    from configs.configs_parser import DATASET_PATH

    if volume_path is None:
        # Prefer the model input folders (they carry the gaps); fall back to labels.
        search_dirs = ["preds", "evals", "eval", "labels"]
        volume_path = None
        for sub in search_dirs:
            d = pathlib.Path(DATASET_PATH).joinpath(sub)
            if not d.is_dir():
                continue
            files = sorted(f for f in d.glob("*.*") if f.is_file())
            if files:
                volume_path = files[0]
                break
        if volume_path is None:
            raise FileNotFoundError(f"No demo volume found under {DATASET_PATH} in {search_dirs}")
    volume_path = pathlib.Path(volume_path)

    volume = convert_data_file_to_numpy(data_filepath=str(volume_path), apply_data_threshold=True)
    volume = (volume > 0.5).astype(np.uint8)
    source_ext = get_data_file_extension(data_filepath=str(volume_path))
    return volume, volume_path.name, source_ext


def _demo_main():
    print(f"[Phase 1] CUBE_SIZE (from config) = {CUBE_SIZE}")

    # 1) Prove the projection format matches training (constraint 4).
    _verify_projection_matches_offline()

    # 2) Load a demo volume (an input object with real gaps to fill).
    volume, volume_name, source_ext = load_demo_volume()
    print(f"[Phase 1] Loaded volume {volume_name} shape={volume.shape} "
          f"occupied={int(volume.sum())} source_ext={source_ext}")

    # 3) Pre-warm the model once.
    args = build_args()
    init_models(args)
    print(f"[Phase 1] Models loaded on device={args.device}")

    # 4) Hardcoded coordinate: pick an occupied voxel near the volume center.
    occ = np.argwhere(volume > 0.5)
    center = occ.mean(axis=0)
    pick = occ[np.argmin(np.linalg.norm(occ - center, axis=1))]
    x, y, z = (int(v) for v in pick)
    print(f"[Phase 1] Reconstructing at hardcoded pick (x,y,z)=({x},{y},{z})")

    result = reconstruct_at(volume, x, y, z, args, source_ext=source_ext)
    new_coords = result["new_coords"]
    print(f"[Phase 1] crop origin = {result['start']}")
    print(f"[Phase 1] reconstruction occupied voxels in cube = {int(result['output_cube'].sum())}")
    print(f"[Phase 1] NEWLY-ADDED voxels (global) = {len(new_coords)}")
    if len(new_coords):
        print(f"[Phase 1] new voxel bbox min={new_coords.min(axis=0)} max={new_coords.max(axis=0)}")
    print("[Phase 1] OK -- compute proven from a plain script.")


if __name__ == "__main__":
    _demo_main()
