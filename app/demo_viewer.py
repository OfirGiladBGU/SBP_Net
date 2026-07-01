"""
SBP-Net Interactive Show & Tell Viewer
======================================

Live single-box reconstruction demo for ICIP Show & Tell.

Idea
----
The full pipeline (crop -> project -> per-cube predict -> merge) is too heavy to
run end-to-end interactively on a full volume. But the heavy part is the
per-volume crop + 2D projection + density bookkeeping, NOT the tiny network.

So we split the cost:
  * STARTUP (once, slow): load the volume, run `prepare_2d_projections_and_3d_cubes`
    to crop every cube and precompute its 6 projections. Cache the result.
  * ON CLICK (cheap, live): map the clicked voxel to its covering cube, run the
    REAL `single_predict` on that one cube, OR the reconstructed cube back into
    the displayed volume exactly like `full_merge` does, and recolor it.

This is honest live inference: the attendee picks a region, your actual model
reconstructs it on the spot, on a laptop CPU.

Usage
-----
    python demo_viewer.py --volume path/to/structure.npy
    python demo_viewer.py --volume path/to/PA000005.nii.gz --no-noise-filter

Then in the window: hover the structure and press 'P' to pick a point
(or use the box widget, see --pick-mode). The cube covering that point is
reconstructed live.

Requires: pyvista  (pip install pyvista)
Everything else is your existing repo + its deps.
"""

import argparse
import pathlib
import sys
import datetime
import types

import numpy as np
import torch

# --- make the repo importable -------------------------------------------------
# Adjust REPO_ROOT if you place this script somewhere other than the repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from configs.configs_parser import *  # DATA_3D_SIZE, DATA_3D_STRIDE, IMAGES_6_VIEWS, EVALS, etc.
from datasets.dataset_utils import convert_data_file_to_numpy, get_data_file_stem
from evaluator.predict_pipeline import init_pipeline_models, single_predict
from online_pipeline import prepare_2d_projections_and_3d_cubes

try:
    import pyvista as pv
except ImportError:
    raise SystemExit("This viewer needs PyVista. Install it with:  pip install pyvista")


# =============================================================================
# 1. In-memory result capture
# =============================================================================
# `single_predict` exports the reconstructed cube to disk via export_output_3d
# instead of returning it. For a live viewer we want the array back in memory.
# Rather than editing your pipeline, we monkeypatch the export function in the
# predict_pipeline module to ALSO stash the array in a dict we can read.
#
# If you prefer, you can instead modify single_predict to `return data_3d_output`
# and skip this whole section.

import evaluator.predict_pipeline as pp

_LAST_RECONSTRUCTION = {}


def _capture_export_output_3d(data_3d_stem, data_3d_filepath, data_3d_output):
    """Drop-in replacement that captures the output instead of writing a file."""
    arr = data_3d_output.detach().cpu().numpy()
    arr = np.squeeze(arr)
    _LAST_RECONSTRUCTION[data_3d_stem] = (arr > 0.5).astype(np.uint8)


pp.export_output_3d = _capture_export_output_3d


# =============================================================================
# 2. Build a minimal args namespace (mirrors online_pipeline.py's __main__)
# =============================================================================
def build_args(use_noise_filter: bool = True):
    args = types.SimpleNamespace()
    # Paper 2D-only config (matches online_pipeline.py "Paper config")
    args.model_2d = "ae_2d_to_2d"
    args.input_size_model_2d = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])
    args.model_3d = ""
    args.input_size_model_3d = (1, DATA_3D_SIZE[0], DATA_3D_SIZE[1], DATA_3D_SIZE[2])

    args.run_2d_flow = True
    args.run_3d_flow = True
    args.export_2d = False
    args.export_3d = True          # stays True so our captured exporter fires
    args.parallel_predict = False  # single cube at a time; no threading needed

    args.seed = 42
    args.mode = "online"
    args.no_cuda = True            # CPU is fine; net is tiny
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    return args


# =============================================================================
# 3. Startup: heavy work done ONCE
# =============================================================================
class DemoState:
    """Holds everything precomputed at startup so clicks stay cheap."""

    def __init__(self, volume_path: str, args):
        self.args = args
        self.volume_path = volume_path

        print("[startup] Loading volume...")
        self.volume = convert_data_file_to_numpy(data_filepath=volume_path)
        self.volume = (self.volume > 0).astype(np.uint8)
        # Live display buffer: we OR reconstructions into this.
        self.display = self.volume.copy()
        self.added = np.zeros_like(self.volume, dtype=np.uint8)  # newly filled voxels

        print("[startup] Initializing models...")
        init_pipeline_models(args=args)

        print("[startup] Cropping cubes + precomputing projections (the slow part)...")
        t0 = datetime.datetime.now()
        # NOTE: prepare_2d_projections_and_3d_cubes takes (input_filepath, input_folder).
        # input_folder is only used to compute the stem; the parent dir works fine.
        input_folder = str(pathlib.Path(volume_path).parent)
        self.log_data, self.projections_data = prepare_2d_projections_and_3d_cubes(
            input_filepath=volume_path,
            input_folder=input_folder,
        )
        dt = datetime.datetime.now() - t0
        print(f"[startup] Done. {len(self.projections_data)} cubes cached in {dt}.")

        # Precompute cube centers for fast click->cube lookup.
        col0 = self.log_data.columns[0]
        self._stems = []
        centers = []
        boxes = []
        for _, row in self.log_data.iterrows():
            stem = str(row[col0])
            if stem not in self.projections_data:
                continue  # was filtered out by density check
            sx, ex = int(row["start_x"]), int(row["end_x"])
            sy, ey = int(row["start_y"]), int(row["end_y"])
            sz, ez = int(row["start_z"]), int(row["end_z"])
            self._stems.append(stem)
            centers.append(((sx + ex) / 2, (sy + ey) / 2, (sz + ez) / 2))
            boxes.append((sx, ex, sy, ey, sz, ez))
        self._centers = np.array(centers) if centers else np.empty((0, 3))
        self._boxes = boxes

    # ---- click -> cube -------------------------------------------------------
    def cube_for_point(self, xyz):
        """Return (stem, box) for the cube best covering a clicked voxel coord."""
        if len(self._stems) == 0:
            return None, None
        x, y, z = xyz
        # Prefer cubes that actually contain the point; among them (overlap!),
        # pick the one whose center is nearest -> most centered view of region.
        containing = [
            i for i, (sx, ex, sy, ey, sz, ez) in enumerate(self._boxes)
            if sx <= x < ex and sy <= y < ey and sz <= z < ez
        ]
        candidates = containing if containing else range(len(self._stems))
        cand_centers = self._centers[list(candidates)]
        d = np.linalg.norm(cand_centers - np.array([x, y, z]), axis=1)
        best = list(candidates)[int(np.argmin(d))]
        return self._stems[best], self._boxes[best]

    # ---- live reconstruction of ONE cube ------------------------------------
    def reconstruct_cube(self, stem, box):
        """Run the REAL single_predict on one cube; OR result into display."""
        _LAST_RECONSTRUCTION.pop(stem, None)
        t0 = datetime.datetime.now()
        single_predict(
            args=self.args,
            data_3d_filepath=stem,          # online mode keys off the stem
            projections_data=self.projections_data,
            log_data=self.log_data,
            enable_debug=False,
            run_2d_flow=self.args.run_2d_flow,
            run_3d_flow=self.args.run_3d_flow,
            export_2d=False,
            export_3d=True,                 # triggers our capture hook
        )
        dt = (datetime.datetime.now() - t0).total_seconds()

        recon = _LAST_RECONSTRUCTION.get(stem)
        if recon is None:
            print(f"[click] No output captured for {stem}.")
            return 0, dt

        sx, ex, sy, ey, sz, ez = box
        size_x, size_y, size_z = ex - sx, ey - sy, ez - sz
        region = recon[:size_x, :size_y, :size_z]

        before = self.display[sx:ex, sy:ey, sz:ez].copy()
        merged = np.logical_or(before, region).astype(np.uint8)
        newly = ((merged == 1) & (before == 0)).astype(np.uint8)

        self.display[sx:ex, sy:ey, sz:ez] = merged
        self.added[sx:ex, sy:ey, sz:ez] = np.logical_or(
            self.added[sx:ex, sy:ey, sz:ez], newly
        ).astype(np.uint8)

        n_new = int(newly.sum())
        print(f"[click] cube {stem}: +{n_new} voxels in {dt:.2f}s")
        return n_new, dt


# =============================================================================
# 4. PyVista UI
# =============================================================================
def voxels_to_points(vol, value=1):
    pts = np.argwhere(vol == value).astype(np.float32)
    return pts


class Viewer:
    def __init__(self, state: DemoState, point_size=4.0):
        self.state = state
        self.point_size = point_size
        self.plotter = pv.Plotter(window_size=(1100, 850))
        self.plotter.set_background("white")
        self._orig_actor = None
        self._added_actor = None
        self._box_actor = None
        self._draw_all()

        # Picking: press 'P' over the structure to pick a point.
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            show_message=False,
            use_picker=True,
            show_point=False,
        )
        self.plotter.add_text(
            "Hover the structure and press P to reconstruct that region",
            font_size=12, color="black", name="help",
        )
        self.plotter.add_text("", font_size=11, color="darkgreen",
                              position="lower_left", name="status")

    def _draw_all(self):
        orig_pts = voxels_to_points(self.state.volume, 1)
        if self._orig_actor is not None:
            self.plotter.remove_actor(self._orig_actor)
        self._orig_actor = self.plotter.add_points(
            orig_pts, color="#3b6ea5", point_size=self.point_size,
            render_points_as_spheres=True, name="orig",
        )
        self._refresh_added()

    def _refresh_added(self):
        added_pts = voxels_to_points(self.state.added, 1)
        if self._added_actor is not None:
            self.plotter.remove_actor(self._added_actor)
            self._added_actor = None
        if len(added_pts) > 0:
            self._added_actor = self.plotter.add_points(
                added_pts, color="#e8543f", point_size=self.point_size + 2,
                render_points_as_spheres=True, name="added",
            )

    def _show_box(self, box):
        sx, ex, sy, ey, sz, ez = box
        if self._box_actor is not None:
            self.plotter.remove_actor(self._box_actor)
        b = pv.Box(bounds=(sx, ex, sy, ey, sz, ez))
        self._box_actor = self.plotter.add_mesh(
            b, style="wireframe", color="#e8543f", line_width=3, name="box",
        )

    def _on_pick(self, point, *args):
        if point is None:
            return
        stem, box = self.state.cube_for_point(point)
        if stem is None:
            return
        self._show_box(box)
        self.plotter.add_text(f"Reconstructing {stem}...", font_size=11,
                              color="black", position="lower_left", name="status")
        self.plotter.render()

        n_new, dt = self.state.reconstruct_cube(stem, box)
        self._refresh_added()
        self.plotter.add_text(
            f"Cube {stem}:  +{n_new} voxels filled  ({dt:.2f}s, live inference)",
            font_size=11, color="darkgreen", position="lower_left", name="status",
        )
        self.plotter.render()

    def run(self):
        self.plotter.show()


# =============================================================================
def main():
    # VOL_PATH = r"D:\AllProjects\PycharmProjects\TreesAutoEncoder\data\PipeForge3DMesh_Best\eval\50.npy"
    VOL_PATH =  r"D:\AllProjects\PycharmProjects\TreesAutoEncoder\data\parse2022_32\eval\PA000317_vessel.nii.gz"
    ap = argparse.ArgumentParser(description="SBP-Net interactive Show & Tell viewer")
    ap.add_argument("--volume", help="Path to a .npy / .nii.gz volume", default=VOL_PATH)
    ap.add_argument("--point-size", type=float, default=4.0)
    args_cli = ap.parse_args()

    args = build_args(use_noise_filter=not args_cli.no_noise_filter)

    state = DemoState(volume_path=args_cli.volume, args=args)
    Viewer(state, point_size=args_cli.point_size).run()


if __name__ == "__main__":
    main()
