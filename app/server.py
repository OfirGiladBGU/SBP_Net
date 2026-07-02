"""
SBP-Net Interactive Demo -- Phase 2 headless local server (Flask).

The backend owns the authoritative volume state (constraint 2): it loads the
volume + model once at startup (pre-warm), crops every reconstruction from the
*current* state, ORs the new result in, and returns only the newly-added voxels.
A single-flight lock serializes reconstructions (constraint 3). Everything runs
locally and headless -- no VTK, no rendering in Python.

Run:
    conda run -n TAE --no-capture-output python app/server.py
    # then open http://127.0.0.1:5000/

All heavy lifting lives in app/reconstruct_core.reconstruct_at.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see reconstruct_core.py

import argparse
import base64
import json
import pathlib
import sys
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

ROOT_PATH = str(pathlib.Path(__file__).absolute().parent.parent)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from app.reconstruct_core import (
    CUBE_SIZE,
    build_args,
    full_inference,
    init_models,
    load_demo_volume,
    reconstruct_at,
)


def _png_data_url(image: np.ndarray, upscale: int = 4) -> str:
    """Encode a small grayscale HxW uint8 image as a base64 PNG data URL.

    Nearest-neighbour upscaled so the 32x32 projections are legible in the panel.
    """
    img = np.ascontiguousarray(image.astype(np.uint8))
    if upscale > 1:
        img = cv2.resize(img, (img.shape[1] * upscale, img.shape[0] * upscale),
                         interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_views(views: dict) -> dict:
    """{view: HxW uint8} -> {view: png-data-url}."""
    return {view: _png_data_url(img) for view, img in views.items()}

STATIC_DIR = pathlib.Path(__file__).absolute().parent.joinpath("static")

app = Flask(__name__, static_folder=None)


class DemoState:
    """Authoritative volume state held in memory by the backend."""

    def __init__(self, volume: np.ndarray, name: str, args, source_ext: str = ".npy"):
        self.name = name
        self.args = args
        self.source_ext = source_ext                          # projection convention (constraint 4)
        self.original = (volume > 0.5).astype(np.uint8)      # immutable reference
        self.volume = self.original.copy()                    # current state (mutated)
        self.reconstructed = np.zeros_like(self.original)     # voxels added by the model
        # The lock IS the concurrency guard: one reconstruction in flight (constraint 3).
        self.lock = threading.Lock()
        # Set by POST /full_inference/cancel to stop an in-flight full run early.
        self.cancel_event = threading.Event()

    def occupied_coords(self, mask: np.ndarray):
        """Flat [x0,y0,z0,x1,y1,z1,...] int list of occupied voxels in `mask`."""
        coords = np.argwhere(mask > 0.5).astype(np.int32)
        return coords.reshape(-1).tolist()

    def snapshot(self) -> dict:
        original_only = (self.original > 0.5) & (self.reconstructed <= 0.5)
        return {
            "name": self.name,
            "shape": [int(s) for s in self.volume.shape],
            "cube_size": int(CUBE_SIZE),
            "original": self.occupied_coords(original_only),
            "reconstructed": self.occupied_coords(self.reconstructed),
        }

    def reset(self):
        self.volume = self.original.copy()
        self.reconstructed = np.zeros_like(self.original)


STATE: DemoState = None  # populated by init_state() at startup


def init_state(volume_path=None, use_cuda: bool = True):
    global STATE
    volume, name, source_ext = load_demo_volume(volume_path)
    args = build_args(use_cuda=use_cuda)
    init_models(args)  # pre-warm: load the model weights once
    STATE = DemoState(volume=volume, name=name, args=args, source_ext=source_ext)
    print(
        f"[Server] Ready. volume={name} shape={STATE.volume.shape} occupied={int(STATE.original.sum())} "
        f"cube_size={CUBE_SIZE} source_ext={source_ext} device={args.device}"
    )
    return STATE


# --------------------------------------------------------------------------- #
# Static frontend                                                             #
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


# --------------------------------------------------------------------------- #
# API                                                                         #
# --------------------------------------------------------------------------- #
@app.route("/volume", methods=["GET"])
def get_volume():
    """Occupied voxels of the current authoritative state (for the initial draw)."""
    return jsonify(STATE.snapshot())


@app.route("/reconstruct", methods=["POST"])
def post_reconstruct():
    """
    Body: {"x": int, "y": int, "z": int}. Crops a centered cube from the CURRENT
    state, runs the real pipeline, ORs the result in, returns the newly-added
    voxels. Single-flight: returns 409 if a reconstruction is already running.
    """
    payload = request.get_json(force=True, silent=True) or {}
    try:
        x, y, z = int(payload["x"]), int(payload["y"]), int(payload["z"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "expected integer fields x, y, z"}), 400

    shape = STATE.volume.shape
    if not (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
        return jsonify({"error": f"({x},{y},{z}) out of bounds {shape}"}), 400

    # The lock is both the UX signal and the concurrency lock (constraint 3).
    if not STATE.lock.acquire(blocking=False):
        return jsonify({"error": "busy", "detail": "a reconstruction is in flight"}), 409
    try:
        result = reconstruct_at(STATE.volume, x, y, z, STATE.args, source_ext=STATE.source_ext)
        new_coords = result["new_coords"]
        if len(new_coords):
            gi, gj, gk = new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]
            # OR the new voxels into the authoritative state (constraint 2).
            STATE.volume[gi, gj, gk] = 1
            STATE.reconstructed[gi, gj, gk] = 1
        return jsonify({
            "added": int(len(new_coords)),
            "start": result["start"],
            "new": new_coords.reshape(-1).astype(int).tolist(),
            # 6-view 2D input/output of the 2D network (before/after) for the panel.
            "views": {
                "before": _encode_views(result["views_before"]),
                "after": _encode_views(result["views_after"]),
            },
        })
    finally:
        STATE.lock.release()


@app.route("/full_inference", methods=["GET"])
def get_full_inference():
    """
    Run the paper's full stride-grid pipeline over the whole current volume and
    stream per-cube progress as Server-Sent Events. Each message is JSON:
      {"type":"progress","done":D,"total":T,"added":A,"new":[x,y,z,...]}  # live voxels
      {"type":"done","done":D,"total":T,"added":A,"cancelled":bool}
    Progress events carry the newly-added voxels so the frontend can draw the
    reconstruction filling in live. On completion (or cancel) the authoritative
    state is replaced with the merged result. Single-flight: 409 if busy.
    Cancel an in-flight run with POST /full_inference/cancel.
    """
    if not STATE.lock.acquire(blocking=False):
        return jsonify({"error": "busy", "detail": "a reconstruction is in flight"}), 409

    # Optional ?workers=N to tune the parallel pool live (default: machine-sized).
    try:
        workers = int(request.args["workers"]) if "workers" in request.args else None
    except (TypeError, ValueError):
        workers = None

    STATE.cancel_event.clear()  # fresh cancel flag for this run

    def stream():
        try:
            for ev in full_inference(STATE.volume, STATE.args, source_ext=STATE.source_ext,
                                     workers=workers, cancel_event=STATE.cancel_event):
                if "result" in ev:
                    result = ev.pop("result")
                    # Replace the authoritative state; everything the model added
                    # over the original becomes "reconstructed" (constraint 2).
                    STATE.volume = (result > 0.5).astype(np.uint8)
                    STATE.reconstructed = ((STATE.volume > 0.5) & (STATE.original <= 0.5)).astype(np.uint8)
                    yield f"data: {json.dumps({'type': 'done', **ev})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'progress', **ev})}\n\n"
        except Exception as exc:  # surface pipeline errors to the client
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        finally:
            STATE.cancel_event.clear()
            STATE.lock.release()

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    return Response(stream_with_context(stream()), mimetype="text/event-stream", headers=headers)


@app.route("/full_inference/cancel", methods=["POST"])
def post_full_inference_cancel():
    """Signal an in-flight full inference to stop after its current cube(s).
    Does NOT take the lock (the run holds it) -- it only flips the cancel flag."""
    STATE.cancel_event.set()
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def post_reset():
    """Restore the original volume state (drops all reconstructions)."""
    with STATE.lock:
        STATE.reset()
    return jsonify(STATE.snapshot())


def main():
    # NOTE:
    # CONFIG_FILENAME = "experiment_sota/parse2022_LC_32_50_ours_eval.yaml" - PARSE2022
    # CONFIG_FILENAME = "experiment1/PipeForge3DMesh_Best_LC_32.yaml" - PipeForge3D Mesh
    # CONFIG_FILENAME = "experiment1/PipeForge3DPCD_Best_LC_32.yaml" - PipeForge3D PCD

    # VOLUME_PATH = r".\data\PipeForge3DMesh_Best\eval\50.npy"
    VOLUME_PATH =  r".\data\parse2022_32\eval\PA000150_vessel.nii.gz"

    parser = argparse.ArgumentParser(description="SBP-Net interactive demo server")
    parser.add_argument("--volume", type=str, default=VOLUME_PATH, help="Path to a .npy volume to load")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    cli = parser.parse_args()

    init_state(volume_path=cli.volume, use_cuda=not cli.no_cuda)
    # threaded=True so the single-flight lock (not the server) governs concurrency;
    # use_reloader=False so the heavy model is not loaded twice.
    app.run(host=cli.host, port=cli.port, threaded=True, use_reloader=False, debug=False)


if __name__ == "__main__":
    main()
