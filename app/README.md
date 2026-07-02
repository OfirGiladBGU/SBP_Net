# SBP-Net Interactive Demo

An interactive 3D demo of the SBP-Net thin-structure reconstruction pipeline.
Orbit a volume, **click a point on the structure**, and the model reconstructs a
cube around that click and patches the new voxels into the view live.

- **Browser (WebGL2)** renders the point cloud, handles the camera + picking.
- **Python (Flask)** holds the volume + model in memory and runs the *real*
  inference pipeline (online projection → 2D model → reproject/OR-fuse →
  **mandatory 3D continuity filter**). No rendering happens in Python.
- Fully local, headless backend. **No VTK / PyVista / matplotlib GUI.**

See [`DEMO_PLAN.md`](DEMO_PLAN.md) for the design and non-negotiable constraints.

## Layout

| File | Role |
|------|------|
| `reconstruct_core.py` | Phase 1 core: `reconstruct_at(volume, x, y, z, ...)`. Centered dynamic crop → real pipeline → newly-added voxels in **global** coords, in memory. Run directly for a proof-of-compute + projection-format check. |
| `server.py` | Phase 2 Flask server. Owns the authoritative volume state; single-flight lock. Serves the frontend. |
| `static/index.html`, `static/main.js` | Phase 3–5 WebGL2 frontend: point-cloud renderer, orbit camera, GPU color-picking, live click→reconstruct loop. |
| `demo_viewer.py` | Legacy VTK/PyVista viewer (blocked by machine policy). **Not used** by this demo; kept here for reference only. |

## Requirements

Runs in the project's conda env (`SBP`: Python 3.10, torch + CUDA) plus **Flask**.
Base deps come from [`../manual_requirements.txt`](../manual_requirements.txt); the
one extra install for this demo is in [`extra_requirements.txt`](extra_requirements.txt)
(just Flask — the web viewer is vanilla JS/WebGL2 served from Python, no Node). The
cube size is read from the active config (`DATA_CROP_SIZE` / `DATA_2D_SIZE`), not
hardcoded.

```bash
conda activate SBP
pip install flask          # see app/extra_requirements.txt
```

> On this Windows machine the OpenMP runtime conflict (`OMP: Error #15`) is worked
> around with `KMP_DUPLICATE_LIB_OK=TRUE`, which the app sets itself. PyCharm sets
> it too; only matters when launching from a plain shell.

## Run

From the repo root:

```bash
# 1) Prove the compute (no web layer) — prints projection-match + voxel counts
conda run -n SBP --no-capture-output python app/reconstruct_core.py

# 2) Start the demo server (loads volume + model once, then serves the UI)
conda run -n SBP --no-capture-output python app/server.py
#    -> open http://127.0.0.1:5000/
```

Options: `--volume path/to/volume.npy` (defaults to the active config's held-out
eval object), `--port 5000`, `--no-cuda`.

## API

| Method / path | Purpose |
|---|---|
| `GET /volume` | Current state: `{name, shape, cube_size, original[], reconstructed[]}` (flat `x,y,z` voxel lists). |
| `POST /reconstruct` `{x,y,z}` | Reconstruct a centered cube from the **current** state, OR it in, return newly-added voxels **plus** `views.before` / `views.after` (the 2D network input/output as PNGs). `409` if one is already in flight. |
| `GET /full_inference` | Run the paper's full stride-grid pipeline over the whole volume; **Server-Sent Events** stream per-cube progress (`{type:"progress",done,total,added}` … `{type:"done",…}`), then the merged result replaces the state. `409` if busy. |
| `POST /reset` | Restore the original volume (drops all reconstructions). |

## Controls

Drag to orbit · wheel to zoom · right/shift-drag to pan · **click a point** to
reconstruct around it. A blocking loader is shown while the pipeline runs (it is
also the concurrency lock — one reconstruction at a time). Top-right panel: voxel
(lit) vs. points view, point size, show/hide reconstructed, Reset, Recenter, and
**Run Full Inference** (streams progress while the whole volume is processed).

After a click, the **bottom panel** shows the 6 projections of that cube with a
**Before / After** flip — the 2D input the network saw vs. the output it produced.
