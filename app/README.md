# SBP-Net Interactive Demo

An interactive 3D demo of the SBP-Net thin-structure reconstruction pipeline.
Orbit a volume, **click a point on the structure**, and the model reconstructs a
cube around that click and patches the new voxels into the view live.

- **Browser (WebGL2)** renders the point cloud, handles the camera + picking.
- **Python (Flask)** holds the volume + model in memory and runs the *real*
  inference pipeline (online projection → 2D model → reproject/OR-fuse). No rendering happens in Python.
- Fully local, headless backend. **No VTK / PyVista / matplotlib GUI.**

See [`DEMO_PLAN.md`](DEMO_PLAN.md) for the design and non-negotiable constraints.

## Layout

| File | Role |
|------|------|
| `reconstruct_core.py` | Phase 1 core: `reconstruct_at(volume, x, y, z, ...)`. Centered dynamic crop → real pipeline → newly-added voxels in **global** coords, in memory. Run directly for a proof-of-compute + projection-format check. |
| `server.py` | Phase 2 Flask server. Owns the authoritative volume state; single-flight lock. Serves the frontend. Also the supervisor that owns the worker process (see [Datasets](#datasets)). |
| `configs/*.yaml` | One file per selectable dataset: which `configs/` file to load, where its volumes live, which one to open by default. |
| `app_configs.py` | Scans `app/configs/`. Deliberately free of `configs_parser`/torch imports — the supervisor uses it *before* the pipeline is imported. Run directly to see what this machine can serve. |
| `static/index.html`, `static/main.js` | Phase 3–5 WebGL2 frontend: point-cloud renderer, trackball camera, GPU color-picking, live click→reconstruct loop. |

## Datasets

The demo has a **Dataset** picker; the entries come from `app/configs/*.yaml`, so
adding a dataset is dropping in a file — nothing is hardcoded in `server.py`:

```yaml
LABEL: "PARSE2022 (pulmonary vessels)"
ORDER: 1
CONFIG_FILENAME: "experiment_sota/parse2022_LC_32_50_ours_eval.yaml"
VOLUMES_PATH: "data/parse2022/eval"          # scanned to fill the Volume picker
DEFAULT_VOLUME_PATH: "data/parse2022/eval/PA000310_vessel.nii.gz"
```

Paths are relative to the repo root. A dataset whose config or volumes are
missing is still listed, greyed out, with the reason in its tooltip
(`python app/app_configs.py` prints the same report).

**Why switching a dataset restarts the backend.** `configs_parser.py` computes
every constant at import time, and ~20 modules do `from configs.configs_parser
import *`, which *copies* those values into their own namespace — so the active
config can't be changed once imported. `python app/server.py` therefore runs a
small **supervisor** that launches a **worker** with `SBP_CONFIG_FILENAME` set
(the one hook added to `configs_parser.py`). Picking a dataset makes the worker
exit; the supervisor relaunches it on the new config, and the page polls until
it answers (a few seconds). The worker refuses to start if `configs_parser`
didn't bind to the config the descriptor names — a silently mismatched config
would produce plausible-looking wrong results.

Switching only the **volume** stays in-process: same config, same loaded models,
no restart.

**Load file…** opens a normal file picker and loads any volume from anywhere on
the machine, without it having to be in `VOLUMES_PATH`. It is uploaded to the
local backend rather than opened by path — the backend stays headless (no OS
file dialog in Python, per `DEMO_PLAN.md`) and the demo keeps working if the
browser is ever on a different machine. The file keeps its extension end-to-end,
because that selects the projection rotation convention (constraint 4). It loads
under the **current** config's models and shows up in the Volume picker as
`name (loaded file)`; switching dataset returns to that dataset's default.

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

# 3) See which datasets this machine can actually serve
conda run -n SBP --no-capture-output python app/app_configs.py
```

Options: `--config <name>` (a stem from `app/configs/`, default = first usable
one), `--volume path/to/volume.npy` (defaults to that dataset's
`DEFAULT_VOLUME_PATH`), `--port 5000`, `--no-cuda`.

`--serve` runs the worker alone, with no supervisor: useful under a debugger,
but the Dataset picker is then disabled (nothing could restart it) and
`POST /config` returns `501`.

## API

| Method / path | Purpose |
|---|---|
| `GET /volume` | Current state: `{name, shape, cube_size, config, config_label, volume_path, original[], reconstructed[]}` (flat `x,y,z` voxel lists). |
| `GET /configs` | Every `app/configs/` dataset + its volumes, which one is active, and `supervised` (whether a restart is possible). |
| `POST /config` `{name, volume?}` | Switch dataset. Replies `{restarting:true}`, then the worker exits and the supervisor relaunches it on that config; poll `GET /volume` until it answers. `409` if busy, `501` if unsupervised. |
| `POST /volume/select` `{path}` | Load another volume of the **active** dataset — no restart. Returns the new snapshot. `400` unless `path` is one of that dataset's volumes. |
| `POST /volume/upload` (multipart `file`) | Load a volume from **anywhere** — it need not be in `VOLUMES_PATH`. Same config, no restart. The bytes go to a temp file (keeping the original extension), are read with the project's loader, and the copy is dropped. `400` on an unsupported/unreadable file, `413` over 512 MB. |
| `POST /reconstruct` `{x,y,z}` | Reconstruct a centered cube from the **current** state, OR it in, return newly-added voxels **plus** `views.before` / `views.after` (the 2D network input/output as PNGs). `409` if one is already in flight. |
| `GET /full_inference?workers=N` | Run the paper's full stride-grid pipeline over the whole volume, **in parallel** (`workers` threads, default = machine-sized; `1` = sequential). **Server-Sent Events** stream per-cube progress incl. the cube's new voxels (`{type:"progress",done,total,added,new:[x,y,z,…]}`) for live drawing, then `{type:"done",…,cancelled}`. The merged result (partial if cancelled) replaces the state. `409` if busy. |
| `POST /full_inference/cancel` | Signal an in-flight full run to stop after its current cube(s); the partial result is kept. Does not take the lock. |
| `POST /reset` | Restore the original volume (drops all reconstructions). |

## Controls

Drag to rotate · wheel to zoom · right/shift-drag to pan · ctrl/alt-drag to roll
· **click a point** to reconstruct around it. Rotation is a trackball: drag
deltas are applied about the camera's *own* axes, so the object always follows
the cursor from any viewpoint (no world-up axis, no pole singularity). Recenter
restores the default view. A blocking loader is shown while the pipeline runs (it
is also the concurrency lock — one reconstruction at a time). Top-right panel:
**Dataset** and **Volume** pickers, **Load file…**, voxel (lit) vs. points view, point size
(active only in points/PCD mode — voxel view sizes cubes from the volume shape),
show/hide reconstructed, Reset, Recenter, and
**Run Full Inference** — runs the whole volume in parallel and **draws the
reconstruction filling in live**, cube by cube, with a progress bar and a
**Cancel** button that stops it early (keeping whatever was completed).

After a click, the **bottom panel** shows the 6 projections of that cube with a
**Before / After** flip — the 2D input the network saw vs. the output it produced.
