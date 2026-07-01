# SBP-Net Interactive Demo — Build Plan

Working spec for building the ICIP Show & Tell interactive demo. This file is the
source of truth for the build. Read it fully before writing code, and check work
against the **Non-negotiable constraints** section on every phase.

---

## Goal

An interactive 3D demo of the SBP-Net thin-structure reconstruction pipeline:

1. A 3D viewer shows an input volume (loaded from a numpy array; existing file
   converters produce the numpy).
2. The user orbits the object, then picks a point on the structure's surface
   (e.g. press `P`, move/orbit, click a location).
3. The picked point's XYZ becomes the **center of a 32³ box** (dynamically cropped
   around the click — NOT the fixed stride grid).
4. That single cube is sent through the **real** inference pipeline (online
   projection → 2D model → reprojection/fusion → **3D continuity filter**).
5. The reconstructed voxels are **patched live** into the viewer. No forced save
   to disk in the interactive loop.
6. (Later, optional) export the intermediate 6-view projections to a chosen path.

All computation runs **locally** on the demo laptop. Browser renders; Python
computes **and serves**. No Node.js anywhere in the stack — a single Python
process serves the WebGL front-end and answers reconstruction requests.

---

## Architecture

**Python-only toolchain — no Node.js.** A single Python process (FastAPI or
Flask) does everything: serves the static WebGL front-end files AND runs the
model. Node is never required. WebGL is browser technology, so the *viewer code
itself* is authored in JS/GLSL (there is no Python that runs inside the browser),
but nothing in the build or run stack uses Node — Python delivers the page to the
browser and handles all compute.

**Browser (JS / WebGL2, served by Python)** — renders the structure, handles
camera + picking, sends the clicked voxel coordinate to the backend, receives new
voxels, patches them into the scene. This is the only JS/GLSL in the project. No
rendering happens in Python.

**Python backend (headless, local HTTP server)** — holds the loaded volume and
model in memory, exposes reconstruction as an endpoint, AND serves the front-end
static files (mount the static folder in FastAPI/Flask; e.g. `StaticFiles` in
FastAPI or `static_folder` in Flask). **No VTK, no PyVista, no matplotlib GUI, no
`interactive_plot_*` imports.** Headless only.

Run model: `python server.py` starts one process that serves `index.html` +
JS/shaders and answers `/reconstruct`. For quick static-only iteration on the
front-end, `python -m http.server` also works — but the real app is served by the
backend so the page and the compute come from the same origin. Do NOT introduce
`npm`, `npx`, a Node dev server, or any JS build tooling; author the `.js`/`.html`/
shader files directly and let Python serve them.

This split is deliberate: it avoids VTK entirely, which sidesteps the Windows
Application Control policy that blocks VTK DLLs on the dev machine. Do not
reintroduce VTK/PyVista anywhere.

---

## Reuse existing code (do NOT reimplement the model)

Relevant modules already in the repo:

- `evaluator/predict_pipeline.py` — `single_predict`, `init_pipeline_models`.
  Note: `single_predict` currently **exports the reconstructed cube to disk** via
  `export_output_3d` instead of returning it. For the demo we want the cube back
  **in memory**.
- `evaluator/online_utils.py` — `online_preprocess_2d`, `online_preprocess_3d`
  (the reproject + OR-fuse + continuity-filter logic).
- `online_pipeline.py` — `prepare_2d_projections_and_3d_cubes` (reference for how
  cropping + projection + prediction are wired for the online path).
- `datasets_forge/dataset_2d_creator.py` — `crop_mini_cubes` (cube metadata has
  `start_x/end_x/start_y/end_y/start_z/end_z`); reference for the projection call.
- `datasets/dataset_utils.py` — `project_3d_to_2d`, `reverse_rotations`,
  `convert_data_file_to_numpy`, `get_data_file_stem`.
- `configs/configs_parser.py` — `DATA_3D_SIZE`, `DATA_3D_STRIDE`,
  `DATA_2D_SIZE`, `IMAGES_6_VIEWS`, weights paths, and the `APPLY_*` filter flags.

New code building on / modifying the core is **explicitly allowed**, including a
new online projection path for a single dynamically-cropped centered cube. It does
not have to be something the existing offline flow supported.

---

## Non-negotiable constraints

These override any speed/simplicity optimization. Do not drop them.

1. **The 3D continuity filter is mandatory.**
   `components_continuity_3d_local_connectivity` (i.e. `APPLY_NOISE_FILTER_3D`
   behavior) MUST run on every reconstruction. Without it the results look bad.
   Do not disable or "optimize it away" for responsiveness. A blocking loader
   during compute is the accepted cost.

2. **Backend owns the authoritative volume state.**
   The backend holds the current display volume (original + everything
   reconstructed so far). Each reconstruction crops from the *current* backend
   state, ORs the new result into that state, and returns only the newly added
   voxels. The frontend is a pure mirror — it only appends what the backend
   returns. This guarantees the view always reflects the latest completed result
   and cannot drift or go stale.

3. **Serialize clicks — one reconstruction in flight at a time.**
   While a reconstruction is computing, the frontend blocks further picks and
   shows a loading state. The loader is both the UX signal and the concurrency
   lock. No overlapping/racing reconstructions.

4. **The online projection must match the training projection format.**
   The new single-cube online projection must produce the same 6-view
   representation (normalization, orientation/rotation convention) that the 2D
   model was trained on. Mismatch silently degrades reconstruction quality.
   Verify early by diffing against a known offline-projected cube.

5. **No disk round-trip in the interactive loop.** Reconstructed cube returns in
   memory. Disk export (`export_output_3d` / `full_merge`) is the batch path and
   must not be on the click path. (Projection *export* is a separate opt-in
   feature — see Phase 5.)

---

## Phases (build in order)

### Phase 1 — Core: `reconstruct_at(volume, x, y, z)` — THE BALLGAME
Prove the compute before any web layer. A bare script, hardcoded coordinate.

- Dynamically crop a `DATA_3D_SIZE` (32³) cube **centered** on (x,y,z), clamped
  to volume bounds.
- Run the new online centered-crop projection → 2D preprocess → 2D model →
  3D preprocess (reproject + fusion) → **continuity filter** (mandatory).
- Return newly-added voxel coordinates in **global** volume coordinates.
- Print voxel counts. Confirm projection output matches offline format (constraint 4).

**If this works from a plain script, the rest is plumbing. Do not proceed until it
does.** If the dynamic centered-crop projection needs setup that isn't obvious,
that is where the real work is — solve it here, in a 10-line test, not later.

### Phase 2 — Headless local server (also serves the front-end)
- FastAPI or Flask. Load volume + model **once at startup** (pre-warm).
- Backend **owns volume state** (constraint 2).
- Serves the static WebGL front-end (`index.html`, JS, shaders) from the same
  process — FastAPI `StaticFiles` or Flask static folder. No Node, no separate
  web server for production. `python server.py` is the single entry point.
- `GET /volume` → occupied voxel coords of current state as JSON (for initial draw).
- `POST /reconstruct {x,y,z}` → runs `reconstruct_at` against current state, updates
  state, returns new voxel coords.
- One request in flight at a time (constraint 3, enforced server-side too).
- Test with `curl` before any frontend.

### Phase 3 — WebGL2 point-cloud renderer
- Static `index.html` + JS, **served by the Python backend from Phase 2** (no Node,
  no build step). Reuse existing WebGL fluency from `main.js`
  (context, camera, pointer-lock orbit, render loop). Author the JS/GLSL by hand.
- Render occupied voxels as **points** via a vertex buffer (NOT raytraced).
- Controllable `gl_PointSize` so points are comfortable click targets.
- Fetch `GET /volume` on load and draw. Original vs. reconstructed voxels in
  distinct colors.

### Phase 4 — Picking + live loop
- **GPU color-picking**: render point IDs encoded as color to an offscreen
  framebuffer, read the pixel under the cursor → nearest point → its XYZ.
  Use a small pick tolerance so slight misses snap to the nearest point.
- Loop: pick → **lock + show loader** → `POST /reconstruct` → append returned
  voxels in highlight color → re-render → unlock.
- This is the one genuinely new graphics concept; treat picking as the trickiest part.

### Phase 5 — Polish (only after the loop works)
- Swap the **points** renderer for **instanced voxel cubes**
  (`drawArraysInstanced`: one small cube stamped at every voxel position in a
  single draw call). Picking and backend do NOT change — only the draw code.
  This looks better and is easier to click. Do this as an isolated upgrade.
- Reset button (one backend call to restore original state).
- Before/after toggle.
- (Optional) Show the 6 projections of the last-picked cube as a 2D overlay —
  good "here's what the model saw" storytelling.
- (Optional, last) Export intermediate projections to a chosen output path.

---

## Sequencing discipline

- Phase 1 is the whole ballgame. Everything after it is standard plumbing.
- Points first, voxels later — do not build instanced voxels before the click→fill
  loop works end-to-end, or the renderer will eat the time the reconstruction loop
  needs.
- Optional features (projection export) must not gate or complicate the core.

## Out of scope / known non-issues

- **Node.js / npm / JS build tooling is not used.** The stack is Python-only for
  serving and compute; the browser viewer is hand-authored JS/GLSL served by the
  Python backend. Do not introduce a Node dev server, bundler, or `package.json`.
  "Python only" here means no Node in the toolchain — it does not mean the viewer
  can be written in Python (WebGL runs in the browser and must be JS/GLSL).
- The VTK Application Control block is a machine policy, not a repo problem. This
  architecture avoids VTK entirely; do not try to "fix" VTK here.
- The `torch.load` FutureWarning is cosmetic; optionally pass `weights_only=True`
  during cleanup, but it is not part of this build.
