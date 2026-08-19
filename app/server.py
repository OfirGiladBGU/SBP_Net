"""
SBP-Net Interactive Demo -- Phase 2 headless local server (Flask).

The backend owns the authoritative volume state (constraint 2): it loads the
volume + model once at startup (pre-warm), crops every reconstruction from the
*current* state, ORs the new result in, and returns only the newly-added voxels.
A single-flight lock serializes reconstructions (constraint 3). Everything runs
locally and headless -- no VTK, no rendering in Python.

Datasets are declared in app/configs/*.yaml (see app/app_configs.py), so the UI
can offer a dataset picker instead of anything being hardcoded here.

    conda run -n TAE --no-capture-output python app/server.py
    # then open http://127.0.0.1:5000/

PROCESS MODEL -- why there are two processes:
    configs_parser.py computes every constant at import time and ~20 modules do
    `from configs.configs_parser import *`, which COPIES those values into their
    own namespace. So the active config cannot be changed once it is imported.
    Running `python app/server.py` therefore starts a small SUPERVISOR, which
    launches a WORKER (`--serve`) with SBP_CONFIG_FILENAME set for the chosen
    dataset. Switching datasets in the UI makes the worker exit with
    _SWITCH_EXIT_CODE; the supervisor relaunches it on the new config. That
    keeps the config guaranteed-consistent instead of half-reloaded.
    Switching only the *volume* needs no restart -- see POST /volume/select.

All heavy lifting lives in app/reconstruct_core.reconstruct_at.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see reconstruct_core.py

import argparse
import base64
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename

ROOT_PATH = str(pathlib.Path(__file__).absolute().parent.parent)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from app import app_configs, cache_store

# NOTE: app.reconstruct_core (torch + configs_parser) is imported lazily by
# init_state(). The supervisor runs this same file and must NOT pull the
# pipeline in -- it has to pick the config *before* configs_parser is imported.
CORE = None

# Worker exit code meaning "relaunch me on a different config" (see _supervise).
_SWITCH_EXIT_CODE = 42
# Set by the supervisor on the worker's environment; without it a restart would
# never come back, so POST /config refuses instead of killing the server.
_SUPERVISED_ENV = "SBP_DEMO_SUPERVISED"


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

# Cap for "Load file…" uploads. The bundled eval volumes are 0.4-5.5 MB, so this
# is roomy; it exists so a mis-drop can't buffer something enormous.
MAX_UPLOAD_MB = 512

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(413)
def _upload_too_large(_):
    """Keep the frontend on the JSON path even when Flask rejects the body."""
    return jsonify({"error": "file too large", "detail": f"limit is {MAX_UPLOAD_MB} MB"}), 413


class DemoState:
    """Authoritative volume state held in memory by the backend."""

    def __init__(self, config, args, volume: np.ndarray, name: str,
                 source_ext: str = ".npy", volume_path=None):
        self.config = config                                  # the app/configs/ descriptor in use
        self.args = args
        # The lock IS the concurrency guard: one reconstruction in flight (constraint 3).
        self.lock = threading.Lock()
        # Set by POST /full_inference/cancel to stop an in-flight full run early.
        self.cancel_event = threading.Event()
        self.load_volume(volume, name, source_ext, volume_path)

    def load_volume(self, volume: np.ndarray, name: str, source_ext: str,
                    volume_path=None, custom: bool = False):
        """(Re)seat the authoritative volume. The models are untouched -- they
        depend on the config, not on which volume of it is being shown.

        `custom` marks a volume loaded through "Load file…" -- it is not one of
        the dataset's VOLUMES_PATH entries, so it has no selectable path."""
        self.name = name
        self.custom = custom
        self.volume_path = None if custom else (app_configs._rel(volume_path) if volume_path else None)
        self.source_ext = source_ext                          # projection convention (constraint 4)
        self.original = (volume > 0.5).astype(np.uint8)       # immutable reference
        self.volume = self.original.copy()                    # current state (mutated)
        self.reconstructed = np.zeros_like(self.original)     # voxels added by the model

    def occupied_coords(self, mask: np.ndarray):
        """Flat [x0,y0,z0,x1,y1,z1,...] int list of occupied voxels in `mask`."""
        coords = np.argwhere(mask > 0.5).astype(np.int32)
        return coords.reshape(-1).tolist()

    def live_config(self):
        """The descriptor re-read from disk, falling back to the loaded one.

        Display-only settings (INITIAL_VOLUMES_ROTATION) are picked up from an
        edited .yaml without a restart -- that is why the UI's Reload App button
        is enough to see a tweaked rotation. Anything the *pipeline* binds at
        import time (CONFIG_FILENAME) still needs the worker relaunched.
        """
        return app_configs.get_config(self.config.name) or self.config

    def snapshot(self) -> dict:
        original_only = (self.original > 0.5) & (self.reconstructed <= 0.5)
        return {
            "name": self.name,
            "shape": [int(s) for s in self.volume.shape],
            "cube_size": int(CORE.CUBE_SIZE),
            "config": self.config.name,
            "config_label": self.config.label,
            "rotation": self.live_config().initial_rotation,   # display orientation, [x,y,z] deg
            "volume_path": self.volume_path,
            "custom_volume": bool(self.custom),
            "original": self.occupied_coords(original_only),
            "reconstructed": self.occupied_coords(self.reconstructed),
        }

    def reset(self):
        self.volume = self.original.copy()
        self.reconstructed = np.zeros_like(self.original)


STATE: DemoState = None  # populated by init_state() at startup


def init_state(config, volume_path=None, use_cuda: bool = True):
    """Import the pipeline (binding configs_parser to `config`), then pre-warm."""
    global STATE, CORE

    from app import reconstruct_core
    CORE = reconstruct_core

    # Fail loudly rather than serve one config's weights under another's name:
    # a silently mismatched config produces plausible-looking wrong results.
    from configs.configs_parser import CONFIG_FILENAME as ACTIVE_CONFIG_FILENAME
    if ACTIVE_CONFIG_FILENAME != config.config_filename:
        raise RuntimeError(
            f"Config mismatch: descriptor '{config.name}' expects "
            f"'{config.config_filename}' but configs_parser loaded "
            f"'{ACTIVE_CONFIG_FILENAME}'. Launch via `python app/server.py` so the "
            f"supervisor can set SBP_CONFIG_FILENAME."
        )

    resolved = config.resolve_volume(volume_path)
    volume, name, source_ext = CORE.load_demo_volume(resolved)
    args = CORE.build_args(use_cuda=use_cuda)
    CORE.init_models(args)  # pre-warm: load the model weights once
    STATE = DemoState(config=config, args=args, volume=volume, name=name,
                      source_ext=source_ext, volume_path=resolved)
    print(
        f"[Server] Ready. config={config.name} ({config.config_filename}) volume={name} "
        f"shape={STATE.volume.shape} occupied={int(STATE.original.sum())} "
        f"cube_size={CORE.CUBE_SIZE} source_ext={source_ext} device={args.device}"
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


@app.route("/configs", methods=["GET"])
def get_configs():
    """Every dataset in app/configs/ + which one is live, for the pickers."""
    return jsonify({
        "active": {
            "config": STATE.config.name,
            "volume": STATE.volume_path,          # None for a "Load file…" volume
            "volume_name": STATE.name,
            "custom": bool(STATE.custom),
        },
        "configs": [cfg.to_dict() for cfg in app_configs.list_configs()],
        # False => POST /config can't restart, so the UI hides the dataset picker.
        "supervised": os.environ.get(_SUPERVISED_ENV) == "1",
    })


@app.route("/config", methods=["POST"])
def post_config():
    """
    Body: {"name": <descriptor>, "volume": <repo-relative path, optional>}.
    Switches the active dataset. The config is baked in at import time, so this
    asks the supervisor for a clean restart on the new config: the response is
    sent, then this worker exits with _SWITCH_EXIT_CODE and is relaunched. The
    client should poll GET /volume until the new worker answers.
    """
    if os.environ.get(_SUPERVISED_ENV) != "1":
        return jsonify({"error": "unsupervised",
                        "detail": "started with --serve; run `python app/server.py` "
                                  "so a supervisor can restart the worker"}), 501

    payload = request.get_json(force=True, silent=True) or {}
    name = payload.get("name")
    config = app_configs.get_config(name) if name else None
    if config is None:
        return jsonify({"error": f"unknown config '{name}'"}), 400
    problems = config.problems()
    if problems:
        return jsonify({"error": f"config '{name}' is not usable", "detail": problems}), 400

    # `volume` is optional (the dataset's default is used); but if one is named
    # explicitly it must be real -- don't restart onto a different volume silently.
    requested_volume = payload.get("volume")
    if requested_volume and config.find_volume(requested_volume) is None:
        return jsonify({"error": f"'{requested_volume}' is not a volume of dataset "
                                 f"'{config.name}'"}), 400

    # Don't tear the process down underneath a running reconstruction.
    if not STATE.lock.acquire(blocking=False):
        return jsonify({"error": "busy", "detail": "a reconstruction is in flight"}), 409
    # NOTE: the lock is deliberately never released -- this process is going away,
    # and holding it stops anything new from starting during the handover.

    volume = config.resolve_volume(requested_volume)
    _request_switch(config.name, volume)
    return jsonify({
        "restarting": True,
        "config": config.name,
        "label": config.label,
        "volume": app_configs._rel(volume) if volume else None,
    })


@app.route("/volume/select", methods=["POST"])
def post_volume_select():
    """
    Body: {"path": <repo-relative path within the active config's VOLUMES_PATH>}.
    Swaps the volume WITHOUT a restart -- same config, so the loaded models stay
    valid. Returns the new state snapshot. 409 if a reconstruction is in flight.
    """
    payload = request.get_json(force=True, silent=True) or {}
    requested = payload.get("path")
    # Strict: an unknown path is an error, never a silent fallback to another volume.
    resolved = STATE.config.find_volume(requested)
    if resolved is None:
        return jsonify({"error": f"'{requested}' is not a volume of dataset "
                                 f"'{STATE.config.name}'"}), 400

    if not STATE.lock.acquire(blocking=False):
        return jsonify({"error": "busy", "detail": "a reconstruction is in flight"}), 409
    try:
        volume, name, source_ext = CORE.load_demo_volume(resolved)
        STATE.load_volume(volume, name, source_ext, resolved)
        print(f"[Server] Volume -> {name} shape={STATE.volume.shape} "
              f"occupied={int(STATE.original.sum())} source_ext={source_ext}")
        return jsonify(STATE.snapshot())
    finally:
        STATE.lock.release()


@app.route("/volume/upload", methods=["POST"])
def post_volume_upload():
    """
    Multipart `file` field: load ANY volume the user picks, from anywhere on
    their machine -- it does not have to live in the dataset's VOLUMES_PATH.
    Same config, so the loaded models stay valid and there is no restart.

    The bytes are written to a temp file and read with the project's own loader,
    then the temp copy is dropped (the volume lives in memory from then on).
    The original extension is preserved on that temp file on purpose: it selects
    the projection rotation convention (constraint 4), so renaming a .nii.gz to
    .npy would silently project it the wrong way.
    """
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "expected a multipart 'file' field"}), 400

    original = pathlib.Path(upload.filename).name          # ignore any client path
    if not original.lower().endswith(app_configs.VOLUME_EXTENSIONS):
        return jsonify({
            "error": f"unsupported file type: {original}",
            "detail": f"expected one of {', '.join(app_configs.VOLUME_EXTENSIONS)}",
        }), 400

    # secure_filename strips separators/oddities; keep the extension it may drop.
    safe = secure_filename(original) or "upload"
    if not safe.lower().endswith(app_configs.VOLUME_EXTENSIONS):
        ext = next(e for e in app_configs.VOLUME_EXTENSIONS if original.lower().endswith(e))
        safe = f"upload{ext}"

    if not STATE.lock.acquire(blocking=False):
        return jsonify({"error": "busy", "detail": "a reconstruction is in flight"}), 409

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="sbp_demo_upload_")
        tmp_path = pathlib.Path(tmpdir).joinpath(safe)
        upload.save(str(tmp_path))
        volume, _, source_ext = CORE.load_demo_volume(tmp_path)
        # Report the name the user recognises, not the sanitized temp one.
        STATE.load_volume(volume, original, source_ext, custom=True)
        print(f"[Server] Volume -> {original} (uploaded) shape={STATE.volume.shape} "
              f"occupied={int(STATE.original.sum())} source_ext={source_ext}")
        return jsonify(STATE.snapshot())
    except Exception as exc:
        return jsonify({"error": f"could not load '{original}'", "detail": str(exc)}), 400
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        STATE.lock.release()


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
        result = CORE.reconstruct_at(STATE.volume, x, y, z, STATE.args, source_ext=STATE.source_ext)
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


def _cache_readable():
    """(ok, reason) -- may we READ a cache entry for the current volume?

    Entries are validated against STATE.original, which never changes, so a
    cached result can be loaded even after the user has clicked around: the
    replay ORs it onto whatever is on screen, exactly like a live run would.
    """
    if STATE.custom:
        return False, "volume was loaded from a file (no stable identity)"
    if not STATE.name:
        return False, "volume has no name"
    return True, ""


def _cache_writable():
    """(ok, reason) -- may we WRITE the result of this run?

    Only a run over a pristine volume is the artifact we cache. Running full
    inference on top of click reconstructions is valid, it just isn't that
    artifact, and it must not overwrite the pristine entry for this volume.
    """
    ok, reason = _cache_readable()
    if not ok:
        return ok, reason
    if bool(STATE.reconstructed.any()):
        return False, "state already contains reconstructions (not a pristine volume)"
    return True, ""


@app.route("/cache", methods=["GET"])
def get_cache():
    """Whether a full-inference cache entry can be loaded / would be written."""
    readable, read_why = _cache_readable()
    writable, write_why = _cache_writable()
    info = (cache_store.status(STATE.config.name, STATE.name, STATE.original)
            if readable else {"available": False, "reason": read_why})
    return jsonify({
        "config": STATE.config.name,
        "volume": STATE.name,
        # loadable => the "Load Cached Result" button can do something right now
        "loadable": bool(readable and info.get("available")),
        # cacheable => a fresh run from here would be stored
        "cacheable": writable,
        "reason": read_why or info.get("reason", "") or write_why,
        "entry": info,
    })


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

    CACHING (app/cache/, see cache_store.py): a complete run on a pristine
    dataset volume is stored and replayed instantly next time -- as the same
    event stream, so the volume still fills in live.
      ?refresh=1     always recompute, then overwrite the entry
      ?cache_only=1  replay the cache, never compute (an `error` event if there
                     is nothing to replay) -- this is the UI's "Load Cached
                     Result" button
    The `done` event reports `cached` and, on a fresh run, `saved`.
    """
    if not STATE.lock.acquire(blocking=False):
        return jsonify({"error": "busy", "detail": "a reconstruction is in flight"}), 409

    # Optional ?workers=N to tune the parallel pool live (default: machine-sized).
    try:
        workers = int(request.args["workers"]) if "workers" in request.args else None
    except (TypeError, ValueError):
        workers = None

    STATE.cancel_event.clear()  # fresh cancel flag for this run

    def _flag(name):
        return str(request.args.get(name, "0")).lower() in ("1", "true", "yes")

    # ?refresh=1 recomputes and overwrites; ?cache_only=1 refuses to compute.
    refresh = _flag("refresh")
    cache_only = _flag("cache_only")
    readable, read_why = _cache_readable()
    writable, write_why = _cache_writable()

    cached_added, cached_meta, cache_error = None, None, None
    if readable and not refresh:
        cached_added, info = cache_store.load(STATE.config.name, STATE.name, STATE.original)
        if cached_added is None:
            print(f"[Cache] miss {STATE.config.name}/{STATE.name}: {info}")
        else:
            cached_meta = info
            print(f"[Cache] hit {STATE.config.name}/{STATE.name}: "
                  f"{len(cached_added):,} voxels (cached {info.get('created')})")
    if cache_only and cached_added is None:
        # Explicit "load from cache" with nothing to load: say so rather than
        # silently running a 2-minute computation the user didn't ask for.
        cache_error = (f"no cached result for {STATE.config.name}/{STATE.name}"
                       + (f" — {read_why}" if read_why else ""))
    elif not writable:
        print(f"[Cache] run will not be stored: {write_why}")

    def stream():
        started = time.time()
        try:
            if cache_error:
                yield f"data: {json.dumps({'type': 'error', 'error': cache_error})}\n\n"
                return
            if cached_added is not None:
                # Cache hit: same event shape, so the frontend still draws the
                # volume filling in live -- just without the compute.
                events = cache_store.replay(STATE.volume, cached_added,
                                            total_cubes=int(cached_meta.get("total_cubes", 0)))
            else:
                events = CORE.full_inference(STATE.volume, STATE.args, source_ext=STATE.source_ext,
                                             workers=workers, cancel_event=STATE.cancel_event)
            for ev in events:
                if "result" in ev:
                    result = ev.pop("result")
                    # Replace the authoritative state; everything the model added
                    # over the original becomes "reconstructed" (constraint 2).
                    STATE.volume = (result > 0.5).astype(np.uint8)
                    STATE.reconstructed = ((STATE.volume > 0.5) & (STATE.original <= 0.5)).astype(np.uint8)
                    ev["cached"] = cached_added is not None
                    # Store only complete, freshly computed runs (DEMO_PLAN Phase 6).
                    if cached_added is None and writable and not ev.get("cancelled"):
                        try:
                            meta = cache_store.save(
                                STATE.config.name, STATE.name, STATE.original, result,
                                total_cubes=int(ev.get("total", 0)),
                                seconds=time.time() - started,
                                source_path=STATE.volume_path,
                            )
                            ev["saved"] = True
                            print(f"[Cache] saved {STATE.config.name}/{STATE.name}: "
                                  f"{meta['added']:,} voxels in {meta['seconds']}s")
                        except Exception as exc:      # a cache failure must not fail the run
                            ev["saved"] = False
                            print(f"[Cache] could not save {STATE.config.name}/{STATE.name}: {exc}")
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


# --------------------------------------------------------------------------- #
# Supervisor: owns the worker process so the config can be swapped cleanly.   #
# --------------------------------------------------------------------------- #
def _switch_file(port: int) -> pathlib.Path:
    """Where a worker leaves the dataset it wants to be relaunched on."""
    return pathlib.Path(tempfile.gettempdir()).joinpath(f"sbp_demo_switch_{port}.json")


def _request_switch(config_name: str, volume_path):
    """Record the requested dataset, then exit so the supervisor relaunches us.

    The delay lets Flask flush the response first; os._exit skips interpreter
    cleanup so the port is released immediately (and no half-torn-down pipeline
    can keep running).
    """
    payload = {"config": config_name, "volume": str(volume_path) if volume_path else None}
    _switch_file(_PORT).write_text(json.dumps(payload))
    print(f"[Server] Switching to config '{config_name}' -- restarting worker.")
    threading.Timer(0.5, lambda: os._exit(_SWITCH_EXIT_CODE)).start()


_PORT = 5000  # set by _serve(); used by _request_switch to name the switch file


def _serve(cli):
    """Worker process: SBP_CONFIG_FILENAME is already set by the supervisor."""
    global _PORT
    _PORT = cli.port

    config = app_configs.get_config(cli.config)
    if config is None:
        raise SystemExit(f"[Server] Unknown dataset '{cli.config}'. "
                         f"Available: {[c.name for c in app_configs.list_configs()]}")

    init_state(config=config, volume_path=cli.volume, use_cuda=not cli.no_cuda)
    # threaded=True so the single-flight lock (not the server) governs concurrency;
    # use_reloader=False so the heavy model is not loaded twice.
    app.run(host=cli.host, port=cli.port, threaded=True, use_reloader=False, debug=False)


def _supervise(cli):
    """
    Parent process: run a worker, and relaunch it whenever it asks for a
    different dataset. Nothing heavy is imported here -- picking the config has
    to happen before configs_parser is imported anywhere (see module docstring).
    """
    switch_file = _switch_file(cli.port)
    switch_file.unlink(missing_ok=True)                 # ignore a stale request
    selection = {"config": cli.config, "volume": cli.volume}

    while True:
        config = app_configs.get_config(selection["config"])
        if config is None:
            config = app_configs.default_config()
            print(f"[Supervisor] Unknown dataset '{selection['config']}', "
                  f"falling back to '{config.name}'.")
        problems = config.problems()
        if problems:
            print(f"[Supervisor] Dataset '{config.name}' is not usable: {'; '.join(problems)}")
            return 1

        cmd = [sys.executable, os.path.abspath(__file__), "--serve",
               "--config", config.name, "--host", cli.host, "--port", str(cli.port)]
        if selection.get("volume"):
            cmd += ["--volume", str(selection["volume"])]
        if cli.no_cuda:
            cmd.append("--no-cuda")

        # The worker inherits the chosen config through the environment, so
        # configs_parser binds to it at its very first import.
        env = dict(os.environ,
                   SBP_CONFIG_FILENAME=config.config_filename,
                   **{_SUPERVISED_ENV: "1"})

        print(f"[Supervisor] Starting worker: dataset={config.name} "
              f"config={config.config_filename}")
        proc = subprocess.Popen(cmd, env=env)
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            return 0

        if code != _SWITCH_EXIT_CODE:
            return code

        if switch_file.is_file():
            try:
                selection = json.loads(switch_file.read_text())
            except json.JSONDecodeError:
                pass                                    # keep the current selection
            switch_file.unlink(missing_ok=True)
        time.sleep(1.0)                                 # let the port drain before rebinding


def main():
    default_config = app_configs.default_config()

    parser = argparse.ArgumentParser(description="SBP-Net interactive demo server")
    parser.add_argument("--config", type=str, default=default_config.name,
                        help=f"Dataset from app/configs/ "
                             f"(available: {', '.join(c.name for c in app_configs.list_configs())})")
    parser.add_argument("--volume", type=str, default=None,
                        help="Volume to load (defaults to the dataset's DEFAULT_VOLUME_PATH)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--serve", action="store_true", default=False,
                        help="Internal: run the worker directly, without a supervisor "
                             "(the dataset picker is then disabled)")
    cli = parser.parse_args()

    if cli.serve:
        _serve(cli)
        return 0
    return _supervise(cli)


if __name__ == "__main__":
    sys.exit(main() or 0)
