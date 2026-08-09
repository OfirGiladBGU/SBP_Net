"""
Full-inference result cache -- Phase 6 of app/DEMO_PLAN.md (first slice).

Full inference runs the paper's whole stride grid over a volume and is by far the
most expensive thing the demo does. This module stores the finished result so a
re-run is instant, which is what makes it safe to show live.

Layout mirrors the dataset descriptors, one entry per volume:

    app/cache/
      parse2022/
        PA000310_vessel.nii.gz.npz     # the voxels full inference ADDED
        PA000310_vessel.nii.gz.json    # what produced them (the validity key)
      pipeforge3d_mesh/
        50.npy.npz
        50.npy.json

Only the *added* voxels are stored, not the merged volume: they are what the
frontend draws, they compress far better than a 512^3 array, and rebuilding the
merged state is just `original | added`.

VALIDITY IS EVERYTHING (see DEMO_PLAN Phase 6). A stale hit that serves another
config's result is worse than no cache at all, so an entry is used only when the
config file, the volume's content, and every pipeline parameter that changes the
answer all still match. Anything else is a miss with a printed reason.

Scope for now: full-inference results of *dataset* volumes only. Volumes loaded
through "Load file..." have no stable identity and are not cached; custom saves
and voxel editing are explicitly out of scope for this slice.
"""

import hashlib
import json
import pathlib
import time

import numpy as np

CACHE_PATH = pathlib.Path(__file__).absolute().parent.joinpath("cache")

# Bump when the stored format changes in a way old entries can't satisfy.
SCHEMA_VERSION = 1


def _volume_digest(volume: np.ndarray) -> str:
    """Content hash of the binarized volume -- the volume's real identity.

    Keyed on content, not path: editing a file in place must miss the cache.
    """
    binary = np.ascontiguousarray((volume > 0.5).astype(np.uint8))
    digest = hashlib.sha1()
    digest.update(str(binary.shape).encode())
    digest.update(binary.tobytes())
    return digest.hexdigest()


def pipeline_signature() -> dict:
    """Every config value that changes what full inference produces.

    Imported lazily: configs_parser must not be pulled into the supervisor
    process (see the process-model note in server.py).
    """
    from configs.configs_parser import (
        CONFIG_FILENAME,
        DATA_3D_SIZE, DATA_3D_STRIDE,
        DENSITY_LOWER_THRESHOLD, DENSITY_UPPER_THRESHOLD,
        APPLY_FUSION, APPLY_INPUT_MERGE_2D, APPLY_INPUT_MERGE_3D,
        APPLY_THRESHOLD_2D, APPLY_THRESHOLD_3D, THRESHOLD_2D, THRESHOLD_3D,
        APPLY_NOISE_FILTER_2D, APPLY_NOISE_FILTER_3D,
        HARD_NOISE_FILTER_2D, HARD_NOISE_FILTER_3D,
        APPLY_CONTINUITY_FIX_2D, APPLY_CONTINUITY_FIX_3D,
        PREDICT_CONNECTIVITY_TYPE_2D, PREDICT_CONNECTIVITY_TYPE_3D,
        BINARY_DILATION, WEIGHTS_2D_PATH, WEIGHTS_3D_PATH,
    )
    return {
        "config_filename": CONFIG_FILENAME,
        "data_3d_size": list(DATA_3D_SIZE),
        "data_3d_stride": list(DATA_3D_STRIDE),
        "density_lower": float(DENSITY_LOWER_THRESHOLD),
        "density_upper": float(DENSITY_UPPER_THRESHOLD),
        "apply_fusion": bool(APPLY_FUSION),
        "apply_input_merge_2d": bool(APPLY_INPUT_MERGE_2D),
        "apply_input_merge_3d": bool(APPLY_INPUT_MERGE_3D),
        "apply_threshold_2d": bool(APPLY_THRESHOLD_2D),
        "apply_threshold_3d": bool(APPLY_THRESHOLD_3D),
        "threshold_2d": float(THRESHOLD_2D),
        "threshold_3d": float(THRESHOLD_3D),
        "apply_noise_filter_2d": bool(APPLY_NOISE_FILTER_2D),
        "apply_noise_filter_3d": bool(APPLY_NOISE_FILTER_3D),
        "hard_noise_filter_2d": bool(HARD_NOISE_FILTER_2D),
        "hard_noise_filter_3d": bool(HARD_NOISE_FILTER_3D),
        "apply_continuity_fix_2d": bool(APPLY_CONTINUITY_FIX_2D),
        "apply_continuity_fix_3d": bool(APPLY_CONTINUITY_FIX_3D),
        "connectivity_2d": int(PREDICT_CONNECTIVITY_TYPE_2D),
        "connectivity_3d": int(PREDICT_CONNECTIVITY_TYPE_3D),
        "binary_dilation": bool(BINARY_DILATION),
        "weights_2d": str(WEIGHTS_2D_PATH),
        "weights_3d": str(WEIGHTS_3D_PATH),
    }


def entry_paths(config_name: str, volume_name: str):
    """(npz, json) paths for one cached volume. `volume_name` keeps its extension."""
    safe = pathlib.Path(str(volume_name)).name          # never escape the cache dir
    folder = CACHE_PATH.joinpath(pathlib.Path(str(config_name)).name)
    return folder.joinpath(f"{safe}.npz"), folder.joinpath(f"{safe}.json")


def read_meta(config_name: str, volume_name: str):
    """The stored metadata for an entry, or None when there isn't one."""
    _, meta_path = entry_paths(config_name, volume_name)
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def _mismatch(meta: dict, volume: np.ndarray) -> str:
    """Why `meta` cannot be used for `volume`; empty string when it can."""
    if meta.get("schema_version") != SCHEMA_VERSION:
        return f"schema {meta.get('schema_version')} != {SCHEMA_VERSION}"
    if list(meta.get("volume_shape", [])) != [int(s) for s in volume.shape]:
        return f"volume shape {meta.get('volume_shape')} != {list(volume.shape)}"
    if meta.get("volume_digest") != _volume_digest(volume):
        return "volume content changed"
    if meta.get("cancelled"):
        return "entry is from a cancelled (partial) run"

    current = pipeline_signature()
    stored = meta.get("pipeline", {})
    changed = [k for k, v in current.items() if stored.get(k) != v]
    if changed:
        return f"pipeline params changed: {', '.join(sorted(changed))}"
    return ""


def status(config_name: str, volume_name: str, volume: np.ndarray) -> dict:
    """Whether a usable entry exists for this (config, volume), and why not."""
    if not volume_name:
        return {"available": False, "reason": "volume has no cacheable identity"}
    meta = read_meta(config_name, volume_name)
    if meta is None:
        return {"available": False, "reason": "no cache entry"}
    npz_path, _ = entry_paths(config_name, volume_name)
    if not npz_path.is_file():
        return {"available": False, "reason": "cache data file missing"}
    reason = _mismatch(meta, volume)
    if reason:
        return {"available": False, "reason": reason, "stale": True}
    return {
        "available": True,
        "added": int(meta.get("added", 0)),
        "total_cubes": int(meta.get("total_cubes", 0)),
        "created": meta.get("created"),
        "seconds": meta.get("seconds"),
    }


def load(config_name: str, volume_name: str, volume: np.ndarray):
    """The cached added-voxel coords (N,3) int32 + meta, or (None, reason)."""
    state = status(config_name, volume_name, volume)
    if not state["available"]:
        return None, state["reason"]
    npz_path, _ = entry_paths(config_name, volume_name)
    try:
        with np.load(npz_path) as data:
            added = data["added"].astype(np.int32)
    except (OSError, ValueError, KeyError) as exc:
        return None, f"unreadable cache data: {exc}"
    return added, read_meta(config_name, volume_name)


def save(config_name: str, volume_name: str, volume: np.ndarray, result: np.ndarray,
         total_cubes: int = 0, seconds: float = 0.0, source_path=None) -> dict:
    """
    Store what full inference ADDED to `volume` (i.e. result & ~volume).

    Only complete runs belong here -- a cancelled run is deliberately partial
    (DEMO_PLAN Phase 6), so callers must not call this for one.
    """
    if not volume_name:
        raise ValueError("cannot cache a volume without a stable name")

    original = volume > 0.5
    added_mask = (result > 0.5) & ~original
    added = np.argwhere(added_mask).astype(np.int32)

    npz_path, meta_path = entry_paths(config_name, volume_name)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, added=added)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "config": config_name,
        "volume_name": pathlib.Path(str(volume_name)).name,
        "source_path": str(source_path) if source_path else None,
        "volume_shape": [int(s) for s in volume.shape],
        "volume_digest": _volume_digest(volume),
        "volume_occupied": int(original.sum()),
        "added": int(len(added)),
        "total_cubes": int(total_cubes),
        "cancelled": False,
        "seconds": round(float(seconds), 2),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": pipeline_signature(),
    }
    with open(meta_path, "w") as stream:
        json.dump(meta, stream, indent=2)
    return meta


def clear(config_name: str, volume_name: str) -> bool:
    """Drop one entry. True if anything was removed."""
    removed = False
    for path in entry_paths(config_name, volume_name):
        if path.is_file():
            path.unlink()
            removed = True
    return removed


def replay(volume: np.ndarray, added: np.ndarray, total_cubes: int = 0, chunks: int = 150):
    """
    Emit a cached result as the same event stream a real run produces.

    A cache hit must still fill the volume in live (DEMO_PLAN Phase 6) -- dumping
    the final state in one frame throws away the demo's best visual. So the
    voxels are handed over in chunks, exactly like per-cube progress, just
    without the compute. Final event carries the merged volume for the server to
    adopt as the new authoritative state.
    """
    result = (volume > 0.5).astype(np.uint8)
    if len(added):
        result[added[:, 0], added[:, 1], added[:, 2]] = 1

    steps = max(1, min(int(chunks), len(added))) if len(added) else 1
    # Report progress against the real cube count when we know it, so a cached
    # run reads the same as a live one.
    total = int(total_cubes) or steps
    done_added = 0
    for i in range(steps):
        lo = (i * len(added)) // steps
        hi = ((i + 1) * len(added)) // steps
        batch = added[lo:hi]
        done_added += len(batch)
        yield {
            "done": ((i + 1) * total) // steps,
            "total": total,
            "added": done_added,
            "new": batch.reshape(-1).astype(int).tolist(),
        }
    yield {"done": total, "total": total, "added": int(len(added)),
           "result": result, "cancelled": False, "cached": True}
