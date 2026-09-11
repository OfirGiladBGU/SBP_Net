"""
Swap the active config WITHOUT restarting the process.

`configs_parser` computes its constants at import time and ~20 modules do
`from configs.configs_parser import *`, which COPIES those values into their own
namespace. That is why switching datasets used to relaunch the worker. Measured
cost of that relaunch:

    import torch                 5.9s
    app.reconstruct_core         8.4s   (cv2, nibabel, open3d, models, ...)
    ------------------------------------
    pure restart overhead       14.3s
    init_models + volume load    3.7s   <- the only part a switch actually needs

So the restart was ~4x the real work. This module removes it.

WHY RELOAD RATHER THAN JUST ASSIGNING THE NEW VALUES. Patching each module's
copies with setattr would be faster still, but it is wrong here: several modules
DERIVE state from a config value at import time, and a patched name leaves the
derivative stale. On the demo's own hot path:

    app/reconstruct_core.py:46  CUBE_SIZE            <- DATA_2D_SIZE
    app/reconstruct_core.py:53  _INPUT_SIZE_MODEL_2D <- DATA_2D_SIZE

Switching between a 32-cube and a 16-cube config would then keep cropping at 32
and feed the model the wrong input shape -- silently. Re-executing the module
body recomputes those; assignment cannot.

WHY THIS IS SAFE, having been rejected earlier in the build: the danger of
module reloading is a *silently* half-applied config. So the swap does not trust
itself. It records which module attributes are genuine copies of config values,
and afterwards proves every one of them matches the new config; anything left
over raises, and the caller falls back to the proven process restart. A swap
either lands completely or does not count as having happened.
"""

import importlib
import os
import pathlib
import sys
from types import ModuleType

ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent

# Never reload these: they own live state (the Flask app, STATE, the caches) or
# are the machinery doing the reloading.
NEVER_RELOAD = {
    "app", "app.server", "app.config_swap", "app.app_configs",
    "app.volume_cache", "app.cache_store", "app.build_cache",
}

# When several modules are stale, refresh them in this order: a module must be
# rebound before anything that imports names from it, or the importer copies
# values that are about to change. Anything not listed is reloaded afterwards.
RELOAD_ORDER = [
    "datasets.dataset_utils",
    "datasets.dataset_list",
    "datasets.custom_datasets_2d",
    "datasets.custom_datasets_3d",
    "evaluator.online_utils",
    "evaluator.offline_utils",
    "datasets_forge.dataset_2d_creator",
    "datasets_forge.dataset_3d_creator",
    "evaluator.predict_pipeline",
    "app.reconstruct_core",
]
_ORDER_INDEX = {name: i for i, name in enumerate(RELOAD_ORDER)}


_ROOT_PREFIX = os.path.normcase(str(ROOT_PATH)) + os.sep
_is_project_cache = {}


def _is_project_module(module) -> bool:
    """Is this one of our modules (vs stdlib / site-packages)?

    Memoised, and deliberately string-based: Path.resolve() touches the
    filesystem, and this runs for every loaded module on every scan pass -- it
    was costing ~1.3s per swap, which was the whole cost of a swap.
    """
    path = getattr(module, "__file__", None)
    if not path:
        return False
    cached = _is_project_cache.get(path)
    if cached is None:
        cached = os.path.normcase(os.path.abspath(path)).startswith(_ROOT_PREFIX)
        _is_project_cache[path] = cached
    return cached


def _project_modules():
    for name, module in list(sys.modules.items()):
        if module is None or name in NEVER_RELOAD or name.startswith("configs"):
            continue
        if _is_project_module(module):
            yield name, module


def _config_values(parser) -> dict:
    """The plain data `from configs_parser import *` would copy."""
    values = {}
    for name, value in vars(parser).items():
        if name.startswith("_"):
            continue
        if isinstance(value, ModuleType) or callable(value):
            continue                        # classes/functions aren't config state
        values[name] = value
    return values


def _differs(a, b) -> bool:
    """Definitely-not-equal test that tolerates numpy arrays and odd types."""
    try:
        equal = (a == b)
    except Exception:
        return a is not b
    if isinstance(equal, bool):
        return not equal
    try:
        return not bool(getattr(equal, "all", lambda: equal)())
    except Exception:
        return a is not b


def _copied_pairs(values: dict) -> set:
    """
    (module, attribute) pairs that are genuine copies of a config value.

    Determined by agreement with the config that is loaded RIGHT NOW, before
    anything changes. This is what separates a real copy from a module that
    happens to define its own variable under a name configs_parser also uses --
    e.g. datasets_forge/metrics_3d.py sets its own DATASET_PATH from DATA_PATH,
    and generate_2d_preds.py its own CROPS_PATH. Those never agree, so they are
    never tracked, and they can't make the swap look permanently incomplete.
    """
    pairs = set()
    for name, module in _project_modules():
        namespace = getattr(module, "__dict__", {})
        for key, value in values.items():
            if key in namespace and not _differs(namespace[key], value):
                pairs.add((name, key))
    return pairs


def _stale(tracked: set, values: dict):
    """Tracked copies that no longer match the (new) config."""
    out = []
    for name, module in _project_modules():
        namespace = getattr(module, "__dict__", {})
        for key, value in values.items():
            if (name, key) in tracked and key in namespace and _differs(namespace[key], value):
                out.append((name, module, key))
                break
    return out


def swap(config_filename: str, max_passes: int = 6):
    """
    Rebind every loaded module to `config_filename`, or raise.

    Raising is meaningful: the caller must then restart the process, because a
    partial swap is exactly the failure this design refuses to ship.
    Returns (configs_parser, [names of modules reloaded]).
    """
    parser = importlib.import_module("configs.configs_parser")
    # Snapshot BEFORE anything moves, so "copy of a config value" is decided
    # against the config those copies actually came from.
    tracked = _copied_pairs(_config_values(parser))

    previous = os.environ.get("SBP_CONFIG_FILENAME")
    os.environ["SBP_CONFIG_FILENAME"] = config_filename
    try:
        importlib.reload(parser)
        if parser.CONFIG_FILENAME != config_filename:
            raise RuntimeError(f"configs_parser loaded '{parser.CONFIG_FILENAME}', "
                               f"expected '{config_filename}'")

        reloaded = []
        for _ in range(max_passes):
            values = _config_values(parser)
            stale = _stale(tracked, values)
            if not stale:
                break
            # Dependency order first; everything else after, so a module is
            # never refreshed from a neighbour that is itself about to change.
            stale.sort(key=lambda item: _ORDER_INDEX.get(item[0], len(RELOAD_ORDER)))
            for name, module, _key in stale:
                importlib.reload(module)
                reloaded.append(name)
        else:
            left = _stale(tracked, _config_values(parser))
            raise RuntimeError("config did not settle: " + ", ".join(
                f"{name}.{key}" for name, _m, key in left[:6]))

        # Final proof, after everything has settled.
        remaining = _stale(tracked, _config_values(parser))
        if remaining:
            raise RuntimeError("stale config bindings remain: " + ", ".join(
                f"{name}.{key}" for name, _m, key in remaining[:6]))

        return parser, reloaded
    except Exception:
        if previous is None:
            os.environ.pop("SBP_CONFIG_FILENAME", None)
        else:
            os.environ["SBP_CONFIG_FILENAME"] = previous
        raise
