"""
Pre-build the full-inference cache for whole datasets (DEMO_PLAN Phase 6).

Running full inference live takes minutes; app/cache_store.py makes a *re*-run
instant, but only once something has been computed. This script does that
computing up front, so the demo machine starts out with every volume cached.

    # everything this machine can serve
    python app/build_cache.py

    # just these datasets
    python app/build_cache.py --configs parse2022 pipeforge3d_mesh

    # one volume, or a rebuild of entries that already exist
    python app/build_cache.py --configs parse2022 --volumes PA000310_vessel.nii.gz
    python app/build_cache.py --refresh

Like server.py this runs as a PARENT + one WORKER per dataset: configs_parser
bakes its constants in at import time and ~20 modules copy them, so a single
process cannot switch configs. The parent picks the config, the worker inherits
it through SBP_CONFIG_FILENAME and loads the models once for all of that
dataset's volumes.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see reconstruct_core.py

import argparse
import pathlib
import subprocess
import sys
import time

ROOT_PATH = str(pathlib.Path(__file__).absolute().parent.parent)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

# Kept import-light on purpose: the parent must choose the config BEFORE
# configs_parser (or torch) is imported anywhere.
from app import app_configs


def _fmt(seconds: float) -> str:
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s" if seconds >= 60 else f"{seconds:.1f}s"


# --------------------------------------------------------------------------- #
# Worker: one dataset, models loaded once, every volume cached.               #
# --------------------------------------------------------------------------- #
def _build(config, volume_names, refresh: bool, workers) -> int:
    from app import cache_store, reconstruct_core

    from configs.configs_parser import CONFIG_FILENAME as ACTIVE
    if ACTIVE != config.config_filename:
        raise SystemExit(f"[{config.name}] config mismatch: expected "
                         f"'{config.config_filename}', configs_parser loaded '{ACTIVE}'")

    volumes = config.volumes()
    if volume_names:
        wanted = {n.lower() for n in volume_names}
        volumes = [v for v in volumes if v["name"].lower() in wanted]
        missing = wanted - {v["name"].lower() for v in volumes}
        if missing:
            print(f"[{config.name}] not in {config.volumes_path}: {sorted(missing)}")
    if not volumes:
        print(f"[{config.name}] nothing to do")
        return 0

    args = reconstruct_core.build_args()
    reconstruct_core.init_models(args)
    print(f"[{config.name}] models loaded on {args.device}; {len(volumes)} volume(s)", flush=True)

    failures = 0
    for i, entry in enumerate(volumes, 1):
        name = entry["name"]
        path = app_configs._resolve(entry["path"])
        head = f"[{config.name}] ({i}/{len(volumes)}) {name}"
        try:
            volume, _, source_ext = reconstruct_core.load_demo_volume(path)

            state = cache_store.status(config.name, name, volume)
            if state["available"] and not refresh:
                print(f"{head}: already cached ({state['added']:,} voxels) -- skipping", flush=True)
                continue

            started = time.time()
            total_cubes, result = 0, None
            for ev in reconstruct_core.full_inference(volume, args, source_ext=source_ext,
                                                      workers=workers):
                total_cubes = ev.get("total", total_cubes)
                if "result" in ev:
                    result = ev["result"]
            elapsed = time.time() - started

            meta = cache_store.save(config.name, name, volume, result,
                                    total_cubes=total_cubes, seconds=elapsed,
                                    source_path=entry["path"])
            print(f"{head}: +{meta['added']:,} voxels, {total_cubes} cubes, {_fmt(elapsed)}",
                  flush=True)
        except Exception as exc:
            failures += 1
            print(f"{head}: FAILED -- {type(exc).__name__}: {exc}", flush=True)
    return failures


# --------------------------------------------------------------------------- #
# Parent: one worker process per dataset.                                     #
# --------------------------------------------------------------------------- #
def _run_all(names, volume_names, refresh: bool, workers) -> int:
    configs = []
    for name in names:
        config = app_configs.get_config(name)
        if config is None:
            print(f"[build] unknown dataset '{name}' -- "
                  f"have {[c.name for c in app_configs.list_configs()]}")
            return 2
        problems = config.problems()
        if problems:
            print(f"[build] skipping '{name}': {'; '.join(problems)}")
            continue
        configs.append(config)
    if not configs:
        print("[build] no usable datasets")
        return 1

    started = time.time()
    failures = 0
    for config in configs:
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--config", config.name]
        if volume_names:
            cmd += ["--volumes", *volume_names]
        if refresh:
            cmd.append("--refresh")
        if workers is not None:
            cmd += ["--workers", str(workers)]
        env = dict(os.environ, SBP_CONFIG_FILENAME=config.config_filename)
        print(f"\n=== {config.label} ({config.config_filename}) ===", flush=True)
        failures += subprocess.call(cmd, env=env)

    # Reported from disk, so the summary reflects what is actually cached now.
    print(f"\n=== cache summary (built in {_fmt(time.time() - started)}) ===")
    from app import cache_store
    total_bytes = 0
    for config in configs:
        for entry in config.volumes():
            meta = cache_store.read_meta(config.name, entry["name"])
            npz, _ = cache_store.entry_paths(config.name, entry["name"])
            if meta and npz.is_file():
                total_bytes += npz.stat().st_size
                print(f"  OK  {config.name:<18} {entry['name']:<28} "
                      f"{meta['added']:>9,} voxels  {npz.stat().st_size / 1024:>7.0f} KB")
            else:
                print(f"  --  {config.name:<18} {entry['name']:<28} (not cached)")
    print(f"  total {total_bytes / 1024 / 1024:.1f} MB in {cache_store.CACHE_PATH}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="Pre-build the full-inference cache")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="dataset names from app/configs/ (default: all usable ones)")
    parser.add_argument("--volumes", nargs="*", default=None,
                        help="only these volume filenames (default: every volume)")
    parser.add_argument("--refresh", action="store_true",
                        help="rebuild entries that already exist")
    parser.add_argument("--workers", type=int, default=None, help="per-cube thread pool size")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=str, default=None, help=argparse.SUPPRESS)
    cli = parser.parse_args()

    if cli.worker:
        config = app_configs.get_config(cli.config)
        if config is None:
            raise SystemExit(f"unknown dataset '{cli.config}'")
        return _build(config, cli.volumes, cli.refresh, cli.workers)

    names = cli.configs if cli.configs else [c.name for c in app_configs.list_configs()]
    return _run_all(names, cli.volumes, cli.refresh, cli.workers)


if __name__ == "__main__":
    sys.exit(main() or 0)
