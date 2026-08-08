"""
Dataset descriptors for the interactive demo -- the `app/configs/` folder.

Each `app/configs/*.yaml` names one selectable dataset:

    LABEL:               human-readable name shown in the UI
    ORDER:               optional sort key for the dropdown
    CONFIG_FILENAME:     the configs/ file configs_parser.py must load
    VOLUMES_PATH:        folder scanned to populate the volume picker
    DEFAULT_VOLUME_PATH: volume loaded when this dataset is selected

Dropping a new .yaml in that folder adds a dataset -- nothing here or in
server.py hardcodes the list.

IMPORTANT: this module must stay free of `configs.configs_parser` (and torch)
imports. The server's supervisor process reads descriptors to decide which
CONFIG_FILENAME to hand the worker, and that decision has to be made *before*
configs_parser is ever imported -- see the note in server.py.
"""

import pathlib

import yaml

ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent
DESCRIPTORS_PATH = pathlib.Path(__file__).absolute().parent.joinpath("configs")

# 3D formats convert_data_file_to_numpy() understands (.png is 2D -> excluded).
VOLUME_EXTENSIONS = (".nii.gz", ".npy", ".npz", ".pcd", ".ply", ".off", ".obj", ".binvox")


def _resolve(path) -> pathlib.Path:
    """Descriptor path (repo-relative, either slash style) -> absolute path."""
    if path is None:
        return None
    p = pathlib.Path(str(path).replace("\\", "/"))
    return p if p.is_absolute() else ROOT_PATH.joinpath(p)


def _rel(path) -> str:
    """Absolute path -> repo-relative posix string (what the API speaks)."""
    try:
        return pathlib.Path(path).relative_to(ROOT_PATH).as_posix()
    except ValueError:
        return pathlib.Path(path).as_posix()


class AppConfig:
    """One entry of app/configs/, with its paths resolved against the repo root."""

    def __init__(self, name: str, data: dict):
        self.name = name                                        # descriptor filename stem
        self.label = data.get("LABEL") or name
        self.order = data.get("ORDER", 999)
        self.config_filename = data.get("CONFIG_FILENAME")
        self.volumes_path = _resolve(data.get("VOLUMES_PATH"))
        self.default_volume_path = _resolve(data.get("DEFAULT_VOLUME_PATH"))

    @property
    def config_filepath(self) -> pathlib.Path:
        return ROOT_PATH.joinpath("configs", self.config_filename) if self.config_filename else None

    def problems(self) -> list:
        """Why this dataset can't be served (empty list == selectable)."""
        issues = []
        if not self.config_filename:
            issues.append("CONFIG_FILENAME is missing")
        elif not self.config_filepath.is_file():
            issues.append(f"config not found: configs/{self.config_filename}")
        if self.volumes_path is None:
            issues.append("VOLUMES_PATH is missing")
        elif not self.volumes_path.is_dir():
            issues.append(f"volumes folder not found: {_rel(self.volumes_path)}")
        elif not self.volumes():
            issues.append(f"no volumes in {_rel(self.volumes_path)}")
        return issues

    def volumes(self) -> list:
        """Selectable volumes in VOLUMES_PATH, as [{name, path}] (path repo-relative)."""
        if self.volumes_path is None or not self.volumes_path.is_dir():
            return []
        found = [f for f in sorted(self.volumes_path.iterdir())
                 if f.is_file() and str(f).lower().endswith(VOLUME_EXTENSIONS)]
        return [{"name": f.name, "path": _rel(f)} for f in found]

    def find_volume(self, requested) -> pathlib.Path:
        """
        Strict lookup: the requested volume, or None if it is not one of ours.

        Accepted only if it lives in VOLUMES_PATH -- the picker offers a list of
        choices, not an arbitrary file-open. Callers acting on an explicit user
        request should use this and report a failure, rather than quietly
        substituting a different volume.
        """
        if not requested:
            return None
        candidate = _resolve(requested)
        allowed = {_resolve(v["path"]) for v in self.volumes()}
        return candidate if (candidate in allowed and candidate.is_file()) else None

    def resolve_volume(self, requested=None) -> pathlib.Path:
        """
        Pick a volume to load at STARTUP, tolerating a missing/unusable request:
        the request if valid, else DEFAULT_VOLUME_PATH, else the first volume in
        VOLUMES_PATH, else None (which lets load_demo_volume() fall back to the
        config's own dataset folders).
        """
        found = self.find_volume(requested)
        if found is not None:
            return found
        if self.default_volume_path is not None and self.default_volume_path.is_file():
            return self.default_volume_path
        volumes = self.volumes()
        return _resolve(volumes[0]["path"]) if volumes else None

    def to_dict(self) -> dict:
        issues = self.problems()
        return {
            "name": self.name,
            "label": self.label,
            "config_filename": self.config_filename,
            "volumes": self.volumes(),
            "default_volume": _rel(self.default_volume_path) if self.default_volume_path else None,
            "available": not issues,
            "problems": issues,
        }


def list_configs() -> list:
    """Every descriptor in app/configs/, ordered by ORDER then label."""
    if not DESCRIPTORS_PATH.is_dir():
        return []
    configs = []
    for path in sorted(DESCRIPTORS_PATH.glob("*.yaml")):
        try:
            with open(path, "r") as stream:
                data = yaml.safe_load(stream) or {}
        except Exception as exc:
            print(f"[Configs] Skipping {path.name}: {exc}")
            continue
        configs.append(AppConfig(name=path.stem, data=data))
    return sorted(configs, key=lambda c: (c.order, c.label))


def get_config(name: str) -> AppConfig:
    """Descriptor by name (the .yaml stem), or None."""
    return next((c for c in list_configs() if c.name == name), None)


def default_config() -> AppConfig:
    """First selectable descriptor; falls back to the first one that exists."""
    configs = list_configs()
    if not configs:
        raise FileNotFoundError(f"No dataset descriptors found in {DESCRIPTORS_PATH}")
    return next((c for c in configs if not c.problems()), configs[0])


if __name__ == "__main__":
    # Quick check of what the demo can offer on this machine.
    for cfg in list_configs():
        issues = cfg.problems()
        mark = "OK " if not issues else "XX "
        print(f"{mark} {cfg.name:<20} {cfg.label:<32} {cfg.config_filename}")
        print(f"     volumes: {len(cfg.volumes())} in {_rel(cfg.volumes_path) if cfg.volumes_path else '?'}")
        selected = cfg.resolve_volume()
        print(f"     default: {_rel(selected) if selected else '(none)'}")
        for issue in issues:
            print(f"     ! {issue}")
