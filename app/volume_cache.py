"""
In-memory cache of decoded volumes.

Decoding is not cheap -- a 512x512x343 .nii.gz costs ~3.5s, which is most of the
time a dataset or volume switch takes once the process no longer restarts. Keep
the decoded arrays around and switching back to one you have already seen is
instant.

Deliberately free of any `configs.configs_parser` import: `config_swap` reloads
every module that holds config-derived state, and this cache must SURVIVE those
reloads (a cache that is thrown away on each switch is no cache at all).

Keyed on path + mtime + size, so editing a volume on disk misses correctly.
"""

import pathlib

# Roughly 5 parse2022 volumes (each ~90 MB decoded). Tune if the demo machine is
# tight on RAM; eviction is least-recently-used.
MAX_BYTES = 1024 * 1024 * 1024

_entries = {}       # key -> (volume, name, source_ext)
_order = []         # keys, least-recently-used first
_bytes = 0


def _key(path):
    p = pathlib.Path(path)
    stat = p.stat()
    return (str(p.resolve()).lower(), stat.st_mtime_ns, stat.st_size)


def _evict_to_fit(incoming: int):
    global _bytes
    while _order and _bytes + incoming * 2 > MAX_BYTES:      # volume + its buffer
        oldest = _order.pop(0)
        volume, _, _ = _entries.pop(oldest)
        _coords.pop(oldest, None)
        _working.pop(oldest, None)
        _bytes -= volume.nbytes


def load(loader, path):
    """`loader(path) -> (volume, name, source_ext)`, memoised.

    The cached array is handed out read-only: callers derive their own state
    from it (DemoState copies), and a stray in-place write would otherwise
    corrupt every later load of the same file.
    """
    try:
        key = _key(path)
    except OSError:
        return loader(path)                     # unreadable stat -> just load it

    global _bytes
    if key in _entries:
        _order.remove(key)
        _order.append(key)
        return _entries[key]

    volume, name, source_ext = loader(path)
    try:
        volume.flags.writeable = False
    except (AttributeError, ValueError):
        pass                                    # not a numpy array, or a view

    if volume.nbytes <= MAX_BYTES:
        _evict_to_fit(volume.nbytes)
        _entries[key] = (volume, name, source_ext)
        _order.append(key)
        _bytes += volume.nbytes
    return volume, name, source_ext


_working = {}       # key -> mutable scratch buffer the size of the volume


def working_copy(path, original):
    """A mutable copy of `original`, reusing a per-file buffer.

    The demo's working volume must be writable (reconstructions OR into it), but
    allocating a fresh 90MB array inside the server costs ~1s -- the pages have
    to be faulted in every time, and it dominated a dataset switch. Copying into
    a buffer whose pages are already resident is ~40x cheaper.
    """
    import numpy as np
    if path is None:
        return original.copy()                  # uploaded volume: no stable key
    try:
        key = _key(path)
    except OSError:
        return original.copy()
    buffer = _working.get(key)
    if buffer is None or buffer.shape != original.shape or buffer.dtype != original.dtype:
        buffer = np.empty_like(original)
        _working[key] = buffer
    np.copyto(buffer, original)
    return buffer


_coords = {}        # key -> (N,3) int array of occupied voxels


def occupied(path, volume):
    """Occupied-voxel coordinates of `volume`, memoised per file.

    np.argwhere over a 512x512x343 volume costs ~1s -- it scans all 89.9M
    voxels regardless of how few are set -- and the original volume's occupancy
    never changes once loaded. Cached against the same key as the decode, so
    coming back to a volume costs nothing. `path=None` (an uploaded volume, no
    stable identity) just computes it.
    """
    import numpy as np
    if path is None:
        return np.argwhere(volume > 0.5).astype(np.int32)
    try:
        key = _key(path)
    except OSError:
        return np.argwhere(volume > 0.5).astype(np.int32)
    if key not in _coords:
        _coords[key] = np.argwhere(volume > 0.5).astype(np.int32)
    return _coords[key]


def stats() -> dict:
    return {"volumes": len(_entries), "mb": round(_bytes / 1024 / 1024, 1),
            "budget_mb": round(MAX_BYTES / 1024 / 1024)}


def clear():
    global _bytes
    _entries.clear()
    _order.clear()
    _coords.clear()
    _working.clear()
    _bytes = 0
