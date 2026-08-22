"""Machine-independent path resolution.

Image data lives outside the repository so that git worktrees stay disposable and
the same checkout works on every machine. Resolution order for the data root:

1. ``UNA_DATA_ROOT`` environment variable
2. ``data_root`` key in ``una.config.json`` at the repository root (gitignored)
3. ``<repo>/data`` as a last resort

Every path this module hands out is absolute, so callers never depend on the
current working directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SPLITS = ("train", "test")

CONFIG_FILENAME = "una.config.json"
ENV_DATA_ROOT = "UNA_DATA_ROOT"


def repo_root() -> Path:
    """Return the repository root, located by walking up from this file."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"could not locate repository root above {here}")


def _config() -> dict:
    path = repo_root() / CONFIG_FILENAME
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return loaded


def data_root() -> Path:
    """Return the root directory holding downloaded and derived data."""
    env_value = os.environ.get(ENV_DATA_ROOT)
    if env_value:
        return Path(env_value).expanduser().resolve()

    configured = _config().get("data_root")
    if configured:
        return Path(str(configured)).expanduser().resolve()

    return repo_root() / "data"


def coco_json(split: str) -> Path:
    """Return the organizer-provided COCO file for ``split``.

    These are committed to the repository, so they resolve relative to the repo
    rather than the data root.
    """
    _check_split(split)
    return repo_root() / "fathomnet-2026" / f"{split}_dataset.json"


def images_dir(split: str) -> Path:
    """Return the directory holding original downloaded images for ``split``."""
    _check_split(split)
    return data_root() / "images" / split


def cache_dir(split: str) -> Path:
    """Return the directory holding preprocessed (resized) images for ``split``."""
    _check_split(split)
    return data_root() / "cache" / split


def index_path(split: str) -> Path:
    """Return the path of the local availability index written by ``doctor``."""
    _check_split(split)
    return data_root() / "index" / f"{split}.local_index.json"


def runs_dir() -> Path:
    """Return the directory holding experiment outputs."""
    return data_root() / "runs"


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


def describe() -> dict:
    """Return a JSON-serializable summary of every resolved path."""
    source = "default"
    if os.environ.get(ENV_DATA_ROOT):
        source = f"env:{ENV_DATA_ROOT}"
    elif _config().get("data_root"):
        source = f"config:{CONFIG_FILENAME}"

    return {
        "repo_root": str(repo_root()),
        "data_root": str(data_root()),
        "data_root_source": source,
        "splits": {
            split: {
                "coco_json": str(coco_json(split)),
                "images_dir": str(images_dir(split)),
                "cache_dir": str(cache_dir(split)),
                "index_path": str(index_path(split)),
            }
            for split in SPLITS
        },
    }
