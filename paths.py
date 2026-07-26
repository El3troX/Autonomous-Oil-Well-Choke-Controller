"""Shared filesystem helpers for repository data and image artifacts."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"


def data_path(name: str) -> Path:
    """Return the absolute path for a file under the data directory."""
    return DATA_DIR / name


def image_path(name: str) -> Path:
    """Return the absolute path for a file under the images directory."""
    return IMAGES_DIR / name


def resolve_data_path(path_like):
    """Resolve a dataset path, falling back to the data directory."""
    candidate = Path(path_like)
    if candidate.exists():
        return candidate

    data_candidate = DATA_DIR / candidate.name
    if data_candidate.exists():
        return data_candidate

    return candidate


def resolve_image_path(path_like):
    """Resolve an image path, falling back to the images directory."""
    candidate = Path(path_like)
    if candidate.exists():
        return candidate

    image_candidate = IMAGES_DIR / candidate.name
    if image_candidate.exists():
        return image_candidate

    return candidate
