import os
import shutil
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


def save_file(file_bytes: bytes, relative_path: str) -> str:
    """Save file to local disk. Returns full path."""
    full_path = os.path.join(settings.local_storage_path, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved: {full_path}")
    return full_path


def read_file(relative_path: str) -> bytes:
    """Read file from local disk."""
    full_path = os.path.join(settings.local_storage_path, relative_path)
    with open(full_path, "rb") as f:
        return f.read()


def delete_file(relative_path: str):
    """Delete file from local disk."""
    full_path = os.path.join(settings.local_storage_path, relative_path)
    if os.path.exists(full_path):
        os.remove(full_path)
        logger.info(f"Deleted: {full_path}")


def delete_folder(relative_path: str):
    """Delete entire folder."""
    full_path = os.path.join(settings.local_storage_path, relative_path)
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
        logger.info(f"Deleted folder: {full_path}")


def file_exists(relative_path: str) -> bool:
    full_path = os.path.join(settings.local_storage_path, relative_path)
    return os.path.exists(full_path)