"""File I/O utilities and format detection."""

import mimetypes
from pathlib import Path
from typing import Literal

from .config import SUPPORTED_AUDIO_FORMATS, SUPPORTED_IMAGE_FORMATS
from .exceptions import FileFormatError


def detect_file_type(file_path: Path) -> Literal["audio", "image", "unknown"]:
    """
    Detect file type from extension and MIME type.

    Args:
        file_path: Path to the file

    Returns:
        File type: "audio", "image", or "unknown"
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    # Check by extension first
    if suffix in SUPPORTED_AUDIO_FORMATS:
        return "audio"
    if suffix in SUPPORTED_IMAGE_FORMATS:
        return "image"

    # Fallback to MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type:
        if mime_type.startswith("audio/"):
            return "audio"
        if mime_type.startswith("image/"):
            return "image"

    return "unknown"


def validate_audio_file(file_path: Path) -> None:
    """
    Validate that file is a supported audio format.

    Args:
        file_path: Path to the audio file

    Raises:
        FileFormatError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    file_type = detect_file_type(file_path)
    if file_type != "audio":
        raise FileFormatError(
            f"Unsupported audio format: {file_path.suffix}. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )


def validate_image_file(file_path: Path) -> None:
    """
    Validate that file is a supported image format.

    Args:
        file_path: Path to the image file

    Raises:
        FileFormatError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    file_type = detect_file_type(file_path)
    if file_type != "image":
        raise FileFormatError(
            f"Unsupported image format: {file_path.suffix}. Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
        )


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in megabytes
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)
