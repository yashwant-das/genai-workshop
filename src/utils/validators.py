"""Input validation utilities."""

from pathlib import Path

from .config import (
    MAX_AUDIO_FILE_SIZE_MB,
    MAX_IMAGE_FILE_SIZE_MB,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
)
from .exceptions import ValidationError
from .file_handlers import get_file_size_mb, validate_audio_file, validate_image_file


def validate_file_exists(file_path: Path) -> None:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to validate
        
    Raises:
        ValidationError: If file does not exist
    """
    if not file_path.exists():
        raise ValidationError(f"File does not exist: {file_path}")
    
    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")


def validate_audio_input(file_path: Path) -> None:
    """
    Validate audio file input (exists, format, size).
    
    Args:
        file_path: Path to audio file
        
    Raises:
        ValidationError: If validation fails
        FileFormatError: If format is unsupported
    """
    validate_file_exists(file_path)
    validate_audio_file(file_path)
    
    file_size_mb = get_file_size_mb(file_path)
    if file_size_mb > MAX_AUDIO_FILE_SIZE_MB:
        raise ValidationError(
            f"Audio file too large: {file_size_mb:.2f} MB "
            f"(max: {MAX_AUDIO_FILE_SIZE_MB} MB)"
        )


def validate_image_input(file_path: Path) -> None:
    """
    Validate image file input (exists, format, size).
    
    Args:
        file_path: Path to image file
        
    Raises:
        ValidationError: If validation fails
        FileFormatError: If format is unsupported
    """
    validate_file_exists(file_path)
    validate_image_file(file_path)
    
    file_size_mb = get_file_size_mb(file_path)
    if file_size_mb > MAX_IMAGE_FILE_SIZE_MB:
        raise ValidationError(
            f"Image file too large: {file_size_mb:.2f} MB "
            f"(max: {MAX_IMAGE_FILE_SIZE_MB} MB)"
        )


def validate_output_path(output_path: Path, create_parents: bool = True) -> None:
    """
    Validate and prepare output path.
    
    Args:
        output_path: Path for output file
        create_parents: Whether to create parent directories
        
    Raises:
        ValidationError: If path is invalid
    """
    if output_path.exists() and output_path.is_dir():
        raise ValidationError(f"Output path is a directory: {output_path}")
    
    if create_parents:
        output_path.parent.mkdir(parents=True, exist_ok=True)

