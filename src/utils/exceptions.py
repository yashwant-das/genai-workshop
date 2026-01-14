"""Custom exception hierarchy for AI utilities."""


class AIUtilityError(Exception):
    """Base exception for all AI utility errors."""

    pass


class TranscriptionError(AIUtilityError):
    """Raised when audio transcription fails."""

    pass


class ModelError(AIUtilityError):
    """Raised when model operations fail (loading, inference, etc.)."""

    pass


class FileFormatError(AIUtilityError):
    """Raised when file format is unsupported or invalid."""

    pass


class ValidationError(AIUtilityError):
    """Raised when input validation fails."""

    pass


class ConfigurationError(AIUtilityError):
    """Raised when configuration is invalid or missing."""

    pass
