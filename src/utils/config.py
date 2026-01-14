"""Configuration management with environment variable support."""

import os
from pathlib import Path
from typing import Literal

# Model Configuration
DEFAULT_LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:latest")
DEFAULT_VISION_MODEL: str = os.getenv("VISION_MODEL", "moondream:latest")
DEFAULT_WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "tiny")  # tiny, base, small

# Audio Processing Configuration
AUDIO_CHUNK_SIZE_SECONDS: int = int(os.getenv("AUDIO_CHUNK_SIZE", "30"))
MAX_AUDIO_DURATION_HOURS: float = float(os.getenv("MAX_AUDIO_DURATION", "1.0"))

# Transcript Processing Configuration
MAX_TRANSCRIPT_TOKENS: int = int(os.getenv("MAX_TRANSCRIPT_TOKENS", "2000"))

# File Size Limits (for 8GB M1 Mac)
MAX_AUDIO_FILE_SIZE_MB: int = int(os.getenv("MAX_AUDIO_FILE_SIZE_MB", "500"))
MAX_IMAGE_FILE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_FILE_SIZE_MB", "10"))

# Supported File Formats
SUPPORTED_AUDIO_FORMATS: set[str] = {".wav", ".mp3", ".m4a", ".flac"}
SUPPORTED_IMAGE_FORMATS: set[str] = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

# Ollama Configuration
OLLAMA_TIMEOUT_SECONDS: int = int(os.getenv("OLLAMA_TIMEOUT", "30"))
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Output Format Defaults
DEFAULT_SUMMARY_FORMAT: Literal["markdown", "json"] = os.getenv("DEFAULT_SUMMARY_FORMAT", "markdown")
DEFAULT_STRUCTURED_FORMAT: Literal["json", "markdown"] = os.getenv("DEFAULT_STRUCTURED_FORMAT", "json")

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Paths
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
TESTS_FIXTURES_DIR: Path = PROJECT_ROOT / "tests" / "fixtures"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)
