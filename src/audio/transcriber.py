"""MLX Whisper wrapper for audio transcription."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None  # type: ignore

from ..utils.config import DEFAULT_WHISPER_MODEL
from ..utils.exceptions import ModelError, TranscriptionError
from ..utils.validators import validate_audio_input

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single segment of transcribed audio."""
    
    start: float
    end: float
    text: str
    
    def __str__(self) -> str:
        """Format segment as string with timestamps."""
        return f"[{self.start:.2f}s - {self.end:.2f}s] {self.text}"


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    
    segments: list[TranscriptSegment]
    full_text: str
    language: Optional[str] = None
    
    def get_text_with_timestamps(self) -> str:
        """Get full transcript with timestamps."""
        return "\n".join(str(seg) for seg in self.segments)
    
    def get_text_only(self) -> str:
        """Get full transcript without timestamps."""
        return self.full_text


class Transcriber:
    """Wrapper for MLX Whisper transcription."""
    
    def __init__(self, model: Optional[str] = None) -> None:
        """
        Initialize transcriber.
        
        Args:
            model: Whisper model name (tiny, base, small, medium, large)
                   Defaults to DEFAULT_WHISPER_MODEL (tiny for 8GB RAM)
        """
        if mlx_whisper is None:
            raise ModelError(
                "mlx_whisper is not installed. "
                "Install it with: pip install mlx-whisper"
            )
        
        self.model_name = model or DEFAULT_WHISPER_MODEL
        self._model: Any = None
        logger.info(f"Initialized transcriber with model: {self.model_name}")
    
    def _load_model(self) -> None:
        """Lazy load the Whisper model."""
        if self._model is None:
            try:
                logger.info(f"Loading Whisper model: {self.model_name}")
                # MLX Whisper typically loads models like this
                # Adjust based on actual API
                self._model = mlx_whisper.load_model(self.model_name)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise ModelError(
                    f"Failed to load Whisper model '{self.model_name}': {e}"
                ) from e
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, auto-detect if None)
            task: Task type ("transcribe" or "translate")
            
        Returns:
            TranscriptionResult with segments and full text
            
        Raises:
            TranscriptionError: If transcription fails
            ValidationError: If audio file is invalid
        """
        validate_audio_input(audio_path)
        
        self._load_model()
        
        try:
            logger.info(f"Transcribing: {audio_path}")
            
            # MLX Whisper transcribe API
            # Adjust based on actual mlx_whisper API
            result = mlx_whisper.transcribe(
                str(audio_path),
                model=self._model,
                language=language,
                task=task,
            )
            
            # Parse result into segments
            segments = []
            full_text_parts = []
            
            # MLX Whisper typically returns segments with start, end, text
            for segment in result.get("segments", []):
                seg = TranscriptSegment(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    text=segment.get("text", "").strip(),
                )
                segments.append(seg)
                full_text_parts.append(seg.text)
            
            full_text = " ".join(full_text_parts)
            detected_language = result.get("language")
            
            logger.info(f"Transcription complete: {len(segments)} segments")
            
            return TranscriptionResult(
                segments=segments,
                full_text=full_text,
                language=detected_language,
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
    
    def transcribe_file(self, audio_path: Path) -> str:
        """
        Transcribe audio file and return text only.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        result = self.transcribe(audio_path)
        return result.get_text_only()

