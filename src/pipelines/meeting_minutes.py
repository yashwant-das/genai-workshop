"""Meeting minutes generator pipeline."""

import logging
from pathlib import Path
from typing import Any, Literal, Optional

from ..audio.audio_pipeline import AudioPipeline
from ..llm.client import OllamaClient
from ..llm.formatters import extract_json, format_structured_output
from ..llm.prompts import get_prompt
from ..utils.exceptions import ModelError

logger = logging.getLogger(__name__)


class MeetingMinutesGenerator:
    """Generate structured meeting minutes from audio."""

    def __init__(
        self,
        whisper_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Initialize meeting minutes generator.

        Args:
            whisper_model: Whisper model name
            llm_model: LLM model name
        """
        self.audio_pipeline = AudioPipeline(
            whisper_model=whisper_model,
            llm_model=llm_model,
        )
        self.llm_client = OllamaClient(model=llm_model)
        logger.info("Initialized meeting minutes generator")

    def generate(
        self,
        audio_path: Path,
        output_format: Literal["json", "markdown"] = "json",
    ) -> dict[str, Any] | str:
        """
        Generate meeting minutes from audio file.

        Args:
            audio_path: Path to audio file
            output_format: Output format ("json" or "markdown")

        Returns:
            Meeting minutes as dict (if json) or formatted string
        """
        logger.info(f"Generating meeting minutes from: {audio_path}")

        # Step 1: Transcribe audio
        transcript_result = self.audio_pipeline.transcribe_only(audio_path)
        transcript_text = transcript_result.get_text_only()

        # Step 2: Generate structured minutes using LLM
        prompt_template = get_prompt("meeting_minutes")
        system, user = prompt_template.format(transcript=transcript_text)

        try:
            logger.info("Generating structured meeting minutes")
            response = self.llm_client.generate(prompt=user, system=system)

            # Extract JSON from response
            minutes_data = extract_json(response)

            # Add metadata
            if isinstance(minutes_data, dict):
                minutes_data["metadata"] = {
                    "source_file": str(audio_path),
                    "transcript_length": len(transcript_text),
                    "segments_count": len(transcript_result.segments),
                }

            # Format output
            if output_format == "markdown":
                return format_structured_output(minutes_data, output_format="markdown")
            else:
                return minutes_data

        except Exception as e:
            logger.error(f"Meeting minutes generation failed: {e}")
            raise ModelError(f"Failed to generate meeting minutes: {e}") from e
