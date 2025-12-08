"""Complete audio processing pipeline: audio → transcript → summary."""

import logging
from pathlib import Path
from typing import Literal, Optional

from .summarizer import TranscriptSummarizer
from .transcriber import TranscriptionResult, Transcriber
from ..utils.exceptions import TranscriptionError

logger = logging.getLogger(__name__)


class AudioPipeline:
    """Complete pipeline for audio transcription and summarization."""
    
    def __init__(
        self,
        whisper_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Initialize audio pipeline.
        
        Args:
            whisper_model: Whisper model name
            llm_model: LLM model name for summarization
        """
        self.transcriber = Transcriber(model=whisper_model)
        self.summarizer = TranscriptSummarizer(model=llm_model)
        logger.info("Initialized audio pipeline")
    
    def process(
        self,
        audio_path: Path,
        summarize: bool = True,
        summary_style: Literal["concise", "detailed"] = "concise",
        extract_actions: bool = False,
    ) -> dict[str, str | TranscriptionResult]:
        """
        Process audio file: transcribe and optionally summarize.
        
        Args:
            audio_path: Path to audio file
            summarize: Whether to generate summary
            summary_style: Summary style if summarizing
            extract_actions: Whether to extract action items
            
        Returns:
            Dictionary with transcript, summary (if requested), and action items (if requested)
        """
        logger.info(f"Processing audio: {audio_path}")
        
        # Step 1: Transcribe
        try:
            transcript_result = self.transcriber.transcribe(audio_path)
            transcript_text = transcript_result.get_text_only()
            logger.info("Transcription complete")
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise TranscriptionError(f"Failed to transcribe audio: {e}") from e
        
        result: dict[str, str | TranscriptionResult] = {
            "transcript": transcript_result,
            "transcript_text": transcript_text,
        }
        
        # Step 2: Summarize (if requested)
        if summarize:
            try:
                summary = self.summarizer.summarize_long_transcript(
                    transcript_text,
                    style=summary_style,
                    format_output="markdown",
                )
                result["summary"] = summary
                logger.info("Summary generated")
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
                result["summary"] = "Summary generation failed."
        
        # Step 3: Extract action items (if requested)
        if extract_actions:
            try:
                action_items = self.summarizer.extract_action_items(transcript_text)
                result["action_items"] = action_items
                logger.info("Action items extracted")
            except Exception as e:
                logger.warning(f"Action item extraction failed: {e}")
                result["action_items"] = "Action item extraction failed."
        
        return result
    
    def transcribe_only(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe audio without summarization.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            TranscriptionResult
        """
        return self.transcriber.transcribe(audio_path)

