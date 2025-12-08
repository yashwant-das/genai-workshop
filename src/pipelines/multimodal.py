"""Multimodal pipelines combining audio and vision."""

import logging
from pathlib import Path
from typing import Any, Optional

from ..audio.audio_pipeline import AudioPipeline
from ..vision.vision_pipeline import VisionPipeline
from ..llm.client import OllamaClient
from ..llm.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class MultimodalPipeline:
    """Combine audio and vision processing."""
    
    def __init__(
        self,
        whisper_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Initialize multimodal pipeline.
        
        Args:
            whisper_model: Whisper model name
            vision_model: Vision model name
            llm_model: LLM model name
        """
        self.audio_pipeline = AudioPipeline(whisper_model=whisper_model, llm_model=llm_model)
        self.vision_pipeline = VisionPipeline(vision_model=vision_model, llm_model=llm_model)
        self.llm_client = OllamaClient(model=llm_model)
        logger.info("Initialized multimodal pipeline")
    
    def process_whiteboard_meeting(
        self,
        audio_path: Path,
        whiteboard_image_path: Path,
    ) -> dict[str, Any]:
        """
        Process a meeting with whiteboard: combine audio transcript and whiteboard image.
        
        Args:
            audio_path: Path to meeting audio
            whiteboard_image_path: Path to whiteboard photo
            
        Returns:
            Combined meeting information
        """
        logger.info("Processing whiteboard meeting")
        
        # Process audio
        audio_result = self.audio_pipeline.process(audio_path, summarize=True)
        
        # Process whiteboard image
        whiteboard_description = self.vision_pipeline.describe_only(whiteboard_image_path)
        
        # Combine using LLM
        combined_prompt = f"""Based on this meeting transcript and whiteboard description, create comprehensive meeting notes.

Meeting Transcript:
{audio_result.get('transcript_text', '')}

Whiteboard Description:
{whiteboard_description}

Create structured meeting notes that combine information from both sources."""
        
        try:
            combined_notes = self.llm_client.generate(
                prompt=combined_prompt,
                system="You are a helpful assistant that creates comprehensive meeting notes from multiple sources.",
            )
            
            return {
                "transcript": audio_result.get("transcript_text", ""),
                "whiteboard_description": whiteboard_description,
                "combined_notes": combined_notes,
                "summary": audio_result.get("summary", ""),
            }
        except Exception as e:
            logger.error(f"Failed to combine sources: {e}")
            return {
                "transcript": audio_result.get("transcript_text", ""),
                "whiteboard_description": whiteboard_description,
                "combined_notes": "Failed to combine sources.",
                "error": str(e),
            }

