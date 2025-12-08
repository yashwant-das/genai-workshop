"""Transcript summarization using LLM."""

import logging
from typing import Literal, Optional

from ..llm.client import OllamaClient
from ..llm.formatters import format_markdown_summary
from ..llm.prompts import get_prompt
from ..utils.config import MAX_TRANSCRIPT_TOKENS
from ..utils.exceptions import ModelError
from .chunker import chunk_transcript_by_tokens

logger = logging.getLogger(__name__)


class TranscriptSummarizer:
    """Summarize transcripts using LLM."""
    
    def __init__(self, model: Optional[str] = None) -> None:
        """
        Initialize summarizer.
        
        Args:
            model: LLM model name (defaults to DEFAULT_LLM_MODEL)
        """
        self.client = OllamaClient(model=model)
        logger.info("Initialized transcript summarizer")
    
    def summarize(
        self,
        transcript: str,
        style: Literal["concise", "detailed"] = "concise",
        format_output: Literal["text", "markdown"] = "text",
    ) -> str:
        """
        Summarize a transcript.
        
        Args:
            transcript: Full transcript text
            style: Summary style ("concise" or "detailed")
            format_output: Output format ("text" or "markdown")
            
        Returns:
            Summary text
        """
        prompt_type = "summary" if style == "concise" else "detailed_summary"
        prompt_template = get_prompt(prompt_type)
        
        system, user = prompt_template.format(transcript=transcript)
        
        try:
            logger.info(f"Generating {style} summary")
            summary = self.client.generate(prompt=user, system=system)
            
            if format_output == "markdown":
                return format_markdown_summary(summary, title="Summary")
            
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise ModelError(f"Failed to summarize transcript: {e}") from e
    
    def summarize_long_transcript(
        self,
        transcript: str,
        style: Literal["concise", "detailed"] = "concise",
        format_output: Literal["text", "markdown"] = "text",
    ) -> str:
        """
        Summarize a long transcript by chunking.
        
        Args:
            transcript: Full transcript text
            style: Summary style ("concise" or "detailed")
            format_output: Output format ("text" or "markdown")
            
        Returns:
            Combined summary text
        """
        # Chunk transcript if too long
        chunks = list(chunk_transcript_by_tokens(transcript, MAX_TRANSCRIPT_TOKENS))
        
        if len(chunks) == 1:
            return self.summarize(transcript, style, format_output)
        
        logger.info(f"Summarizing {len(chunks)} chunks")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Summarizing chunk {i}/{len(chunks)}")
            summary = self.summarize(chunk, style, format_output="text")
            chunk_summaries.append(summary)
        
        # Combine and create final summary
        combined = "\n\n".join(chunk_summaries)
        final_summary = self.summarize(combined, style, format_output)
        
        return final_summary
    
    def extract_action_items(self, transcript: str) -> str:
        """
        Extract action items from transcript.
        
        Args:
            transcript: Full transcript text
            
        Returns:
            List of action items (formatted text)
        """
        prompt_template = get_prompt("action_items")
        system, user = prompt_template.format(transcript=transcript)
        
        try:
            logger.info("Extracting action items")
            action_items = self.client.generate(prompt=user, system=system)
            return action_items
            
        except Exception as e:
            logger.error(f"Action item extraction failed: {e}")
            raise ModelError(f"Failed to extract action items: {e}") from e

