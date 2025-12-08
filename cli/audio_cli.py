"""Audio command handlers for CLI."""

import logging
from argparse import Namespace
from pathlib import Path

from src.audio.audio_pipeline import AudioPipeline
from src.audio.transcriber import Transcriber
from src.pipelines.meeting_minutes import MeetingMinutesGenerator
from src.utils.exceptions import AIUtilityError

logger = logging.getLogger(__name__)


def handle_audio_command(args: Namespace) -> str | dict:
    """
    Handle audio domain commands.
    
    Args:
        args: Parsed command arguments
        
    Returns:
        Command output as string or dict
    """
    command = args.command
    
    match command:
        case "transcribe":
            return handle_transcribe(args)
        case "summarize":
            return handle_summarize(args)
        case "meeting-minutes":
            return handle_meeting_minutes(args)
        case "chapters":
            return handle_chapters(args)
        case _:
            raise ValueError(f"Unknown audio command: {command}")


def handle_transcribe(args: Namespace) -> str:
    """Handle transcribe command."""
    audio_file: Path = args.audio_file
    
    try:
        transcriber = Transcriber()
        result = transcriber.transcribe(audio_file)
        
        if args.with_timestamps:
            return result.get_text_with_timestamps()
        else:
            return result.get_text_only()
            
    except AIUtilityError as e:
        logger.error(f"Transcription failed: {e}")
        raise


def handle_summarize(args: Namespace) -> str:
    """Handle summarize command."""
    audio_file: Path = args.audio_file
    style: str = args.style
    format_output: str = args.format
    
    try:
        pipeline = AudioPipeline()
        result = pipeline.process(
            audio_file,
            summarize=True,
            summary_style=style,  # type: ignore
            extract_actions=False,
        )
        
        summary = result.get("summary", "")
        
        if format_output == "text" and summary.startswith("#"):
            # Remove markdown formatting if text format requested
            lines = summary.split("\n")
            return "\n".join(line for line in lines if not line.startswith("#"))
        
        return summary
        
    except AIUtilityError as e:
        logger.error(f"Summarization failed: {e}")
        raise


def handle_meeting_minutes(args: Namespace) -> dict | str:
    """Handle meeting-minutes command."""
    audio_file: Path = args.audio_file
    output_format: str = args.format
    
    try:
        generator = MeetingMinutesGenerator()
        result = generator.generate(
            audio_file,
            output_format=output_format,  # type: ignore
        )
        
        return result
        
    except AIUtilityError as e:
        logger.error(f"Meeting minutes generation failed: {e}")
        raise


def handle_chapters(args: Namespace) -> str:
    """Handle chapters command."""
    audio_file: Path = args.audio_file
    
    try:
        transcriber = Transcriber()
        result = transcriber.transcribe(audio_file)
        
        # Extract chapter markers (segments with timestamps)
        chapters = []
        for i, segment in enumerate(result.segments, 1):
            chapters.append(
                f"Chapter {i}: {segment.start:.2f}s - {segment.end:.2f}s\n"
                f"{segment.text}\n"
            )
        
        return "\n".join(chapters)
        
    except AIUtilityError as e:
        logger.error(f"Chapter extraction failed: {e}")
        raise

