"""Main CLI entry point."""

import argparse
import logging
import sys
from pathlib import Path

from . import audio_cli, vision_cli

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def setup_parser() -> argparse.ArgumentParser:
    """Set up main argument parser."""
    parser = argparse.ArgumentParser(
        description="Local offline AI utilities using MLX Whisper and Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="domain", help="Domain commands")
    subparsers.required = True
    
    # Audio subcommands
    audio_parser = subparsers.add_parser("audio", help="Audio processing commands")
    audio_subparsers = audio_parser.add_subparsers(dest="command", help="Audio commands")
    audio_subparsers.required = True
    
    # Audio: transcribe
    transcribe_parser = audio_subparsers.add_parser(
        "transcribe",
        help="Transcribe audio file",
    )
    transcribe_parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to audio file",
    )
    transcribe_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    transcribe_parser.add_argument(
        "--with-timestamps",
        action="store_true",
        help="Include timestamps in output",
    )
    
    # Audio: summarize
    summarize_parser = audio_subparsers.add_parser(
        "summarize",
        help="Transcribe and summarize audio file",
    )
    summarize_parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to audio file",
    )
    summarize_parser.add_argument(
        "--style",
        choices=["concise", "detailed"],
        default="concise",
        help="Summary style",
    )
    summarize_parser.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="markdown",
        help="Output format",
    )
    summarize_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    
    # Audio: meeting-minutes
    minutes_parser = audio_subparsers.add_parser(
        "meeting-minutes",
        help="Generate meeting minutes from audio",
    )
    minutes_parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to audio file",
    )
    minutes_parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format",
    )
    minutes_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    
    # Audio: chapters
    chapters_parser = audio_subparsers.add_parser(
        "chapters",
        help="Extract chapter markers from audio",
    )
    chapters_parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to audio file",
    )
    chapters_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    
    # Vision subcommands
    vision_parser = subparsers.add_parser("vision", help="Vision processing commands")
    vision_subparsers = vision_parser.add_subparsers(dest="command", help="Vision commands")
    vision_subparsers.required = True
    
    # Vision: describe
    describe_parser = vision_subparsers.add_parser(
        "describe",
        help="Describe an image",
    )
    describe_parser.add_argument(
        "image_file",
        type=Path,
        help="Path to image file",
    )
    describe_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    
    # Vision: ocr
    ocr_parser = vision_subparsers.add_parser(
        "ocr",
        help="Extract text from image",
    )
    ocr_parser.add_argument(
        "image_file",
        type=Path,
        help="Path to image file",
    )
    ocr_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    
    # Vision: extract-receipt
    receipt_parser = vision_subparsers.add_parser(
        "extract-receipt",
        help="Extract structured data from receipt image",
    )
    receipt_parser.add_argument(
        "image_file",
        type=Path,
        help="Path to receipt image",
    )
    receipt_parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format",
    )
    receipt_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    
    # Vision: analyze-diagram
    diagram_parser = vision_subparsers.add_parser(
        "analyze-diagram",
        help="Analyze and explain a diagram",
    )
    diagram_parser.add_argument(
        "image_file",
        type=Path,
        help="Path to diagram image",
    )
    diagram_parser.add_argument(
        "--detail",
        choices=["basic", "detailed", "technical"],
        default="detailed",
        help="Detail level",
    )
    diagram_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    
    # Vision: qa
    qa_parser = vision_subparsers.add_parser(
        "qa",
        help="Answer questions about a screenshot",
    )
    qa_parser.add_argument(
        "image_file",
        type=Path,
        help="Path to screenshot image",
    )
    qa_parser.add_argument(
        "--question",
        required=True,
        help="Question to answer",
    )
    qa_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )
    
    # Global options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser


def write_output(content: str, output_path: Path | None) -> None:
    """
    Write content to file or stdout.
    
    Args:
        content: Content to write
        output_path: Output file path (None for stdout)
    """
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"Output written to: {output_path}")
    else:
        print(content)


def main() -> int:
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Route to appropriate handler
        if args.domain == "audio":
            result = audio_cli.handle_audio_command(args)
        elif args.domain == "vision":
            result = vision_cli.handle_vision_command(args)
        else:
            parser.print_help()
            return 1
        
        # Write output
        if isinstance(result, str):
            write_output(result, getattr(args, "output", None))
        elif isinstance(result, dict):
            import json
            output = json.dumps(result, indent=2, ensure_ascii=False)
            write_output(output, getattr(args, "output", None))
        else:
            write_output(str(result), getattr(args, "output", None))
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

