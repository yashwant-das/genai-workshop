"""Vision command handlers for CLI."""

import logging
from argparse import Namespace
from pathlib import Path

from src.pipelines.diagram_explainer import DiagramExplainer
from src.pipelines.receipt_parser import ReceiptParser
from src.pipelines.screen_qa import ScreenQA
from src.vision.ocr import OCRExtractor
from src.vision.vision_pipeline import VisionPipeline
from src.utils.exceptions import AIUtilityError

logger = logging.getLogger(__name__)


def handle_vision_command(args: Namespace) -> str | dict:
    """
    Handle vision domain commands.
    
    Args:
        args: Parsed command arguments
        
    Returns:
        Command output as string or dict
    """
    command = args.command
    
    match command:
        case "describe":
            return handle_describe(args)
        case "ocr":
            return handle_ocr(args)
        case "extract-receipt":
            return handle_extract_receipt(args)
        case "analyze-diagram":
            return handle_analyze_diagram(args)
        case "qa":
            return handle_qa(args)
        case _:
            raise ValueError(f"Unknown vision command: {command}")


def handle_describe(args: Namespace) -> str:
    """Handle describe command."""
    image_file: Path = args.image_file
    
    try:
        pipeline = VisionPipeline()
        description = pipeline.describe_only(image_file)
        return description
        
    except AIUtilityError as e:
        logger.error(f"Image description failed: {e}")
        raise


def handle_ocr(args: Namespace) -> str:
    """Handle ocr command."""
    image_file: Path = args.image_file
    
    try:
        extractor = OCRExtractor()
        text = extractor.extract_text(image_file, cleanup=True)
        return text
        
    except AIUtilityError as e:
        logger.error(f"OCR extraction failed: {e}")
        raise


def handle_extract_receipt(args: Namespace) -> dict | str:
    """Handle extract-receipt command."""
    image_file: Path = args.image_file
    output_format: str = args.format
    
    try:
        parser = ReceiptParser()
        result = parser.parse(
            image_file,
            output_format=output_format,  # type: ignore
        )
        return result
        
    except AIUtilityError as e:
        logger.error(f"Receipt parsing failed: {e}")
        raise


def handle_analyze_diagram(args: Namespace) -> str:
    """Handle analyze-diagram command."""
    image_file: Path = args.image_file
    detail: str = args.detail
    
    try:
        explainer = DiagramExplainer()
        explanation = explainer.explain(
            image_file,
            detail_level=detail,  # type: ignore
        )
        return explanation
        
    except AIUtilityError as e:
        logger.error(f"Diagram analysis failed: {e}")
        raise


def handle_qa(args: Namespace) -> str:
    """Handle qa command."""
    image_file: Path = args.image_file
    question: str = args.question
    
    try:
        qa = ScreenQA()
        answer = qa.answer(image_file, question)
        return answer
        
    except AIUtilityError as e:
        logger.error(f"Q&A failed: {e}")
        raise

