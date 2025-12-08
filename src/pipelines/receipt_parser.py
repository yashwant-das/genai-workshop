"""Receipt parser pipeline."""

import logging
from pathlib import Path
from typing import Any, Literal, Optional

from ..llm.client import OllamaClient
from ..llm.formatters import extract_json, format_structured_output
from ..llm.prompts import get_prompt
from ..utils.exceptions import ModelError
from ..vision.ocr import OCRExtractor

logger = logging.getLogger(__name__)


class ReceiptParser:
    """Parse receipts from images and extract structured data."""
    
    def __init__(
        self,
        vision_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Initialize receipt parser.
        
        Args:
            vision_model: Vision model name
            llm_model: LLM model name
        """
        self.ocr_extractor = OCRExtractor(
            vision_model=vision_model,
            llm_model=llm_model,
        )
        self.llm_client = OllamaClient(model=llm_model)
        logger.info("Initialized receipt parser")
    
    def parse(
        self,
        image_path: Path,
        output_format: Literal["json", "markdown"] = "json",
    ) -> dict[str, Any] | str:
        """
        Parse receipt from image and extract structured data.
        
        Args:
            image_path: Path to receipt image
            output_format: Output format ("json" or "markdown")
            
        Returns:
            Receipt data as dict (if json) or formatted string
        """
        logger.info(f"Parsing receipt from: {image_path}")
        
        # Step 1: Extract text using OCR
        try:
            receipt_text = self.ocr_extractor.extract_text(image_path, cleanup=True)
            logger.info("Receipt text extracted")
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise ModelError(f"Failed to extract receipt text: {e}") from e
        
        # Step 2: Extract structured data using LLM
        prompt_template = get_prompt("receipt_extraction")
        system, user = prompt_template.format(receipt_text=receipt_text)
        
        try:
            logger.info("Extracting structured receipt data")
            response = self.llm_client.generate(prompt=user, system=system)
            
            # Extract JSON from response
            receipt_data = extract_json(response)
            
            # Add metadata
            if isinstance(receipt_data, dict):
                receipt_data["metadata"] = {
                    "source_file": str(image_path),
                    "raw_text_length": len(receipt_text),
                }
            
            # Format output
            if output_format == "markdown":
                return format_structured_output(receipt_data, output_format="markdown")
            else:
                return receipt_data
                
        except Exception as e:
            logger.error(f"Receipt parsing failed: {e}")
            raise ModelError(f"Failed to parse receipt: {e}") from e

