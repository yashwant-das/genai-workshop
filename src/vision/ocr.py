"""OCR extraction using vision models."""

import logging
from pathlib import Path
from typing import Optional

from ..llm.client import OllamaClient
from ..llm.formatters import extract_json
from ..llm.prompts import get_prompt
from ..utils.exceptions import ModelError
from .analyzer import VisionAnalyzer

logger = logging.getLogger(__name__)


class OCRExtractor:
    """Extract text from images using vision models."""
    
    def __init__(
        self,
        vision_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Initialize OCR extractor.
        
        Args:
            vision_model: Vision model name for image analysis
            llm_model: LLM model name for text cleanup
        """
        self.vision_analyzer = VisionAnalyzer(model=vision_model)
        self.llm_client = OllamaClient(model=llm_model) if llm_model else None
        logger.info("Initialized OCR extractor")
    
    def extract_text(
        self,
        image_path: Path,
        cleanup: bool = True,
    ) -> str:
        """
        Extract text from image.
        
        Args:
            image_path: Path to image file
            cleanup: Whether to use LLM to clean up extracted text
            
        Returns:
            Extracted text
        """
        try:
            logger.info(f"Extracting text from: {image_path}")
            
            # Use vision model to extract text
            # Moondream can describe text in images
            prompt = "Extract all text visible in this image. Include all words, numbers, and labels exactly as they appear."
            extracted_text = self.vision_analyzer.answer_question(image_path, prompt)
            
            # Clean up text using LLM if requested
            if cleanup and self.llm_client:
                try:
                    prompt_template = get_prompt("ocr_cleanup")
                    system, user = prompt_template.format(ocr_text=extracted_text)
                    cleaned_text = self.llm_client.generate(prompt=user, system=system)
                    logger.info("Text cleaned using LLM")
                    return cleaned_text
                except Exception as e:
                    logger.warning(f"Text cleanup failed: {e}, using raw extraction")
                    return extracted_text
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise ModelError(f"Failed to extract text from image: {e}") from e
    
    def extract_structured_text(
        self,
        image_path: Path,
        structure_type: str = "receipt",
    ) -> dict:
        """
        Extract structured text from image (e.g., receipt data).
        
        Args:
            image_path: Path to image file
            structure_type: Type of structure to extract ("receipt", etc.)
            
        Returns:
            Structured data as dictionary
        """
        # First extract raw text
        raw_text = self.extract_text(image_path, cleanup=True)
        
        # Then use LLM to structure it
        if not self.llm_client:
            raise ModelError("LLM client required for structured extraction")
        
        try:
            if structure_type == "receipt":
                prompt_template = get_prompt("receipt_extraction")
                system, user = prompt_template.format(receipt_text=raw_text)
                
                response = self.llm_client.generate(prompt=user, system=system)
                structured_data = extract_json(response)
                
                logger.info("Structured data extracted")
                return structured_data
            else:
                raise ValueError(f"Unknown structure type: {structure_type}")
                
        except Exception as e:
            logger.error(f"Structured extraction failed: {e}")
            raise ModelError(f"Failed to extract structured data: {e}") from e

