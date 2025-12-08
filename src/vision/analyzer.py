"""Moondream wrapper for vision analysis."""

import base64
import logging
from pathlib import Path
from typing import Any, Optional

import ollama

from ..utils.config import DEFAULT_VISION_MODEL, OLLAMA_BASE_URL
from ..utils.exceptions import ModelError
from ..utils.validators import validate_image_input

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Wrapper for Moondream vision model via Ollama."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize vision analyzer.
        
        Args:
            model: Vision model name (defaults to DEFAULT_VISION_MODEL)
            base_url: Ollama base URL (defaults to OLLAMA_BASE_URL)
        """
        self.model = model or DEFAULT_VISION_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self._client = ollama.Client(host=self.base_url)
        logger.info(f"Initialized vision analyzer with model: {self.model}")
    
    def _encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 for Ollama API.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode("utf-8")
    
    def describe(
        self,
        image_path: Path,
        detail: str = "medium",
    ) -> str:
        """
        Generate description of image.
        
        Args:
            image_path: Path to image file
            detail: Detail level ("low", "medium", "high")
            
        Returns:
            Image description text
            
        Raises:
            ModelError: If analysis fails
        """
        validate_image_input(image_path)
        
        try:
            logger.info(f"Describing image: {image_path}")
            
            # Moondream via Ollama uses vision models
            # The API typically requires base64 encoded images
            image_base64 = self._encode_image(image_path)
            
            response = self._client.generate(
                model=self.model,
                prompt="Describe this image in detail.",
                images=[image_base64],
            )
            
            description = response.response.strip() if response.response else ""
            logger.info("Image description generated")
            
            return description
            
        except Exception as e:
            logger.error(f"Image description failed: {e}")
            raise ModelError(f"Failed to describe image: {e}") from e
    
    def answer_question(
        self,
        image_path: Path,
        question: str,
    ) -> str:
        """
        Answer a question about the image.
        
        Args:
            image_path: Path to image file
            question: Question to ask about the image
            
        Returns:
            Answer text
            
        Raises:
            ModelError: If analysis fails
        """
        validate_image_input(image_path)
        
        try:
            logger.info(f"Answering question about image: {image_path}")
            
            image_base64 = self._encode_image(image_path)
            
            response = self._client.generate(
                model=self.model,
                prompt=question,
                images=[image_base64],
            )
            
            answer = response.response.strip() if response.response else ""
            logger.info("Question answered")
            
            return answer
            
        except Exception as e:
            logger.error(f"Visual Q&A failed: {e}")
            raise ModelError(f"Failed to answer question: {e}") from e
    
    def detect_objects(self, image_path: Path) -> list[dict[str, Any]]:
        """
        Detect objects in image (if supported by model).
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detected objects with bounding boxes (if available)
            
        Note:
            Moondream may not support object detection with bounding boxes.
            This is a placeholder for future enhancement.
        """
        validate_image_input(image_path)
        
        # Moondream primarily does description, not object detection
        # This could be enhanced with a specialized model
        description = self.describe(image_path)
        
        # Return description as a single "object" for now
        return [
            {
                "type": "scene",
                "description": description,
                "confidence": 1.0,
            }
        ]

