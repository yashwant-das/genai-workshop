"""Complete vision processing pipeline: image → description → reasoning."""

import logging
from pathlib import Path
from typing import Optional

from ..llm.client import OllamaClient
from ..llm.prompts import get_prompt
from ..utils.exceptions import ModelError
from .analyzer import VisionAnalyzer
from .descriptor import describe_image_for_llm

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Complete pipeline for image analysis and reasoning."""
    
    def __init__(
        self,
        vision_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Initialize vision pipeline.
        
        Args:
            vision_model: Vision model name
            llm_model: LLM model name for reasoning
        """
        self.vision_analyzer = VisionAnalyzer(model=vision_model)
        self.llm_client = OllamaClient(model=llm_model)
        logger.info("Initialized vision pipeline")
    
    def process(
        self,
        image_path: Path,
        reasoning_type: str = "description",
        question: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Process image: describe and optionally reason about it.
        
        Args:
            image_path: Path to image file
            reasoning_type: Type of reasoning ("description", "diagram", "qa")
            question: Question to answer (for "qa" type)
            
        Returns:
            Dictionary with description and reasoning results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Step 1: Get image description from vision model
        try:
            image_description = describe_image_for_llm(image_path)
            logger.info("Image description generated")
        except Exception as e:
            logger.error(f"Image description failed: {e}")
            raise ModelError(f"Failed to describe image: {e}") from e
        
        result: dict[str, str] = {
            "description": image_description,
        }
        
        # Step 2: Use LLM for deeper reasoning
        try:
            match reasoning_type:
                case "description":
                    prompt_template = get_prompt("image_description")
                    system, user = prompt_template.format(
                        image_description=image_description
                    )
                    reasoning = self.llm_client.generate(prompt=user, system=system)
                    result["reasoning"] = reasoning
                    
                case "diagram":
                    prompt_template = get_prompt("diagram_explanation")
                    system, user = prompt_template.format(
                        diagram_description=image_description
                    )
                    reasoning = self.llm_client.generate(prompt=user, system=system)
                    result["explanation"] = reasoning
                    
                case "qa":
                    if not question:
                        raise ValueError("Question required for Q&A reasoning type")
                    prompt_template = get_prompt("screen_qa")
                    system, user = prompt_template.format(
                        screenshot_description=image_description,
                        question=question,
                    )
                    reasoning = self.llm_client.generate(prompt=user, system=system)
                    result["answer"] = reasoning
                    
                case _:
                    raise ValueError(f"Unknown reasoning type: {reasoning_type}")
            
            logger.info(f"Reasoning complete: {reasoning_type}")
            
        except Exception as e:
            logger.warning(f"Reasoning failed: {e}")
            result["reasoning_error"] = str(e)
        
        return result
    
    def describe_only(self, image_path: Path) -> str:
        """
        Describe image without LLM reasoning.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image description
        """
        return self.vision_analyzer.describe(image_path)

