"""Diagram explanation pipeline."""

import logging
from pathlib import Path
from typing import Literal, Optional

from ..llm.client import OllamaClient
from ..llm.prompts import get_prompt
from ..utils.exceptions import ModelError
from ..vision.vision_pipeline import VisionPipeline

logger = logging.getLogger(__name__)


class DiagramExplainer:
    """Explain diagrams and visual content."""
    
    def __init__(
        self,
        vision_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Initialize diagram explainer.
        
        Args:
            vision_model: Vision model name
            llm_model: LLM model name
        """
        self.vision_pipeline = VisionPipeline(
            vision_model=vision_model,
            llm_model=llm_model,
        )
        self.llm_client = OllamaClient(model=llm_model)
        logger.info("Initialized diagram explainer")
    
    def explain(
        self,
        image_path: Path,
        detail_level: Literal["basic", "detailed", "technical"] = "detailed",
    ) -> str:
        """
        Explain a diagram or visual content.
        
        Args:
            image_path: Path to diagram image
            detail_level: Level of detail in explanation
            
        Returns:
            Explanation text
        """
        logger.info(f"Explaining diagram: {image_path}")
        
        # Use vision pipeline to get description and explanation
        result = self.vision_pipeline.process(
            image_path,
            reasoning_type="diagram",
        )
        
        explanation = result.get("explanation", result.get("description", ""))
        
        if not explanation:
            raise ModelError("Failed to generate diagram explanation")
        
        # Enhance explanation based on detail level
        if detail_level == "technical":
            # Request more technical details
            prompt_template = get_prompt("diagram_explanation")
            system, user = prompt_template.format(
                diagram_description=result["description"]
            )
            user += "\n\nProvide a more technical explanation with specific details about the components, relationships, and logic."
            
            try:
                enhanced = self.llm_client.generate(prompt=user, system=system)
                return enhanced
            except Exception as e:
                logger.warning(f"Technical enhancement failed: {e}")
                return explanation
        
        return explanation

