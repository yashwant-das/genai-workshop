"""Screen reader and Q&A assistant pipeline."""

import logging
from pathlib import Path
from typing import Optional

from ..llm.client import OllamaClient
from ..llm.prompts import get_prompt
from ..utils.exceptions import ModelError
from ..vision.vision_pipeline import VisionPipeline

logger = logging.getLogger(__name__)


class ScreenQA:
    """Answer questions about screenshots and UI elements."""

    def __init__(
        self,
        vision_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Initialize screen Q&A assistant.

        Args:
            vision_model: Vision model name
            llm_model: LLM model name
        """
        self.vision_pipeline = VisionPipeline(
            vision_model=vision_model,
            llm_model=llm_model,
        )
        self.llm_client = OllamaClient(model=llm_model)
        logger.info("Initialized screen Q&A assistant")

    def answer(
        self,
        image_path: Path,
        question: str,
    ) -> str:
        """
        Answer a question about a screenshot.

        Args:
            image_path: Path to screenshot image
            question: Question to answer

        Returns:
            Answer text
        """
        logger.info(f"Answering question about screenshot: {image_path}")

        # Use vision pipeline for Q&A
        result = self.vision_pipeline.process(
            image_path,
            reasoning_type="qa",
            question=question,
        )

        answer = result.get("answer", "")

        if not answer:
            raise ModelError("Failed to generate answer")

        return answer

    def describe_screen(self, image_path: Path) -> str:
        """
        Describe what's shown in a screenshot.

        Args:
            image_path: Path to screenshot image

        Returns:
            Description text
        """
        return self.vision_pipeline.describe_only(image_path)

    def interactive_qa(
        self,
        image_path: Path,
        questions: list[str],
    ) -> dict[str, str]:
        """
        Answer multiple questions about a screenshot.

        Args:
            image_path: Path to screenshot image
            questions: List of questions to answer

        Returns:
            Dictionary mapping questions to answers
        """
        logger.info(f"Answering {len(questions)} questions about screenshot")

        # Get initial description once
        description = self.vision_pipeline.describe_only(image_path)

        # Answer each question
        answers: dict[str, str] = {}
        for question in questions:
            try:
                prompt_template = get_prompt("screen_qa")
                system, user = prompt_template.format(
                    screenshot_description=description,
                    question=question,
                )
                answer = self.llm_client.generate(prompt=user, system=system)
                answers[question] = answer
            except Exception as e:
                logger.warning(f"Failed to answer question '{question}': {e}")
                answers[question] = f"Error: {e}"

        return answers
