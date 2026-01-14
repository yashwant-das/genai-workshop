"""Image description generation utilities."""

import logging
from pathlib import Path
from typing import Literal

from .analyzer import VisionAnalyzer

logger = logging.getLogger(__name__)


def describe_image(
    image_path: Path,
    detail_level: Literal["low", "medium", "high"] = "medium",
) -> str:
    """
    Generate description of an image.

    Args:
        image_path: Path to image file
        detail_level: Level of detail in description

    Returns:
        Image description text
    """
    analyzer = VisionAnalyzer()
    return analyzer.describe(image_path, detail=detail_level)


def describe_image_for_llm(image_path: Path) -> str:
    """
    Generate detailed description suitable for LLM reasoning.

    Args:
        image_path: Path to image file

    Returns:
        Detailed image description
    """
    analyzer = VisionAnalyzer()
    return analyzer.describe(image_path, detail="high")
