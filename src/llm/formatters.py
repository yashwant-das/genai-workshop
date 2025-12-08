"""Output formatting utilities for JSON, markdown, and structured text."""

import json
import logging
import re
from typing import Any, Literal

from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict[str, Any] | list[Any]:
    """
    Extract JSON from text that may contain markdown code blocks or extra text.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON as dict or list
        
    Raises:
        ValidationError: If JSON cannot be extracted or parsed
    """
    # Try to find JSON in code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object/array directly
        json_match = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Use entire text
            json_str = text.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise ValidationError(f"Failed to extract valid JSON from text: {e}") from e


def format_markdown_summary(text: str, title: str = "Summary") -> str:
    """
    Format text as a markdown summary.
    
    Args:
        text: Summary text
        title: Title for the summary
        
    Returns:
        Formatted markdown string
    """
    return f"# {title}\n\n{text}\n"


def format_json_output(data: dict[str, Any] | list[Any], indent: int = 2) -> str:
    """
    Format data as pretty-printed JSON string.
    
    Args:
        data: Data to format
        indent: Indentation level
        
    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def format_structured_output(
    data: dict[str, Any] | list[Any],
    output_format: Literal["json", "markdown"] = "json",
) -> str:
    """
    Format structured data in the specified format.
    
    Args:
        data: Data to format
        output_format: Desired output format
        
    Returns:
        Formatted output string
    """
    match output_format:
        case "json":
            return format_json_output(data)
        case "markdown":
            return _dict_to_markdown(data) if isinstance(data, dict) else str(data)
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")


def _dict_to_markdown(data: dict[str, Any], level: int = 1) -> str:
    """
    Convert dictionary to markdown format.
    
    Args:
        data: Dictionary to convert
        level: Heading level (1-6)
        
    Returns:
        Markdown formatted string
    """
    lines: list[str] = []
    
    for key, value in data.items():
        # Format key as heading or bold
        if level <= 6:
            heading = "#" * level
            lines.append(f"{heading} {key}")
        else:
            lines.append(f"**{key}**")
        
        # Format value
        if isinstance(value, dict):
            lines.append(_dict_to_markdown(value, level + 1))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    lines.append(_dict_to_markdown(item, level + 1))
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(f"{value}\n")
    
    return "\n".join(lines) + "\n"


def validate_json_structure(
    data: dict[str, Any],
    required_keys: list[str],
    optional_keys: list[str] | None = None,
) -> None:
    """
    Validate that a dictionary contains required keys.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        optional_keys: List of optional keys (for documentation)
        
    Raises:
        ValidationError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValidationError(
            f"Missing required keys in JSON structure: {', '.join(missing_keys)}"
        )

