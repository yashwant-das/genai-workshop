"""Prompt templates for different use cases."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PromptTemplate:
    """Base prompt template structure."""

    system: str
    user: str

    def format(self, **kwargs: str) -> tuple[str, str]:
        """
        Format the prompt template with provided variables.

        Args:
            **kwargs: Variables to substitute in the template

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        return (
            self.system.format(**kwargs),
            self.user.format(**kwargs),
        )


# Summary Prompts
SUMMARY_PROMPT = PromptTemplate(
    system="You are a helpful assistant that creates concise, accurate summaries.",
    user="Summarize the following transcript in 3-5 bullet points:\n\n{transcript}",
)

DETAILED_SUMMARY_PROMPT = PromptTemplate(
    system="You are a helpful assistant that creates detailed, comprehensive summaries.",
    user="Create a detailed summary of the following transcript. Include key topics, main points, and important details:\n\n{transcript}",
)

ACTION_ITEMS_PROMPT = PromptTemplate(
    system="You are a helpful assistant that extracts action items from transcripts.",
    user="Extract all action items from the following transcript. Format each as a bullet point with the responsible person (if mentioned) and deadline (if mentioned):\n\n{transcript}",
)

# OCR and Text Extraction Prompts
OCR_CLEANUP_PROMPT = PromptTemplate(
    system="You are a helpful assistant that cleans and structures OCR text.",
    user="Clean and structure the following OCR text. Fix any obvious errors, remove artifacts, and format it properly:\n\n{ocr_text}",
)

RECEIPT_EXTRACTION_PROMPT = PromptTemplate(
    system="You are a helpful assistant that extracts structured data from receipts.",
    user="Extract the following information from this receipt text in JSON format: vendor name, date, total amount, line items (item name and price), tax amount, and currency. If any information is missing, use null:\n\n{receipt_text}",
)

# Vision and Image Analysis Prompts
IMAGE_DESCRIPTION_PROMPT = PromptTemplate(
    system="You are a helpful assistant that describes images in detail.",
    user="Based on this image description: '{image_description}', provide a detailed natural language explanation of what is shown in the image.",
)

DIAGRAM_EXPLANATION_PROMPT = PromptTemplate(
    system="You are a helpful assistant that explains diagrams and visual content.",
    user="Based on this diagram description: '{diagram_description}', provide a clear explanation of what the diagram shows. If it appears to be code or a flowchart, explain the logic or process step by step.",
)

SCREEN_QA_PROMPT = PromptTemplate(
    system="You are a helpful assistant that answers questions about screenshots and UI elements.",
    user="Based on this screenshot description: '{screenshot_description}', answer the following question: {question}",
)

# Meeting Minutes Prompt
MEETING_MINUTES_PROMPT = PromptTemplate(
    system="You are a helpful assistant that creates structured meeting minutes.",
    user="""Create structured meeting minutes from the following transcript. Format the output as JSON with the following structure:
{{
    "title": "Meeting title",
    "date": "Date if mentioned",
    "attendees": ["list of attendees"],
    "agenda": ["list of agenda items"],
    "key_points": ["list of key discussion points"],
    "decisions": ["list of decisions made"],
    "action_items": [
        {{
            "item": "Action item description",
            "assignee": "Person responsible (if mentioned)",
            "deadline": "Deadline (if mentioned)"
        }}
    ]
}}

Transcript:
{transcript}""",
)


def get_prompt(
    prompt_type: Literal[
        "summary",
        "detailed_summary",
        "action_items",
        "ocr_cleanup",
        "receipt_extraction",
        "image_description",
        "diagram_explanation",
        "screen_qa",
        "meeting_minutes",
    ],
) -> PromptTemplate:
    """
    Get a prompt template by type.

    Args:
        prompt_type: Type of prompt to retrieve

    Returns:
        PromptTemplate instance

    Raises:
        ValueError: If prompt_type is invalid
    """
    prompts = {
        "summary": SUMMARY_PROMPT,
        "detailed_summary": DETAILED_SUMMARY_PROMPT,
        "action_items": ACTION_ITEMS_PROMPT,
        "ocr_cleanup": OCR_CLEANUP_PROMPT,
        "receipt_extraction": RECEIPT_EXTRACTION_PROMPT,
        "image_description": IMAGE_DESCRIPTION_PROMPT,
        "diagram_explanation": DIAGRAM_EXPLANATION_PROMPT,
        "screen_qa": SCREEN_QA_PROMPT,
        "meeting_minutes": MEETING_MINUTES_PROMPT,
    }

    if prompt_type not in prompts:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Available types: {', '.join(prompts.keys())}")

    return prompts[prompt_type]
