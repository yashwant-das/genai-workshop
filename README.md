# GenAI Workshop

Local offline AI utilities using MLX Whisper and Ollama for audio transcription, vision analysis, and LLM reasoning.

## Overview

This project provides a suite of practical AI utilities that run entirely on your local machine:

- **Audio Transcription**: Transcribe audio files using MLX Whisper
- **Audio Summarization**: Generate summaries and extract action items from transcripts
- **Meeting Minutes**: Create structured meeting minutes from audio recordings
- **Image Analysis**: Describe images and extract text using vision models
- **Receipt Parsing**: Extract structured data from receipt images
- **Diagram Explanation**: Explain diagrams and visual content
- **Screen Q&A**: Answer questions about screenshots and UI elements

All processing runs locally using:

- **MLX Whisper** for audio transcription (Apple Silicon optimized)
- **Ollama** for LLM reasoning and vision tasks

## Requirements

- **OS**: macOS on Apple Silicon (M1/M2/M3)
- **Python**: 3.13+
- **Virtual Environment**: Recommended (e.g., `~/.ai`)
- **Installed Packages**:
  - `ollama` (Python client)
  - `mlx-whisper` (for audio transcription)
  - `pytest` (for testing)

- **Ollama Models** (install via `ollama pull`):
  - `llama3.2:latest` (text LLM, ~2.0 GB)
  - `moondream:latest` (vision model, ~1.7 GB)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd genai-workshop
```

1. Create and activate the project virtual environment (Python 3.13):

```bash
python3.13 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Ensure Ollama is running and models are installed:

```bash
# Check Ollama is running
ollama list

# Install models if needed
ollama pull llama3.2:latest
ollama pull moondream:latest
```

### Alternative: Use Makefile

```bash
make install  # Creates venv and installs dependencies
```

## Quick Start

### Audio Commands

**Transcribe audio:**

```bash
python genai audio transcribe audio.wav
```

**Transcribe with timestamps:**

```bash
python genai audio transcribe audio.wav --with-timestamps
```

**Summarize audio:**

```bash
python genai audio summarize audio.wav --style concise
```

**Generate meeting minutes:**

```bash
python genai audio meeting-minutes meeting.wav --format json
```

**Extract chapter markers:**

```bash
python genai audio chapters podcast.wav
```

### Vision Commands

**Describe an image:**

```bash
python genai vision describe image.jpg
```

**Extract text (OCR):**

```bash
python genai vision ocr document.png
```

**Extract receipt data:**

```bash
python genai vision extract-receipt receipt.jpg --format json
```

**Analyze a diagram:**

```bash
python genai vision analyze-diagram diagram.png --detail technical
```

**Answer questions about a screenshot:**

```bash
python genai vision qa screenshot.png --question "How do I export this file?"
```

### Output to File

All commands support `--output` to save results to a file:

```bash
python genai audio summarize audio.wav --output summary.md
```

**Note:** After installing the package in development mode (`pip install -e .`), you can also use:

```bash
genai audio transcribe audio.wav
```

## Project Structure

```text
genai-workshop/
├── src/
│   ├── audio/          # Audio transcription and summarization
│   ├── vision/         # Image analysis and OCR
│   ├── llm/            # LLM client and prompts
│   ├── pipelines/      # Complete workflows
│   └── utils/          # Utilities and configuration
├── cli/                # Command-line interface
├── tests/              # Test suite
├── examples/           # Example scripts
└── docs/               # Documentation
```

## Configuration

Configuration is managed via environment variables and defaults in `src/utils/config.py`:

- `LLM_MODEL`: LLM model name (default: `llama3.2:latest`)
- `VISION_MODEL`: Vision model name (default: `moondream:latest`)
- `WHISPER_MODEL`: Whisper model size (default: `tiny`)
- `AUDIO_CHUNK_SIZE`: Audio chunk size in seconds (default: `30`)
- `MAX_TRANSCRIPT_TOKENS`: Max tokens per transcript chunk (default: `2000`)

## Usage Examples

### Python API

```python
from src.audio.audio_pipeline import AudioPipeline
from src.vision.vision_pipeline import VisionPipeline
from pathlib import Path

# Process audio
pipeline = AudioPipeline()
result = pipeline.process(
    Path("meeting.wav"),
    summarize=True,
    summary_style="detailed"
)
print(result["summary"])

# Process image
vision = VisionPipeline()
description = vision.describe_only(Path("diagram.png"))
print(description)
```

## Memory Considerations

This project is optimized for 8GB M1 Macs:

- Uses smallest Whisper model (`tiny`) by default
- Chunks long transcripts for LLM processing
- Processes audio in 30-second chunks
- Limits file sizes (500MB audio, 10MB images)

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]
