# GenAI Workshop: Local AI Experiments

An experimental collection of privacy-focused local AI utilities for audio transcription, vision analysis, and LLM reasoning. This project runs entirely offline on your Apple Silicon (M-series) Mac using MLX Whisper and Ollama.

## Key Features

### Audio Intelligence
- **High-Fidelity Transcription**: Powered by MLX Whisper (Apple Silicon optimized).
- **Intelligent Summarization**: Extract key points and action items from recordings.
- **Meeting Minutes**: Generate structured summaries with attendees, decisions, and deadlines.
- **Chapter Extraction**: Automatically segment long audio files into logical chapters.

### Vision Analysis
- **Image Description**: Natural language explanations of visual content.
- **OCR & Data Extraction**: Extract raw text or structured data (e.g., from receipts).
- **Diagram Explanation**: Step-by-step breakdown of logic flows and charts.
- **Screen Q&A**: Contextual answers based on screenshots and UI elements.

---

## Getting Started

### Prerequisites
- **Hardware**: macOS on Apple Silicon (M1/M2/M3/M4 or later)
- **Software**: Python 3.13 and [Ollama](https://ollama.ai/) installed and running.

### Installation

1. **Clone & Enter**:
   ```bash
   git clone <repository-url>
   cd genai-workshop
   ```

2. **Setup Environment**:
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Models**:
   ```bash
   ollama pull llama3.2:latest   # For text reasoning
   ollama pull moondream:latest  # For vision tasks
   ```

---

## Configuration

Manage behavior via environment variables. Defaults are optimized for baseline Apple Silicon Macs (e.g., 8GB RAM).

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL` | LLM model name | `llama3.2:latest` |
| `VISION_MODEL` | Vision model name | `moondream:latest` |
| `WHISPER_MODEL` | Whisper model size (`tiny`, `base`, `small`) | `tiny` |
| `AUDIO_CHUNK_SIZE` | Processing chunk size (seconds) | `30` |
| `OLLAMA_TIMEOUT` | Timeout for API requests (seconds) | `30` |
| `LOG_LEVEL` | Verbosity (`DEBUG`, `INFO`, `WARNING`) | `INFO` |

> [!TIP]
> Use a `.env` file or export variables in your shell to override these defaults.

---

## Usage

### Command Line Interface (CLI)

The easiest way to use the utilities is via the `genai` command (or `python -m cli.main`).

#### Audio Examples
```bash
# Transcribe
python -m cli.main audio transcribe meeting.wav

# Summarize to Markdown
python -m cli.main audio summarize meeting.wav --style detailed --output summary.md

# Extract Chapters
python -m cli.main audio chapters podcast.mp3
```

#### Vision Examples
```bash
# Describe image
python -m cli.main vision describe samples/images/sample_image.png

# Parse Receipt (JSON)
python -m cli.main vision extract-receipt receipt.png --format json

# Technical Diagram Explanation
python -m cli.main vision analyze-diagram arch.png --detail technical
```

### Python API

Integrate the pipelines directly into your own Python projects.

```python
from src.audio.audio_pipeline import AudioPipeline
from src.vision.vision_pipeline import VisionPipeline
from pathlib import Path

# Process Audio with Summarization
audio = AudioPipeline()
result = audio.process(Path("meeting.wav"), summarize=True)
print(f"Summary: {result['summary']}")

# Analyze an Image
vision = VisionPipeline()
description = vision.describe_only(Path("diagram.png"))
print(f"Analysis: {description}")
```

---

## Technical Architecture

### Project Structure
```text
genai-workshop/
├── cli/                 # Command-line interface entry points
├── src/
│   ├── audio/           # MLX Whisper integration & audio processing
│   ├── vision/          # Moondream vision model integration
│   ├── llm/             # Ollama client & prompt management
│   ├── pipelines/       # Complex multi-step workflows
│   └── utils/           # Shared config & file handlers
└── samples/             # Sample audio and image files for testing
```

### Performance & Memory Optimization
Designed to run efficiently on base-model Apple Silicon (M1/M2/M3/M4+ with 8GB RAM):
- **Chunked Processing**: Audio is processed in 30-second segments to minimize peak memory.
- **Efficient Models**: Defaults to `tiny` Whisper and optimized Ollama models.
- **Memory Safety**: Enforces file size limits (500MB Audio / 10MB Image) to prevent swap thrashing.

---

## Development

### Linting & Formatting
This project uses **Ruff** for code style and quality.
```bash
# Check for issues and sort imports
python -m ruff check .

# Apply formatting
python -m ruff format .
```

### Contributing
1. Fork the repository and create your feature branch.
2. Ensure documentation is updated.
3. Submit a Pull Request.

---

## License
This project is licensed under the [MIT License](LICENSE).
