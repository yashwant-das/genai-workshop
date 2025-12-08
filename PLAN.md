# Plan: Local Offline AI Utilities & Model Exercises

## Initial Request & Context

### Why We Are Building This

I am learning GenAI and want to build **local offline AI utilities** that run entirely on my machine using:

- **MLX Whisper** for audio transcription
- **Ollama** for LLM reasoning and vision tasks

The goal is to create a suite of practical utilities that demonstrate and test all capabilities of the available models while staying within the constraints of my M1 8GB Mac.

### Machine Setup

- **OS**: macOS on Apple Silicon (M1 8GB RAM)
- **Python**: 3.13.11
- **Virtual Environment**: `~/.ai` (activated manually)
- **Installed Packages**:
  - `mlx-whisper` (for audio transcription)
  - `ollama` (Python client for local LLMs)
- **Installed Models via Ollama**:
  - `llama3.2:latest` (text LLM, 2.0 GB)
  - `moondream:latest` (vision→text model, 1.7 GB)
- **GPU**: Apple M1 Neural Engine (via MLX)
- **Constraints**: Avoid unnecessary RAM usage, use small models, avoid heavy dependencies, no CUDA

### Target Applications

1. **Audio → Transcript → Summary pipelines**
   - Meeting transcription and summarization
   - Action item extraction
   - Chapter markers with timestamps

2. **Image analysis or OCR → reasoning with llama3.2**
   - Receipt parsing and data extraction
   - Screenshot analysis
   - Document understanding

3. **Vision → text → deeper reasoning pipelines (moondream + llama3.2)**
   - Whiteboard photo → meeting minutes
   - Diagram → explanation
   - UI screenshot → navigation guidance

4. **Local assistants**
   - Meeting minutes generator
   - Receipt parser
   - Diagram explainer
   - Screen reader/Q&A assistant

### Requirements

- All code must run entirely locally (no external API calls)
- Target Python 3.13+ with modern best practices
- Minimal dependencies (only use packages already installed unless explicitly requested)
- Memory-efficient for 8GB M1 Mac
- Clean, modular code with proper project structure
- Comprehensive testing of all model capabilities

---

## Project Structure

```
gen-ai-testing/
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── transcriber.py      # MLX Whisper wrapper
│   │   ├── summarizer.py       # Audio → Summary pipeline
│   │   └── chunker.py          # Audio chunking utilities
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── analyzer.py         # Moondream vision wrapper
│   │   ├── ocr.py              # OCR extraction
│   │   └── descriptor.py       # Image description
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # Ollama client wrapper
│   │   ├── prompts.py          # Prompt templates
│   │   └── formatters.py       # Output formatting (JSON, markdown)
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── audio_pipeline.py   # Audio → Transcript → Summary
│   │   ├── vision_pipeline.py   # Image → Description → Reasoning
│   │   ├── meeting_minutes.py  # Meeting minutes generator
│   │   ├── receipt_parser.py   # Receipt extraction
│   │   ├── diagram_explainer.py # Diagram explanation
│   │   └── multimodal.py       # Combined pipelines
│   └── utils/
│       ├── __init__.py
│       ├── file_handlers.py    # File I/O, format detection
│       ├── validators.py       # Input validation
│       └── config.py           # Configuration management
├── cli/
│   ├── __init__.py
│   ├── audio_cli.py            # Audio command handlers
│   ├── vision_cli.py           # Vision command handlers
│   └── main.py                 # Main CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_audio/
│   │   ├── test_transcriber.py
│   │   └── test_summarizer.py
│   ├── test_vision/
│   │   ├── test_analyzer.py
│   │   └── test_ocr.py
│   ├── test_llm/
│   │   └── test_client.py
│   ├── test_pipelines/
│   │   ├── test_audio_pipeline.py
│   │   └── test_vision_pipeline.py
│   └── fixtures/               # Test audio/images
│       ├── audio/
│       └── images/
├── examples/
│   ├── audio_examples.py
│   ├── vision_examples.py
│   └── pipeline_examples.py
├── docs/
│   ├── README.md
│   ├── API.md
│   └── USAGE.md
├── .gitignore
├── pyproject.toml              # Python 3.13 project config
├── requirements.txt            # Minimal dependencies
├── README.md                   # Project overview
└── setup.py                    # Optional: package installation
```

## Python 3.13 Best Practices

- **Type Hints**: Full type annotations using `typing.Protocol`, `typing.TypedDict`, `typing.Literal`, `typing.Union`
- **Modern Syntax**: Pattern matching (`match/case`), `functools.cache`, structural pattern matching
- **Error Handling**: Custom exception hierarchy, proper error messages, graceful degradation
- **Async Support**: Consider `asyncio` for I/O-bound operations (file reading, API calls)
- **Path Handling**: Use `pathlib.Path` exclusively (Python 3.13 optimized)
- **Data Classes**: Use `dataclasses` or `typing.NamedTuple` for structured data
- **Logging**: Structured logging with `logging` module, appropriate log levels
- **Documentation**: Type hints + docstrings (Google/NumPy style) for all public APIs

## Implementation Steps

### 1. Project Setup & Configuration

- Create complete directory structure with `__init__.py` files
- Set up `pyproject.toml` with Python 3.13 requirement (`requires-python = ">=3.13"`), project metadata, optional build config
- Create `requirements.txt` with minimal deps: `ollama`, `mlx-whisper` (no extras)
- Add comprehensive `.gitignore` for Python, virtual envs, test artifacts, model caches, `__pycache__`
- Create base `src/utils/config.py` for model names, default prompts, resource limits, environment variable support

### 2. Core Modules (src/)

**Audio Module (`src/audio/`)**
- `transcriber.py`: Wrapper around `mlx_whisper.transcribe()` with error handling, format validation, timestamp extraction, type hints
- `summarizer.py`: Functions to chunk transcripts, call llama3.2 for summaries/action items with streaming support
- `chunker.py`: Audio file chunking utilities (time-based, size-based) for memory efficiency on M1 8GB

**Vision Module (`src/vision/`)**
- `analyzer.py`: Moondream wrapper for image analysis, object detection, scene description with proper error handling
- `ocr.py`: OCR extraction using moondream, text cleanup utilities, structured text extraction
- `descriptor.py`: Image description generation with structured output options

**LLM Module (`src/llm/`)**
- `client.py`: Ollama client wrapper with streaming support, error handling, retry logic, connection pooling
- `prompts.py`: Prompt templates as TypedDict/dataclasses for each use case (summaries, OCR cleanup, Q&A, structured extraction)
- `formatters.py`: Output formatters (JSON extraction, markdown, structured text) with validation

**Pipelines Module (`src/pipelines/`)**
- `audio_pipeline.py`: Complete audio → transcript → summary pipeline with chunking
- `vision_pipeline.py`: Image → moondream → llama3.2 reasoning pipeline
- `meeting_minutes.py`: Meeting minutes generator (audio → structured minutes with attendees, agenda, decisions)
- `receipt_parser.py`: Receipt parser (image → JSON with vendor, date, total, line items)
- `diagram_explainer.py`: Diagram explainer (image → natural language explanation)
- `multimodal.py`: Combined pipelines and assistant mode

**Utils Module (`src/utils/`)**
- `file_handlers.py`: File format detection, validation, path utilities using `pathlib`
- `validators.py`: Input validation (file exists, format supported, size limits for M1 8GB)
- `config.py`: Configuration management (environment variables, defaults, model selection)

### 3. CLI Interface (`cli/`)

- `main.py`: Main entry point using `argparse` (Python stdlib), command routing, help text
- `audio_cli.py`: Commands: `transcribe`, `summarize`, `chapters`, `qa` with proper argument parsing
- `vision_cli.py`: Commands: `describe`, `ocr`, `extract-receipt`, `analyze-diagram`, `vision-qa`
- All CLI functions use type hints, proper error messages, progress indicators, output formatting

### 4. Utilities Implementation

**Meeting Minutes Generator** (`src/pipelines/meeting_minutes.py`)
- Pipeline: Audio → Transcript → Structured summary (attendees, agenda, decisions, action items)
- Output: Markdown or JSON format with timestamps
- CLI: `python -m cli.main audio meeting-minutes <audio_file>`

**Receipt Parser** (`src/pipelines/receipt_parser.py`)
- Pipeline: Image → Moondream OCR → llama3.2 structured extraction
- Output: JSON with vendor, date, total, line items, tax, currency
- CLI: `python -m cli.main vision extract-receipt <image_file>`

**Diagram Explainer** (`src/pipelines/diagram_explainer.py`)
- Pipeline: Image → Moondream description → llama3.2 explanation
- Output: Natural language explanation, pseudocode (if code diagram), structured breakdown
- CLI: `python -m cli.main vision analyze-diagram <image_file>`

**Screen Reader/Q&A** (`src/pipelines/screen_qa.py`)
- Pipeline: Screenshot → Moondream describe → llama3.2 answer user question
- Interactive Q&A mode or single question mode
- CLI: `python -m cli.main vision qa <image_file> --question "How do I export?"`

### 5. Testing Strategy (`tests/`)

**Unit Tests**
- Test each module independently with mocks for ollama/whisper
- Test file handlers, validators, formatters with edge cases
- Test prompt templates for correctness
- Use `unittest` or `pytest` (if added to requirements)

**Integration Tests**
- Test full pipelines with small fixtures in `tests/fixtures/`
- Test error handling (invalid files, model failures, network issues)
- Test memory efficiency with chunking on sample files
- Test output format validation

**Test Matrix**
- Audio: Short (<30s), medium (2-5min), long (10+ min), noisy, clean, different languages (if supported)
- Images: Receipts, screenshots, diagrams, whiteboards, code screenshots, natural scenes
- Edge cases: Empty files, unsupported formats, corrupted data, very large files, missing models

### 6. Documentation

- `README.md`: Project overview, installation, quick start, examples, project structure
- `docs/API.md`: API documentation for all modules with type signatures
- `docs/USAGE.md`: Detailed usage examples for each utility with CLI examples
- Inline docstrings: All public functions/classes with type hints + Google-style docstrings

### 7. Model Capability Testing

**MLX Whisper**
- Transcription accuracy (short/medium/long audio)
- Timestamp precision and segment boundaries
- Multi-language support (if applicable)
- Memory usage with chunking strategies
- Error handling for unsupported formats

**Moondream**
- Object detection accuracy and bounding boxes
- Scene description quality and detail level
- OCR text extraction accuracy
- Visual Q&A capabilities with various question types
- Image format support (JPEG, PNG, HEIC)
- Handling of low-quality or ambiguous images

**Llama3.2**
- Summary quality (concise vs detailed modes)
- Structured JSON extraction accuracy
- Multi-turn conversation capabilities
- Code generation (if needed for explanations)
- Context window management (chunking long inputs)
- Prompt engineering variations (system prompts, few-shot examples)
- Streaming response handling

## Deliverables

- Complete project structure with all modules and proper Python 3.13 practices
- Working CLI with all utilities (`python -m cli.main <command> <args>`)
- Comprehensive test suite with fixtures
- Complete documentation (README, API docs, usage examples)
- Example scripts demonstrating each capability
- Prompt templates optimized for each use case in `src/llm/prompts.py`
- Error handling and edge case coverage throughout
- Type hints and docstrings for maintainability

