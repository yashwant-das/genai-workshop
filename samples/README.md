# Samples Directory

This directory contains sample files for testing and demonstration purposes.

## Structure

- `images/` - Sample images for vision processing (screenshots, diagrams, receipts, etc.)
- `audio/` - Sample audio files for transcription and summarization

## Usage

You can use these files to test the CLI commands:

```bash
# Test vision commands
python genai vision describe samples/images/Screenshot.png
python genai vision ocr samples/images/receipt.jpg

# Test audio commands
python genai audio transcribe samples/audio/meeting.wav
python genai audio summarize samples/audio/podcast.mp3
```

## Note

Files in this directory are tracked in git (unlike files in the root directory).
Add your test files here to keep the project root clean.

