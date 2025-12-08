"""Setup script for genai-workshop package."""

from setuptools import find_packages, setup

setup(
    name="genai-workshop",
    version="0.1.0",
    description="Local offline AI utilities using MLX Whisper and Ollama",
    author="Yashwant",
    packages=find_packages(),
    python_requires=">=3.13",
    install_requires=[
        "ollama>=0.1.0",
        "mlx-whisper>=0.1.0",
        "pytest>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "genai=cli.main:main",
        ],
    },
)

