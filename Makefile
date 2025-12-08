.PHONY: help install test clean

help:
	@echo "Available commands:"
	@echo "  make install    - Create venv and install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Remove build artifacts and cache"

install:
	@echo "Creating virtual environment with Python 3.13..."
	python3.13 -m venv .venv
	@echo "Installing dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Done! Activate with: source .venv/bin/activate"

test:
	.venv/bin/pytest tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

