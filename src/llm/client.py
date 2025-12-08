"""Ollama client wrapper with streaming and error handling."""

import logging
from typing import Any, AsyncGenerator, Iterator, Optional

import ollama

from ..utils.config import DEFAULT_LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECONDS
from ..utils.exceptions import ModelError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper for Ollama client with connection handling and error management."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (defaults to DEFAULT_LLM_MODEL)
            base_url: Ollama base URL (defaults to OLLAMA_BASE_URL)
            timeout: Request timeout in seconds (defaults to OLLAMA_TIMEOUT_SECONDS)
        """
        self.model = model or DEFAULT_LLM_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.timeout = timeout or OLLAMA_TIMEOUT_SECONDS
        self._client = ollama.Client(host=self.base_url, timeout=self.timeout)

    def check_connection(self) -> bool:
        """
        Check if Ollama service is running and accessible.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self._client.list()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False

    def check_model_available(self, model: Optional[str] = None) -> bool:
        """
        Check if specified model is available locally.
        
        Args:
            model: Model name to check (defaults to self.model)
            
        Returns:
            True if model is available, False otherwise
        """
        model_name = model or self.model
        try:
            models = self._client.list()
            available_models = [m["name"] for m in models.get("models", [])]
            return model_name in available_models
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """
        Generate text using Ollama.
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            model: Model name (defaults to self.model)
            stream: Whether to stream responses
            **kwargs: Additional parameters for Ollama API
            
        Returns:
            Generated text (str) or iterator of text chunks (if stream=True)
            
        Raises:
            ModelError: If generation fails or model is unavailable
        """
        model_name = model or self.model
        
        # Check model availability
        if not self.check_model_available(model_name):
            raise ModelError(
                f"Model '{model_name}' is not available. "
                f"Install it with: ollama pull {model_name}"
            )
        
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = self._client.chat(
                model=model_name,
                messages=messages,
                stream=stream,
                **kwargs,
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response.message.content
                
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise ModelError(f"Text generation failed: {e}") from e

    def _stream_response(self, response: Any) -> Iterator[str]:
        """
        Extract text chunks from streaming response.
        
        Args:
            response: Streaming response from Ollama
            
        Yields:
            Text chunks
        """
        try:
            for chunk in response:
                if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                    content = chunk.message.content
                    if content:
                        yield content
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            raise ModelError(f"Streaming failed: {e}") from e

    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text asynchronously (for future async support).
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            model: Model name (defaults to self.model)
            **kwargs: Additional parameters
            
        Yields:
            Text chunks asynchronously
        """
        # Note: Ollama Python client doesn't have native async support yet
        # This is a placeholder for future async implementation
        response = self.generate(prompt, system, model, stream=True, **kwargs)
        for chunk in response:
            yield chunk

