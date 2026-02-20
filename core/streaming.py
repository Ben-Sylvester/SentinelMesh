"""
Streaming Response System

Provides Server-Sent Events (SSE) for token-by-token streaming.
Dramatically improves UX by showing progressive responses.
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """Single chunk in a streaming response."""
    content: str
    done: bool = False
    metadata: Optional[dict] = None


class StreamingManager:
    """
    Manages streaming responses from LLM adapters.
    Converts adapter-specific streaming formats to unified SSE format.
    """
    
    def __init__(self):
        self.active_streams = {}
    
    async def stream_response(
        self,
        adapter,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream response from an adapter.
        
        Args:
            adapter: Model adapter that supports streaming
            prompt: Input prompt
            **kwargs: Additional adapter-specific parameters
        
        Yields:
            StreamChunk objects with partial content
        """
        try:
            # Check if adapter supports streaming
            if not hasattr(adapter, 'stream'):
                # Fallback: yield full response as single chunk
                result = await adapter.run(prompt, kwargs)
                yield StreamChunk(
                    content=result.output or "",
                    done=True,
                    metadata={
                        "tokens": result.tokens,
                        "latency_ms": result.latency_ms
                    }
                )
                return
            
            # Stream from adapter
            async for chunk in adapter.stream(prompt, **kwargs):
                yield StreamChunk(
                    content=chunk.get("content", ""),
                    done=chunk.get("done", False),
                    metadata=chunk.get("metadata")
                )
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield StreamChunk(
                content=f"[Streaming error: {e}]",
                done=True,
                metadata={"error": str(e)}
            )
    
    def format_sse(self, chunk: StreamChunk) -> str:
        """Format chunk as Server-Sent Event."""
        data = {
            "content": chunk.content,
            "done": chunk.done,
        }
        if chunk.metadata:
            data["metadata"] = chunk.metadata
        
        return f"data: {json.dumps(data)}\n\n"


# Streaming-enabled adapter mixin
class StreamingMixin:
    """
    Mixin for adapters to add streaming support.
    Adapters inherit this to enable token-by-token streaming.
    """
    
    async def stream(self, prompt: str, context: dict) -> AsyncIterator[dict]:
        """
        Stream response token-by-token.
        
        Yields:
            Dict with keys: content (str), done (bool), metadata (dict)
        """
        raise NotImplementedError("Adapter must implement stream() method")


# Example: Streaming OpenAI adapter
class StreamingOpenAIAdapter(StreamingMixin):
    """OpenAI adapter with streaming support."""
    
    def __init__(self, model: str):
        self.model = model
        self.name = f"openai:{model}"
    
    async def stream(self, prompt: str, context: dict):
        """Stream from OpenAI API."""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {
                        "content": chunk.choices[0].delta.content,
                        "done": False
                    }
            
            # Final chunk
            yield {"content": "", "done": True}
        
        except ImportError:
            # Graceful fallback if OpenAI not installed
            yield {
                "content": "[OpenAI not installed - streaming unavailable]",
                "done": True,
                "metadata": {"error": "openai_not_installed"}
            }
        except Exception as e:
            yield {
                "content": f"[Streaming error: {e}]",
                "done": True,
                "metadata": {"error": str(e)}
            }


# Example: Streaming Anthropic adapter  
class StreamingAnthropicAdapter(StreamingMixin):
    """Anthropic adapter with streaming support."""
    
    def __init__(self, model: str):
        self.model = model
        self.name = f"anthropic:{model}"
    
    async def stream(self, prompt: str, context: dict):
        """Stream from Anthropic API."""
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic()
            
            async with client.messages.stream(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield {"content": text, "done": False}
            
            yield {"content": "", "done": True}
        
        except ImportError:
            yield {
                "content": "[Anthropic not installed - streaming unavailable]",
                "done": True
            }
        except Exception as e:
            yield {
                "content": f"[Streaming error: {e}]",
                "done": True,
                "metadata": {"error": str(e)}
            }
