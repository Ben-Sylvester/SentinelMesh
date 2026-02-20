"""
Streaming Response System

Provides Server-Sent Events (SSE) streaming for real-time token delivery.
Dramatically improves perceived latency and user experience.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """Single streaming chunk."""
    type: str  # "token", "metadata", "done"
    content: str
    metadata: Optional[Dict[str, Any]] = None


class StreamManager:
    """
    Manages streaming responses using Server-Sent Events (SSE).
    Provides token-by-token delivery for improved UX.
    """
    
    def __init__(self):
        self.active_streams = {}
    
    async def stream_response(
        self,
        prompt: str,
        router,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream a response token by token.
        
        Args:
            prompt: User prompt
            router: Router instance
            user_id: Optional user ID for memory
            session_id: Optional session ID
            **kwargs: Additional router parameters
        
        Yields:
            StreamChunk objects with tokens and metadata
        """
        stream_id = f"stream_{id(prompt)}_{asyncio.get_event_loop().time()}"
        self.active_streams[stream_id] = True
        
        try:
            # First, get strategy selection
            yield StreamChunk(
                type="metadata",
                content="",
                metadata={"status": "selecting_strategy"}
            )
            
            # Route the request
            # For streaming, we need to check if the selected model supports it
            result = await router.route_with_metadata(prompt, **kwargs)
            strategy_result, ctx, reward = result
            
            # Send strategy info
            yield StreamChunk(
                type="metadata",
                content="",
                metadata={
                    "status": "executing",
                    "strategy": ctx.strategy_name,
                    "model": strategy_result.models_used[0] if strategy_result.models_used else "unknown"
                }
            )
            
            # Stream the actual response
            # For now, simulate streaming by chunking the response
            # In production, this would call model.stream() directly
            output = strategy_result.output or ""
            
            # Chunk the output (simulate token-by-token)
            words = output.split()
            for i, word in enumerate(words):
                if not self.active_streams.get(stream_id):
                    break
                
                # Add space except for first word
                token = word if i == 0 else f" {word}"
                
                yield StreamChunk(
                    type="token",
                    content=token,
                    metadata=None
                )
                
                # Small delay to simulate real streaming
                await asyncio.sleep(0.01)
            
            # Send completion metadata
            yield StreamChunk(
                type="done",
                content="",
                metadata={
                    "cost_usd": strategy_result.cost_usd,
                    "latency_ms": strategy_result.latency_ms,
                    "confidence": strategy_result.confidence,
                    "reward": reward
                }
            )
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield StreamChunk(
                type="error",
                content=str(e),
                metadata={"error": str(e)}
            )
        
        finally:
            # Cleanup
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    def cancel_stream(self, stream_id: str):
        """Cancel an active stream."""
        if stream_id in self.active_streams:
            self.active_streams[stream_id] = False
    
    async def stream_with_memory(
        self,
        prompt: str,
        router,
        memory_manager,
        user_id: str,
        session_id: str,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream response with memory context injection.
        
        Combines streaming with memory system for stateful conversations.
        """
        # Recall context
        context = await memory_manager.recall_context(user_id, prompt, k=3)
        
        # Send context metadata
        yield StreamChunk(
            type="metadata",
            content="",
            metadata={
                "status": "context_loaded",
                "memory_count": context["memory_count"],
                "preferences": context["preferences"]
            }
        )
        
        # Augment prompt with context if memories found
        augmented_prompt = prompt
        if context["memories"]:
            memory_text = "\n\n".join([
                f"[Previous: {m['prompt']} -> {m['response']}]"
                for m in context["memories"][:2]
            ])
            augmented_prompt = f"Context from previous conversations:\n{memory_text}\n\nCurrent request: {prompt}"
        
        # Apply preferences
        prefs = context["preferences"]
        if prefs.get("response_length") == "short":
            augmented_prompt += "\n(Keep response concise)"
        elif prefs.get("response_length") == "long":
            augmented_prompt += "\n(Provide detailed explanation)"
        
        # Stream the response
        full_response = ""
        async for chunk in self.stream_response(augmented_prompt, router, user_id, session_id, **kwargs):
            if chunk.type == "token":
                full_response += chunk.content
            yield chunk
        
        # Store in memory after completion
        if full_response:
            await memory_manager.store_interaction(
                user_id=user_id,
                session_id=session_id,
                prompt=prompt,
                response=full_response,
                metadata={"streamed": True}
            )


class AdapterStreamWrapper:
    """
    Wrapper for model adapters to support streaming.
    Provides fallback for models that don't natively support streaming.
    """
    
    def __init__(self, adapter):
        self.adapter = adapter
        self.supports_native_streaming = hasattr(adapter, 'stream')
    
    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream tokens from adapter."""
        if self.supports_native_streaming:
            # Use native streaming
            async for token in self.adapter.stream(prompt, **kwargs):
                yield token
        else:
            # Fallback: chunk the full response
            result = await self.adapter.run(prompt, kwargs)
            output = result.output or ""
            
            # Simulate streaming by yielding words
            words = output.split()
            for i, word in enumerate(words):
                token = word if i == 0 else f" {word}"
                yield token
                await asyncio.sleep(0.01)
