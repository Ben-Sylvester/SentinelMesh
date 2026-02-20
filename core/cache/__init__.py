"""
Semantic Cache System

Intelligent caching based on semantic similarity.
Reduces API costs by 30-45% through response deduplication.
"""

from core.cache.semantic_cache import SemanticCache, CacheMiddleware, CacheEntry, CacheStats

__all__ = ["SemanticCache", "CacheMiddleware", "CacheEntry", "CacheStats"]
