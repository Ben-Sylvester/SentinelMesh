"""
Configuration Management System

Centralized configuration with validation, environment support,
and sensible defaults. Production-ready configuration management.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: str = "sqlite"  # sqlite, postgres, mysql
    host: str = "localhost"
    port: int = 5432
    name: str = "sentinelmesh"
    user: str = "admin"
    password: str = ""
    path: str = "data/sentinelmesh.db"  # For SQLite


@dataclass
class RedisConfig:
    """Redis configuration for caching/sessions."""
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ttl: int = 3600


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    enabled: bool = True
    storage_path: str = "data/memory"
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7


@dataclass
class CacheConfig:
    """Semantic cache configuration."""
    enabled: bool = True
    storage_path: str = "data/cache"
    similarity_threshold: float = 0.95
    default_ttl: int = 3600
    max_entries: int = 10000


@dataclass
class VisionConfig:
    """Vision/multimodal configuration."""
    enabled: bool = True
    default_vision_model: str = "gpt-4-vision"
    default_gen_model: str = "dall-e-3"
    default_ocr_provider: str = "tesseract"


@dataclass
class WorkflowConfig:
    """Workflow engine configuration."""
    enabled: bool = True
    storage_path: str = "data/workflows"
    max_concurrent: int = 10
    default_timeout: int = 3600


@dataclass
class GuardrailsConfig:
    """Guardrails/safety configuration."""
    enabled: bool = True
    auto_redact_pii: bool = True
    block_unsafe: bool = True
    min_safety_score: float = 0.7


@dataclass
class VoiceConfig:
    """Voice interface configuration."""
    enabled: bool = False
    default_stt_provider: str = "openai"
    default_tts_provider: str = "openai"
    default_voice: str = "alloy"


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_enabled: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "logs/sentinelmesh.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_keys_enabled: bool = True
    api_keys_file: str = "data/api_keys.json"
    jwt_enabled: bool = False
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 86400  # 24 hours


@dataclass
class LLMProviderConfig:
    """LLM provider API keys."""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    cohere_api_key: str = ""
    huggingface_api_key: str = ""


class Config:
    """
    Centralized configuration management.
    Loads from environment variables with sensible defaults.
    """
    
    def __init__(self, env: str = "development"):
        self.env = env
        
        # Core configs
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.memory = MemoryConfig()
        self.cache = CacheConfig()
        self.vision = VisionConfig()
        self.workflow = WorkflowConfig()
        self.guardrails = GuardrailsConfig()
        self.voice = VoiceConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.llm_providers = LLMProviderConfig()
        
        # Load from environment
        self._load_from_env()
        
        # Validate
        self._validate()
        
        logger.info(f"Configuration loaded for environment: {env}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        
        # Database
        self.database.type = os.getenv("DB_TYPE", self.database.type)
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.name = os.getenv("DB_NAME", self.database.name)
        self.database.user = os.getenv("DB_USER", self.database.user)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        self.database.path = os.getenv("DB_PATH", self.database.path)
        
        # Redis
        self.redis.enabled = os.getenv("REDIS_ENABLED", "false").lower() == "true"
        self.redis.host = os.getenv("REDIS_HOST", self.redis.host)
        self.redis.port = int(os.getenv("REDIS_PORT", self.redis.port))
        self.redis.password = os.getenv("REDIS_PASSWORD", self.redis.password)
        
        # Memory
        self.memory.enabled = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
        self.memory.storage_path = os.getenv("MEMORY_STORAGE_PATH", self.memory.storage_path)
        
        # Cache
        self.cache.enabled = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache.similarity_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", self.cache.similarity_threshold))
        
        # Vision
        self.vision.enabled = os.getenv("ENABLE_VISION", "true").lower() == "true"
        
        # Workflows
        self.workflow.enabled = os.getenv("ENABLE_WORKFLOWS", "true").lower() == "true"
        
        # Guardrails
        self.guardrails.enabled = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"
        self.guardrails.auto_redact_pii = os.getenv("AUTO_REDACT_PII", "true").lower() == "true"
        self.guardrails.min_safety_score = float(os.getenv("MIN_SAFETY_SCORE", self.guardrails.min_safety_score))
        
        # Voice
        self.voice.enabled = os.getenv("ENABLE_VOICE", "false").lower() == "true"
        
        # API
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", self.api.port))
        self.api.workers = int(os.getenv("API_WORKERS", self.api.workers))
        self.api.reload = os.getenv("API_RELOAD", "false").lower() == "true"
        
        # Logging
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.file = os.getenv("LOG_FILE", self.logging.file)
        
        # Security
        self.security.api_keys_enabled = os.getenv("API_KEYS_ENABLED", "true").lower() == "true"
        self.security.jwt_secret = os.getenv("JWT_SECRET", self.security.jwt_secret)
        
        # LLM Providers
        self.llm_providers.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.llm_providers.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.llm_providers.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.llm_providers.cohere_api_key = os.getenv("COHERE_API_KEY", "")
    
    def _validate(self):
        """Validate configuration."""
        errors = []
        
        # Check required API keys (at least one)
        if not any([
            self.llm_providers.openai_api_key,
            self.llm_providers.anthropic_api_key,
            self.llm_providers.google_api_key,
            self.llm_providers.cohere_api_key
        ]):
            errors.append("At least one LLM provider API key must be configured")
        
        # Check JWT secret in production
        if self.env == "production" and self.security.jwt_enabled and not self.security.jwt_secret:
            errors.append("JWT_SECRET must be set in production")
        
        # Check database config for postgres/mysql
        if self.database.type in ["postgres", "mysql"]:
            if not self.database.password:
                errors.append(f"{self.database.type} requires DB_PASSWORD")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "env": self.env,
            "database": self.database.__dict__,
            "redis": self.redis.__dict__,
            "memory": self.memory.__dict__,
            "cache": self.cache.__dict__,
            "vision": self.vision.__dict__,
            "workflow": self.workflow.__dict__,
            "guardrails": self.guardrails.__dict__,
            "voice": self.voice.__dict__,
            "api": self.api.__dict__,
            "logging": self.logging.__dict__,
            "security": {**self.security.__dict__, "jwt_secret": "***"},  # Redact secret
            "llm_providers": {
                "openai_configured": bool(self.llm_providers.openai_api_key),
                "anthropic_configured": bool(self.llm_providers.anthropic_api_key),
                "google_configured": bool(self.llm_providers.google_api_key),
                "cohere_configured": bool(self.llm_providers.cohere_api_key),
            }
        }
    
    def save_to_file(self, filepath: str):
        """Save non-sensitive config to file."""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """Load config from file."""
        with open(filepath) as f:
            data = json.load(f)
        
        config = cls(env=data.get("env", "development"))
        # Note: File config is supplementary to env vars
        logger.info(f"Configuration loaded from {filepath}")
        return config


# Global config instance
_config: Optional[Config] = None


def get_config(env: Optional[str] = None) -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        env = env or os.getenv("SENTINELMESH_ENV", "development")
        _config = Config(env=env)
    return _config


def reload_config():
    """Reload configuration."""
    global _config
    _config = None
    return get_config()
