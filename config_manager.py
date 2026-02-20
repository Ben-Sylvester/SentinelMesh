"""
Configuration Manager

Centralized configuration system with YAML support, validation,
environment variable override, and hot-reload capability.
"""

import os
import yaml
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CoreConfig:
    """Core system configuration."""
    environment: str = "development"  # development, staging, production
    debug: bool = False
    log_level: str = "INFO"
    port: int = 8000
    host: str = "0.0.0.0"
    workers: int = 4


@dataclass
class APIConfig:
    """API provider configuration."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    default_provider: str = "openai"
    default_model: str = "gpt-4"


@dataclass
class FeaturesConfig:
    """Feature flags configuration."""
    # Phase 1
    enable_memory: bool = True
    enable_streaming: bool = True
    enable_vision: bool = True
    
    # Phase 2
    enable_cache: bool = True
    enable_functions: bool = True
    enable_prompt_library: bool = True
    
    # Phase 3
    enable_workflows: bool = True
    enable_integrations: bool = True
    enable_guardrails: bool = True
    
    # Phase 4
    enable_plugins: bool = False
    enable_collaboration: bool = False
    enable_voice: bool = False


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    storage_path: str = "data/memory"
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.85
    max_context_length: int = 10
    enable_preference_learning: bool = True


@dataclass
class CacheConfig:
    """Cache system configuration."""
    storage_path: str = "data/cache"
    similarity_threshold: float = 0.95
    default_ttl: int = 3600  # 1 hour
    max_cache_size_mb: int = 1000
    cleanup_interval: int = 3600  # 1 hour


@dataclass
class GuardrailsConfig:
    """Guardrails configuration."""
    auto_redact_pii: bool = True
    block_unsafe_content: bool = True
    min_safety_score: float = 0.7
    enable_jailbreak_detection: bool = True
    log_violations: bool = True


@dataclass
class VisionConfig:
    """Vision system configuration."""
    default_provider: str = "openai"  # openai, anthropic, google
    default_quality: str = "balanced"  # cheap, balanced, accurate
    image_generation_model: str = "dall-e-3"
    ocr_provider: str = "tesseract"


@dataclass
class VoiceConfig:
    """Voice interface configuration."""
    default_stt_provider: str = "openai"
    default_tts_provider: str = "openai"
    default_voice: str = "alloy"
    speech_rate: float = 1.0


@dataclass
class WorkflowConfig:
    """Workflow engine configuration."""
    storage_path: str = "data/workflows"
    max_concurrent_workflows: int = 10
    default_timeout: int = 3600
    enable_scheduling: bool = True


@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""
    max_concurrent_requests: int = 100
    request_timeout: int = 300
    rate_limit_per_minute: int = 60
    enable_compression: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_key_required: bool = True
    allowed_origins: list = field(default_factory=lambda: ["*"])
    enable_cors: bool = True
    max_upload_size_mb: int = 100
    rate_limiting: bool = True


class ConfigManager:
    """
    Centralized configuration manager.
    Supports YAML files, environment variables, and hot-reload.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "config.yaml")
        self.config: Dict[str, Any] = {}
        
        # Configuration objects
        self.core = CoreConfig()
        self.api = APIConfig()
        self.features = FeaturesConfig()
        self.memory = MemoryConfig()
        self.cache = CacheConfig()
        self.guardrails = GuardrailsConfig()
        self.vision = VisionConfig()
        self.voice = VoiceConfig()
        self.workflow = WorkflowConfig()
        self.performance = PerformanceConfig()
        self.security = SecurityConfig()
        
        # Load configuration
        self.load()
    
    def load(self):
        """Load configuration from file and environment."""
        # Load from YAML file if exists
        if self.config_path.exists():
            self._load_from_file()
        else:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate()
        
        logger.info("Configuration loaded successfully")
    
    def _load_from_file(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            
            # Update configuration objects
            self._update_config(file_config)
            
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
    
    def _load_from_env(self):
        """Override configuration with environment variables."""
        # Core
        if os.getenv("ENVIRONMENT"):
            self.core.environment = os.getenv("ENVIRONMENT")
        if os.getenv("DEBUG"):
            self.core.debug = os.getenv("DEBUG").lower() == "true"
        if os.getenv("LOG_LEVEL"):
            self.core.log_level = os.getenv("LOG_LEVEL")
        if os.getenv("PORT"):
            self.core.port = int(os.getenv("PORT"))
        
        # API Keys
        if os.getenv("OPENAI_API_KEY"):
            self.api.openai_api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY"):
            self.api.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if os.getenv("GOOGLE_API_KEY"):
            self.api.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Features
        if os.getenv("ENABLE_MEMORY"):
            self.features.enable_memory = os.getenv("ENABLE_MEMORY").lower() == "true"
        if os.getenv("ENABLE_CACHE"):
            self.features.enable_cache = os.getenv("ENABLE_CACHE").lower() == "true"
        if os.getenv("ENABLE_WORKFLOWS"):
            self.features.enable_workflows = os.getenv("ENABLE_WORKFLOWS").lower() == "true"
        
        # Guardrails
        if os.getenv("AUTO_REDACT_PII"):
            self.guardrails.auto_redact_pii = os.getenv("AUTO_REDACT_PII").lower() == "true"
        if os.getenv("MIN_SAFETY_SCORE"):
            self.guardrails.min_safety_score = float(os.getenv("MIN_SAFETY_SCORE"))
    
    def _update_config(self, file_config: Dict):
        """Update configuration objects from file."""
        # Core
        if "core" in file_config:
            for key, value in file_config["core"].items():
                if hasattr(self.core, key):
                    setattr(self.core, key, value)
        
        # API
        if "api" in file_config:
            for key, value in file_config["api"].items():
                if hasattr(self.api, key):
                    setattr(self.api, key, value)
        
        # Features
        if "features" in file_config:
            for key, value in file_config["features"].items():
                if hasattr(self.features, key):
                    setattr(self.features, key, value)
        
        # Memory
        if "memory" in file_config:
            for key, value in file_config["memory"].items():
                if hasattr(self.memory, key):
                    setattr(self.memory, key, value)
        
        # Cache
        if "cache" in file_config:
            for key, value in file_config["cache"].items():
                if hasattr(self.cache, key):
                    setattr(self.cache, key, value)
        
        # Guardrails
        if "guardrails" in file_config:
            for key, value in file_config["guardrails"].items():
                if hasattr(self.guardrails, key):
                    setattr(self.guardrails, key, value)
        
        # Vision
        if "vision" in file_config:
            for key, value in file_config["vision"].items():
                if hasattr(self.vision, key):
                    setattr(self.vision, key, value)
        
        # Voice
        if "voice" in file_config:
            for key, value in file_config["voice"].items():
                if hasattr(self.voice, key):
                    setattr(self.voice, key, value)
        
        # Workflow
        if "workflow" in file_config:
            for key, value in file_config["workflow"].items():
                if hasattr(self.workflow, key):
                    setattr(self.workflow, key, value)
        
        # Performance
        if "performance" in file_config:
            for key, value in file_config["performance"].items():
                if hasattr(self.performance, key):
                    setattr(self.performance, key, value)
        
        # Security
        if "security" in file_config:
            for key, value in file_config["security"].items():
                if hasattr(self.security, key):
                    setattr(self.security, key, value)
    
    def _validate(self):
        """Validate configuration."""
        errors = []
        
        # Check required API keys if features enabled
        if self.features.enable_vision and not self.api.openai_api_key:
            logger.warning("Vision enabled but no OpenAI API key configured")
        
        if self.features.enable_cache and self.cache.similarity_threshold > 1.0:
            errors.append("Cache similarity_threshold must be <= 1.0")
        
        if self.guardrails.min_safety_score < 0 or self.guardrails.min_safety_score > 1:
            errors.append("Guardrails min_safety_score must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
    
    def reload(self):
        """Hot-reload configuration."""
        logger.info("Reloading configuration...")
        self.load()
    
    def save(self, path: Optional[str] = None):
        """Save current configuration to YAML file."""
        save_path = Path(path or self.config_path)
        
        config_dict = {
            "core": self.core.__dict__,
            "api": self.api.__dict__,
            "features": self.features.__dict__,
            "memory": self.memory.__dict__,
            "cache": self.cache.__dict__,
            "guardrails": self.guardrails.__dict__,
            "vision": self.vision.__dict__,
            "voice": self.voice.__dict__,
            "workflow": self.workflow.__dict__,
            "performance": self.performance.__dict__,
            "security": self.security.__dict__
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key."""
        parts = key.split('.')
        value = self
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        
        return value
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        return {
            "core": self.core.__dict__,
            "api": {k: ("***" if "key" in k.lower() and v else v) 
                   for k, v in self.api.__dict__.items()},  # Hide API keys
            "features": self.features.__dict__,
            "memory": self.memory.__dict__,
            "cache": self.cache.__dict__,
            "guardrails": self.guardrails.__dict__,
            "vision": self.vision.__dict__,
            "voice": self.voice.__dict__,
            "workflow": self.workflow.__dict__,
            "performance": self.performance.__dict__,
            "security": self.security.__dict__
        }


# Global configuration instance
config = ConfigManager()
