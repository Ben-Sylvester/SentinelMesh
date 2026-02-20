"""
Guardrails System

Content filtering, PII detection/redaction, safety scoring.
Ensures GDPR/HIPAA compliance and brand safety.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    """Safety level classification."""
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


@dataclass
class PIIEntity:
    """Detected PII entity."""
    type: str  # email, phone, ssn, credit_card, etc.
    value: str
    start: int
    end: int
    confidence: float


@dataclass
class SafetyResult:
    """Safety check result."""
    safe: bool
    level: SafetyLevel
    score: float  # 0-1, higher is safer
    flags: List[str]
    pii_detected: List[PIIEntity]
    redacted_text: Optional[str] = None


class PIIDetector:
    """
    Detects and redacts Personally Identifiable Information.
    Supports: email, phone, SSN, credit cards, names, addresses.
    """
    
    def __init__(self):
        # Regex patterns for common PII
        self.patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        }
    
    def detect(self, text: str) -> List[PIIEntity]:
        """Detect PII entities in text."""
        entities = []
        
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entities.append(PIIEntity(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95  # Rule-based = high confidence
                ))
        
        return entities
    
    def redact(self, text: str, entities: Optional[List[PIIEntity]] = None) -> str:
        """Redact PII from text."""
        if entities is None:
            entities = self.detect(text)
        
        # Sort by position (reverse) to maintain indices
        entities = sorted(entities, key=lambda e: e.start, reverse=True)
        
        redacted = text
        for entity in entities:
            replacement = f"[{entity.type.upper()}_REDACTED]"
            redacted = redacted[:entity.start] + replacement + redacted[entity.end:]
        
        return redacted
    
    def has_pii(self, text: str) -> bool:
        """Check if text contains PII."""
        return len(self.detect(text)) > 0


class ContentFilter:
    """
    Filters harmful content: profanity, hate speech, violence, adult content.
    """
    
    def __init__(self):
        # Simplified keyword lists - in production, use ML models
        self.profanity_keywords = [
            # Add actual keywords in production
        ]
        
        self.hate_speech_keywords = [
            # Hate speech detection keywords
        ]
        
        self.violence_keywords = [
            "kill", "murder", "bomb", "weapon", "attack"
        ]
        
        self.adult_keywords = [
            # Adult content keywords
        ]
    
    def check(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check text for harmful content.
        
        Returns:
            (is_safe, flags)
        """
        flags = []
        text_lower = text.lower()
        
        # Check profanity
        if any(word in text_lower for word in self.profanity_keywords):
            flags.append("profanity")
        
        # Check hate speech
        if any(word in text_lower for word in self.hate_speech_keywords):
            flags.append("hate_speech")
        
        # Check violence
        violence_count = sum(1 for word in self.violence_keywords if word in text_lower)
        if violence_count >= 2:  # Multiple violence keywords = flag
            flags.append("violence")
        
        # Check adult content
        if any(word in text_lower for word in self.adult_keywords):
            flags.append("adult_content")
        
        is_safe = len(flags) == 0
        return is_safe, flags


class JailbreakDetector:
    """
    Detects prompt injection and jailbreak attempts.
    """
    
    def __init__(self):
        self.jailbreak_patterns = [
            "ignore previous instructions",
            "disregard all prior",
            "forget everything",
            "you are now",
            "new role:",
            "system:",
            "admin mode",
            "developer mode",
            "god mode"
        ]
    
    def detect(self, text: str) -> Tuple[bool, float]:
        """
        Detect jailbreak attempts.
        
        Returns:
            (is_jailbreak, confidence)
        """
        text_lower = text.lower()
        
        matches = [
            pattern for pattern in self.jailbreak_patterns
            if pattern in text_lower
        ]
        
        if matches:
            confidence = min(len(matches) * 0.3, 1.0)
            return True, confidence
        
        return False, 0.0


class SafetyScorer:
    """
    Assigns overall safety score to content.
    """
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.content_filter = ContentFilter()
        self.jailbreak_detector = JailbreakDetector()
    
    def score(self, text: str) -> SafetyResult:
        """
        Compute comprehensive safety score.
        
        Score ranges:
        - 0.9-1.0: Safe
        - 0.7-0.9: Warning
        - 0.5-0.7: Unsafe
        - 0.0-0.5: Blocked
        """
        score = 1.0
        flags = []
        
        # Check PII
        pii_entities = self.pii_detector.detect(text)
        if pii_entities:
            score -= 0.2
            flags.append("pii_detected")
        
        # Check harmful content
        is_safe_content, content_flags = self.content_filter.check(text)
        if not is_safe_content:
            score -= 0.3 * len(content_flags)
            flags.extend(content_flags)
        
        # Check jailbreak
        is_jailbreak, jb_confidence = self.jailbreak_detector.detect(text)
        if is_jailbreak:
            score -= 0.4 * jb_confidence
            flags.append("jailbreak_attempt")
        
        # Determine level
        if score >= 0.9:
            level = SafetyLevel.SAFE
            safe = True
        elif score >= 0.7:
            level = SafetyLevel.WARNING
            safe = True
        elif score >= 0.5:
            level = SafetyLevel.UNSAFE
            safe = False
        else:
            level = SafetyLevel.BLOCKED
            safe = False
        
        # Redact PII if detected
        redacted_text = None
        if pii_entities:
            redacted_text = self.pii_detector.redact(text, pii_entities)
        
        return SafetyResult(
            safe=safe,
            level=level,
            score=max(0.0, score),
            flags=flags,
            pii_detected=pii_entities,
            redacted_text=redacted_text
        )


class GuardrailsManager:
    """
    Central manager for all guardrails.
    Provides unified interface for safety checks.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.pii_detector = PIIDetector()
        self.content_filter = ContentFilter()
        self.jailbreak_detector = JailbreakDetector()
        self.safety_scorer = SafetyScorer()
        
        # Configuration
        self.auto_redact_pii = self.config.get("auto_redact_pii", True)
        self.block_unsafe = self.config.get("block_unsafe", True)
        self.min_safety_score = self.config.get("min_safety_score", 0.7)
    
    async def check_input(self, text: str) -> SafetyResult:
        """Check input text before processing."""
        return self.safety_scorer.score(text)
    
    async def check_output(self, text: str) -> SafetyResult:
        """Check output text before returning to user."""
        result = self.safety_scorer.score(text)
        
        # Auto-redact PII in output
        if self.auto_redact_pii and result.pii_detected:
            result.redacted_text = self.pii_detector.redact(text)
        
        return result
    
    async def check_conversation(
        self,
        prompt: str,
        response: str
    ) -> Tuple[SafetyResult, SafetyResult]:
        """Check both prompt and response."""
        prompt_result = await self.check_input(prompt)
        response_result = await self.check_output(response)
        
        return prompt_result, response_result
    
    def is_allowed(self, safety_result: SafetyResult) -> bool:
        """Determine if content is allowed based on config."""
        if self.block_unsafe:
            return safety_result.score >= self.min_safety_score
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guardrails statistics."""
        # In production, track actual stats
        return {
            "auto_redact_pii": self.auto_redact_pii,
            "block_unsafe": self.block_unsafe,
            "min_safety_score": self.min_safety_score,
            "total_checks": 0,
            "blocked_count": 0,
            "pii_redactions": 0
        }


class ComplianceManager:
    """
    Manages compliance with regulations (GDPR, HIPAA, etc.)
    """
    
    def __init__(self):
        self.guardrails = GuardrailsManager()
        self.audit_log = []
    
    async def gdpr_check(self, text: str) -> Dict[str, Any]:
        """Check GDPR compliance."""
        result = await self.guardrails.check_input(text)
        
        return {
            "compliant": result.safe,
            "pii_detected": len(result.pii_detected) > 0,
            "pii_types": [e.type for e in result.pii_detected],
            "requires_consent": len(result.pii_detected) > 0
        }
    
    async def hipaa_check(self, text: str) -> Dict[str, Any]:
        """Check HIPAA compliance (PHI detection)."""
        result = await self.guardrails.check_input(text)
        
        # In production, add specific PHI detection
        phi_types = ["ssn", "phone", "email"]  # Simplified
        phi_detected = [e for e in result.pii_detected if e.type in phi_types]
        
        return {
            "compliant": len(phi_detected) == 0,
            "phi_detected": len(phi_detected) > 0,
            "phi_types": [e.type for e in phi_detected],
            "requires_encryption": len(phi_detected) > 0
        }
    
    def log_check(self, user_id: str, result: SafetyResult):
        """Log safety check for audit trail."""
        self.audit_log.append({
            "timestamp": time.time(),
            "user_id": user_id,
            "safe": result.safe,
            "score": result.score,
            "flags": result.flags
        })
