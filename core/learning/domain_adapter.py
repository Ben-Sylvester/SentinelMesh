"""
Domain Adaptation Engine — Plug-and-Play Industry Learning

Automatically detects deployment industry and adapts learning/inference
to domain-specific vocabulary, compliance requirements, and quality metrics.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Callable
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class IndustryProfile:
    """Profile for a specific industry vertical."""
    name: str
    keywords: Set[str]
    compliance_rules: List[str]
    custom_reward_fn: Optional[Callable] = None
    safety_multiplier: float = 1.0


class DomainAdapter:
    """
    Detects deployment industry from request patterns and adapts
    the learning system accordingly.
    """
    
    # Industry knowledge base (expandable)
    INDUSTRIES = {
        "healthcare": IndustryProfile(
            name="healthcare",
            keywords={
                "patient", "diagnosis", "medical", "doctor", "hospital", "treatment",
                "symptoms", "disease", "medication", "prescription", "clinical",
                "HIPAA", "PHI", "EHR", "ICD", "CPT", "radiology", "pathology"
            },
            compliance_rules=["PHI_redaction", "audit_logging", "consent_tracking"],
            safety_multiplier=1.5,  # Extra penalty for unsafe medical advice
        ),
        
        "finance": IndustryProfile(
            name="finance",
            keywords={
                "portfolio", "investment", "trading", "stock", "bond", "risk",
                "return", "asset", "liability", "SEC", "FINRA", "compliance",
                "audit", "accounting", "GAAP", "balance sheet", "cash flow",
                "derivatives", "hedge", "equity", "debt", "capital"
            },
            compliance_rules=["PII_masking", "audit_trail", "transaction_logging"],
            safety_multiplier=1.3,  # Penalty for hallucinated financial data
        ),
        
        "legal": IndustryProfile(
            name="legal",
            keywords={
                "contract", "law", "statute", "regulation", "case law", "litigation",
                "attorney", "counsel", "court", "judge", "plaintiff", "defendant",
                "discovery", "deposition", "trial", "appeal", "precedent",
                "jurisdiction", "tort", "liability", "damages", "injunction"
            },
            compliance_rules=["citation_verification", "precedent_validation"],
            safety_multiplier=1.4,  # High stakes for incorrect legal advice
        ),
        
        "ecommerce": IndustryProfile(
            name="ecommerce",
            keywords={
                "product", "purchase", "checkout", "cart", "shipping", "delivery",
                "inventory", "SKU", "price", "discount", "coupon", "refund",
                "return", "warranty", "review", "rating", "recommendation",
                "category", "brand", "stock", "availability"
            },
            compliance_rules=["PCI_DSS", "GDPR_compliance"],
            safety_multiplier=1.0,
        ),
        
        "software": IndustryProfile(
            name="software",
            keywords={
                "code", "function", "class", "method", "variable", "API", "bug",
                "debug", "compile", "runtime", "syntax", "error", "exception",
                "algorithm", "data structure", "git", "repository", "deploy",
                "CI/CD", "testing", "unit test", "integration", "refactor"
            },
            compliance_rules=["code_safety", "dependency_audit"],
            safety_multiplier=1.1,  # Moderate penalty for buggy code
        ),
        
        "education": IndustryProfile(
            name="education",
            keywords={
                "student", "teacher", "course", "lesson", "curriculum", "exam",
                "grade", "homework", "assignment", "learning", "study", "tutorial",
                "lecture", "classroom", "syllabus", "pedagogy", "assessment",
                "FERPA", "academic", "university", "college", "school"
            },
            compliance_rules=["FERPA_compliance", "age_appropriate_content"],
            safety_multiplier=1.2,
        ),
        
        "government": IndustryProfile(
            name="government",
            keywords={
                "policy", "regulation", "legislation", "government", "agency",
                "federal", "state", "local", "public", "citizen", "tax",
                "benefits", "services", "permit", "license", "compliance",
                "FedRAMP", "FISMA", "authority", "jurisdiction", "statute"
            },
            compliance_rules=["FedRAMP", "FISMA", "audit_trail"],
            safety_multiplier=1.3,
        },
    }
    
    def __init__(self, history_window: int = 1000):
        self.history_window = history_window
        self.request_history: List[str] = []
        self.detected_industry: Optional[str] = None
        self.detection_confidence: float = 0.0
        
        # Domain-specific vocabulary learned from requests
        self.domain_vocabulary: Set[str] = set()
        self.vocabulary_frequency: Counter = Counter()
    
    def ingest_request(self, text: str):
        """Add a request to the history for industry detection."""
        self.request_history.append(text.lower())
        if len(self.request_history) > self.history_window:
            self.request_history.pop(0)
        
        # Update vocabulary
        words = re.findall(r'\b\w+\b', text.lower())
        self.domain_vocabulary.update(words)
        self.vocabulary_frequency.update(words)
    
    def detect_industry(self, min_confidence: float = 0.3) -> Optional[str]:
        """
        Detect the primary industry from accumulated request history.
        
        Args:
            min_confidence: Minimum confidence threshold to make a detection
        
        Returns:
            Industry name if confident, None otherwise
        """
        if len(self.request_history) < 100:
            return None  # Need more data
        
        # Score each industry
        scores = {}
        for industry, profile in self.INDUSTRIES.items():
            score = 0
            for req in self.request_history:
                # Count keyword matches
                matches = sum(1 for kw in profile.keywords if kw in req)
                score += matches
            scores[industry] = score
        
        # Normalize scores
        total = sum(scores.values())
        if total == 0:
            return None
        
        normalized = {k: v / total for k, v in scores.items()}
        
        # Pick highest scoring industry
        best_industry = max(normalized, key=normalized.get)
        confidence = normalized[best_industry]
        
        if confidence >= min_confidence:
            self.detected_industry = best_industry
            self.detection_confidence = confidence
            logger.info(f"Industry detected: {best_industry} ({confidence:.1%} confidence)")
            return best_industry
        
        return None
    
    def get_industry_profile(self) -> Optional[IndustryProfile]:
        """Get the profile for the detected industry."""
        if self.detected_industry:
            return self.INDUSTRIES[self.detected_industry]
        return None
    
    def compute_domain_reward(self, base_reward: float, result) -> float:
        """
        Adjust reward based on industry-specific criteria.
        
        Args:
            base_reward: Standard reward from compute_reward()
            result: StrategyResult object
        
        Returns:
            Adjusted reward incorporating domain safety/quality
        """
        profile = self.get_industry_profile()
        if not profile:
            return base_reward
        
        # Apply industry-specific safety multiplier
        # (Higher multiplier = greater penalty for errors)
        adjusted = base_reward / profile.safety_multiplier
        
        # Apply custom reward function if defined
        if profile.custom_reward_fn:
            adjusted = profile.custom_reward_fn(result, adjusted)
        
        return adjusted
    
    def get_compliance_requirements(self) -> List[str]:
        """Return compliance rules for the detected industry."""
        profile = self.get_industry_profile()
        return profile.compliance_rules if profile else []
    
    def extract_domain_terms(self, top_n: int = 100) -> List[str]:
        """
        Extract the most frequent domain-specific terms.
        
        Returns:
            List of (term, frequency) tuples
        """
        # Filter out common English words
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "up", "about", "into", "through",
            "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "should", "could",
            "can", "may", "might", "must", "this", "that", "these", "those"
        }
        
        filtered = {
            word: count
            for word, count in self.vocabulary_frequency.items()
            if word not in stopwords and len(word) > 3
        }
        
        return [
            word for word, count 
            in sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]
        ]
    
    def suggest_vocabulary_expansion(self) -> Dict[str, List[str]]:
        """
        Suggest new keywords to add to industry profiles based on
        observed request patterns.
        """
        domain_terms = self.extract_domain_terms(top_n=50)
        profile = self.get_industry_profile()
        
        if not profile:
            return {}
        
        # Find terms not yet in the industry profile
        new_terms = [
            term for term in domain_terms
            if term not in profile.keywords
        ]
        
        return {
            "industry": profile.name,
            "suggested_keywords": new_terms[:20],
            "confidence": self.detection_confidence,
        }
    
    def stats(self) -> Dict:
        """Return domain adaptation statistics."""
        return {
            "detected_industry": self.detected_industry,
            "detection_confidence": self.detection_confidence,
            "request_history_size": len(self.request_history),
            "domain_vocabulary_size": len(self.domain_vocabulary),
            "compliance_rules": self.get_compliance_requirements(),
        }
