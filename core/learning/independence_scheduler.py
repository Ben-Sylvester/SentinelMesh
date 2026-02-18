"""
Independence Scheduler — Progressive Transition to Self-Sufficiency

Manages the gradual shift from 100% external LLM reliance to 90%+ self-model
usage through a 5-level maturity model with safe rollout strategies.
"""

import time
import logging
import random
from typing import Dict, Optional
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class MaturityLevel(IntEnum):
    """Capability maturity levels for independence."""
    BOOTSTRAP   = 0  # 100% external LLM
    LEARNING    = 1  #  20% self-model (simple tasks only)
    COMPETENT   = 2  #  50% self-model (common patterns)
    PROFICIENT  = 3  #  80% self-model (edge cases external)
    EXPERT      = 4  #  95% self-model (nearly independent)


@dataclass
class IndependenceMetrics:
    """Metrics tracking independence progress."""
    timestamp: float
    total_requests: int
    self_model_requests: int
    external_requests: int
    self_model_pct: float
    avg_confidence: float
    maturity_level: int


class IndependenceScheduler:
    """
    Controls when and how to route requests to self-models vs external LLMs
    based on capability maturity and confidence calibration.
    """
    
    # Maturity level thresholds
    MATURITY_THRESHOLDS = {
        MaturityLevel.BOOTSTRAP:  0.00,  # Just starting
        MaturityLevel.LEARNING:   0.20,  # 20% self-model
        MaturityLevel.COMPETENT:  0.50,  # 50% self-model
        MaturityLevel.PROFICIENT: 0.80,  # 80% self-model
        MaturityLevel.EXPERT:     0.95,  # 95% self-model
    }
    
    # Minimum confidence required to use self-model at each level
    CONFIDENCE_THRESHOLDS = {
        MaturityLevel.BOOTSTRAP:  1.00,  # Never use self-model
        MaturityLevel.LEARNING:   0.95,  # Very high confidence only
        MaturityLevel.COMPETENT:  0.85,  # High confidence
        MaturityLevel.PROFICIENT: 0.75,  # Medium-high confidence
        MaturityLevel.EXPERT:     0.60,  # Medium confidence
    }
    
    def __init__(
        self,
        db_path: str = "data/training_corpus.db",
        min_corpus_size_per_level: int = 1000,
    ):
        self.db_path = db_path
        self.min_corpus_size_per_level = min_corpus_size_per_level
        
        # Current state
        self.current_level = MaturityLevel.BOOTSTRAP
        self.total_requests = 0
        self.self_model_requests = 0
        self.external_requests = 0
        
        # Rolling window for confidence tracking
        self.recent_confidences = []
        self.confidence_window_size = 100
        
        self._load_state()
    
    def _load_state(self):
        """Load independence metrics from database."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        
        # Create metrics table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS independence_metrics (
                timestamp       TEXT PRIMARY KEY,
                total_requests  INTEGER NOT NULL,
                self_model_pct  REAL NOT NULL,
                external_pct    REAL NOT NULL,
                avg_confidence  REAL NOT NULL,
                maturity_level  INTEGER NOT NULL
            )
        """)
        conn.commit()
        
        # Load latest state
        row = conn.execute("""
            SELECT total_requests, self_model_pct, maturity_level
            FROM independence_metrics
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchone()
        
        if row:
            total, self_pct, level = row
            self.total_requests = total
            self.self_model_requests = int(total * self_pct)
            self.external_requests = total - self.self_model_requests
            self.current_level = MaturityLevel(level)
            logger.info(f"Loaded independence state: Level {level}, {self_pct:.1%} self-model")
        
        conn.close()
    
    def _save_metrics(self):
        """Persist current independence metrics."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        
        self_pct = (
            self.self_model_requests / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        ext_pct = 1.0 - self_pct
        avg_conf = (
            sum(self.recent_confidences) / len(self.recent_confidences)
            if self.recent_confidences else 0.0
        )
        
        conn.execute("""
            INSERT INTO independence_metrics 
            (timestamp, total_requests, self_model_pct, external_pct, avg_confidence, maturity_level)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            self.total_requests,
            self_pct,
            ext_pct,
            avg_conf,
            int(self.current_level),
        ))
        conn.commit()
        conn.close()
    
    def should_use_self_model(
        self,
        confidence: float,
        cluster_id: Optional[int] = None,
    ) -> bool:
        """
        Decide whether to route this request to self-model or external LLM.
        
        Args:
            confidence: Self-model's confidence score (0.0 - 1.0)
            cluster_id: Task cluster ID (for A/B testing per cluster)
        
        Returns:
            True if should use self-model, False for external LLM
        """
        # Always use external at bootstrap level
        if self.current_level == MaturityLevel.BOOTSTRAP:
            return False
        
        # Check confidence threshold for current maturity level
        required_confidence = self.CONFIDENCE_THRESHOLDS[self.current_level]
        if confidence < required_confidence:
            return False
        
        # Gradual rollout: use self-model with probability based on target %
        target_pct = self.MATURITY_THRESHOLDS[self.current_level]
        current_pct = (
            self.self_model_requests / self.total_requests
            if self.total_requests > 0 else 0.0
        )
        
        # If under target, route to self-model
        # If over target, route to external (maintain target %)
        if current_pct < target_pct:
            return True
        else:
            # Probabilistic: maintain target % with some noise
            return random.random() < target_pct
    
    def record_decision(self, used_self_model: bool, confidence: float):
        """
        Record a routing decision for metrics tracking.
        
        Args:
            used_self_model: Whether self-model was used
            confidence: Confidence score (if self-model was used)
        """
        self.total_requests += 1
        
        if used_self_model:
            self.self_model_requests += 1
            self.recent_confidences.append(confidence)
            if len(self.recent_confidences) > self.confidence_window_size:
                self.recent_confidences.pop(0)
        else:
            self.external_requests += 1
        
        # Persist every 100 requests
        if self.total_requests % 100 == 0:
            self._save_metrics()
    
    def evaluate_maturity_transition(
        self,
        corpus_size: int,
        avg_self_model_accuracy: float,
    ) -> bool:
        """
        Check if system is ready to advance to next maturity level.
        
        Args:
            corpus_size: Number of training examples collected
            avg_self_model_accuracy: Average accuracy of active self-models
        
        Returns:
            True if transitioned to new level, False otherwise
        """
        # Requirements for advancing:
        # 1. Sufficient training data
        # 2. High self-model accuracy
        # 3. Stable performance over time
        
        required_corpus = self.min_corpus_size_per_level * (int(self.current_level) + 1)
        required_accuracy = 0.85  # 85% accuracy vs external baseline
        
        if corpus_size < required_corpus:
            logger.debug(f"Corpus size {corpus_size} < required {required_corpus}")
            return False
        
        if avg_self_model_accuracy < required_accuracy:
            logger.debug(f"Accuracy {avg_self_model_accuracy:.2%} < required {required_accuracy:.2%}")
            return False
        
        # Ready to advance
        next_level = MaturityLevel(int(self.current_level) + 1)
        if next_level > MaturityLevel.EXPERT:
            return False  # Already at max level
        
        self.current_level = next_level
        logger.info(f"🎉 Advanced to maturity level {next_level.name} ({self.MATURITY_THRESHOLDS[next_level]:.0%} independence)")
        self._save_metrics()
        return True
    
    def get_current_level(self) -> MaturityLevel:
        """Return current maturity level."""
        return self.current_level
    
    def get_target_percentage(self) -> float:
        """Return target self-model usage % for current level."""
        return self.MATURITY_THRESHOLDS[self.current_level]
    
    def get_actual_percentage(self) -> float:
        """Return actual self-model usage %."""
        return (
            self.self_model_requests / self.total_requests
            if self.total_requests > 0 else 0.0
        )
    
    def estimate_cost_savings(self, external_cost_per_req: float = 0.002) -> Dict:
        """
        Estimate cost savings from using self-models.
        
        Args:
            external_cost_per_req: Average cost per external LLM request
        
        Returns:
            Dict with savings breakdown
        """
        local_cost_per_req = 0.0001  # ~$0.0001 for local inference (GPU amortized)
        
        total_cost = (
            self.external_requests * external_cost_per_req +
            self.self_model_requests * local_cost_per_req
        )
        
        baseline_cost = self.total_requests * external_cost_per_req
        savings = baseline_cost - total_cost
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "actual_cost_usd": total_cost,
            "baseline_cost_usd": baseline_cost,
            "savings_usd": savings,
            "savings_pct": savings_pct,
            "self_model_pct": self.get_actual_percentage(),
        }
    
    def stats(self) -> Dict:
        """Return independence statistics."""
        return {
            "maturity_level": int(self.current_level),
            "maturity_name": self.current_level.name,
            "target_self_model_pct": self.get_target_percentage(),
            "actual_self_model_pct": self.get_actual_percentage(),
            "total_requests": self.total_requests,
            "self_model_requests": self.self_model_requests,
            "external_requests": self.external_requests,
            "avg_confidence": (
                sum(self.recent_confidences) / len(self.recent_confidences)
                if self.recent_confidences else 0.0
            ),
            **self.estimate_cost_savings(),
        }
