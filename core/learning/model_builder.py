"""
Self-Model Builder — Trains Local Models from Distilled Knowledge

Converts collected training data into executable inference models that
replace external LLM calls for learned task clusters.
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a self-trained model."""
    name: str
    architecture: str  # "seq2seq", "classification", "embedding"
    hidden_size: int
    num_layers: int
    vocab_size: int
    max_length: int
    task_clusters: List[int]


@dataclass
class SelfModel:
    """Metadata for a trained self-model."""
    id: int
    name: str
    config: ModelConfig
    weights_path: Path
    accuracy: float
    size_mb: float
    avg_latency_ms: int
    created_at: float
    active: bool


class ModelBuilder:
    """
    Trains lightweight models on clustered task data.
    Uses distillation to learn from external LLM outputs.
    """
    
    def __init__(
        self,
        models_dir: str = "models/checkpoints",
        db_path: str = "data/training_corpus.db",
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.registry_path = self.models_dir / "registry.json"
        
        self.models: Dict[str, SelfModel] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load metadata for all trained models."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
                for entry in data["models"]:
                    config = ModelConfig(**entry["config"])
                    model = SelfModel(
                        id=entry["id"],
                        name=entry["name"],
                        config=config,
                        weights_path=Path(entry["weights_path"]),
                        accuracy=entry["accuracy"],
                        size_mb=entry["size_mb"],
                        avg_latency_ms=entry["avg_latency_ms"],
                        created_at=entry["created_at"],
                        active=entry["active"],
                    )
                    self.models[model.name] = model
            logger.info(f"Loaded {len(self.models)} self-models from registry")
    
    def _save_registry(self):
        """Save model registry to disk."""
        data = {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "config": {
                        "name": m.config.name,
                        "architecture": m.config.architecture,
                        "hidden_size": m.config.hidden_size,
                        "num_layers": m.config.num_layers,
                        "vocab_size": m.config.vocab_size,
                        "max_length": m.config.max_length,
                        "task_clusters": m.config.task_clusters,
                    },
                    "weights_path": str(m.weights_path),
                    "accuracy": m.accuracy,
                    "size_mb": m.size_mb,
                    "avg_latency_ms": m.avg_latency_ms,
                    "created_at": m.created_at,
                    "active": m.active,
                }
                for m in self.models.values()
            ]
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def train_cluster_model(
        self,
        cluster_id: int,
        training_examples: List,
        architecture: str = "seq2seq",
        hidden_size: int = 256,
        num_layers: int = 4,
        epochs: int = 3,
    ) -> Optional[SelfModel]:
        """
        Train a model on examples from a specific task cluster.
        
        Args:
            cluster_id: Task cluster to train on
            training_examples: List of TrainingExample objects
            architecture: Model architecture type
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            epochs: Training epochs
        
        Returns:
            SelfModel metadata if training successful, None otherwise
        """
        if len(training_examples) < 100:
            logger.warning(f"Cluster {cluster_id} has <100 examples, skipping training")
            return None
        
        logger.info(f"Training model for cluster {cluster_id} with {len(training_examples)} examples")
        
        # Model name
        model_name = f"cluster_{cluster_id}_v{int(time.time())}"
        weights_path = self.models_dir / f"{model_name}.pt"
        
        # Build config
        config = ModelConfig(
            name=model_name,
            architecture=architecture,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=32000,  # BPE tokenizer
            max_length=512,
            task_clusters=[cluster_id],
        )
        
        # Train (placeholder - real training would use PyTorch/transformers)
        accuracy, latency_ms = self._train_model_impl(
            training_examples, config, epochs, weights_path
        )
        
        # Measure model size
        size_mb = 0.0
        if weights_path.exists():
            size_mb = weights_path.stat().st_size / (1024 * 1024)
        
        # Create model metadata
        model = SelfModel(
            id=len(self.models),
            name=model_name,
            config=config,
            weights_path=weights_path,
            accuracy=accuracy,
            size_mb=size_mb,
            avg_latency_ms=latency_ms,
            created_at=time.time(),
            active=True,
        )
        
        self.models[model_name] = model
        self._save_registry()
        
        logger.info(f"✓ Trained {model_name}: {accuracy:.2%} accuracy, {size_mb:.1f}MB, {latency_ms}ms")
        return model
    
    def _train_model_impl(
        self,
        examples: List,
        config: ModelConfig,
        epochs: int,
        save_path: Path,
    ) -> Tuple[float, int]:
        """
        Actual model training implementation.
        
        This is a placeholder. Real implementation would:
        1. Initialize transformer model (distilbert, t5-small, etc)
        2. Create DataLoader from examples
        3. Fine-tune with teacher forcing from external LLM outputs
        4. Evaluate on held-out validation set
        5. Save best checkpoint
        
        For production, use:
        - Hugging Face Transformers for model architecture
        - PyTorch for training loop
        - DeepSpeed/FSDP for large model training
        - ONNX Runtime for optimized inference
        """
        try:
            # Simulate training
            logger.info(f"  [Training] {len(examples)} examples, {epochs} epochs...")
            time.sleep(0.5)  # Simulate training time
            
            # Mock results
            accuracy = np.random.uniform(0.80, 0.95)
            latency_ms = int(np.random.uniform(50, 150))
            
            # Save mock checkpoint
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(f"Mock checkpoint for {config.name}\n")
            
            return accuracy, latency_ms
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 0.0, 0
    
    def get_model_for_cluster(self, cluster_id: int) -> Optional[SelfModel]:
        """Find the best active model that can handle this cluster."""
        candidates = [
            m for m in self.models.values()
            if m.active and cluster_id in m.config.task_clusters
        ]
        if not candidates:
            return None
        # Return model with highest accuracy
        return max(candidates, key=lambda m: m.accuracy)
    
    def evaluate_model(self, model: SelfModel, test_examples: List) -> float:
        """
        Evaluate model accuracy on held-out test examples.
        
        Returns:
            Accuracy score (0.0 - 1.0)
        """
        if not test_examples:
            return 0.0
        
        # Placeholder - real implementation would:
        # 1. Load model weights
        # 2. Run inference on test inputs
        # 3. Compare outputs to ground truth (external LLM outputs)
        # 4. Compute metrics: exact match, BLEU, embedding similarity
        
        logger.info(f"Evaluating {model.name} on {len(test_examples)} test examples")
        
        # Mock evaluation
        return model.accuracy
    
    def deactivate_model(self, model_name: str):
        """Mark a model as inactive (no longer used for inference)."""
        if model_name in self.models:
            self.models[model_name].active = False
            self._save_registry()
            logger.info(f"Deactivated model: {model_name}")
    
    def prune_old_models(self, keep_top_n: int = 5):
        """Keep only the N best models, deactivate rest."""
        active = [m for m in self.models.values() if m.active]
        if len(active) <= keep_top_n:
            return
        
        # Sort by accuracy
        sorted_models = sorted(active, key=lambda m: m.accuracy, reverse=True)
        
        for model in sorted_models[keep_top_n:]:
            self.deactivate_model(model.name)
    
    def stats(self) -> Dict:
        """Return builder statistics."""
        active_models = [m for m in self.models.values() if m.active]
        return {
            "total_models": len(self.models),
            "active_models": len(active_models),
            "avg_accuracy": np.mean([m.accuracy for m in active_models]) if active_models else 0.0,
            "total_size_mb": sum(m.size_mb for m in active_models),
            "avg_latency_ms": int(np.mean([m.avg_latency_ms for m in active_models])) if active_models else 0,
        }
