"""
Model Evolution System — Continuous Model Improvement

Applies evolutionary algorithms to improve self-models over time through
pruning, quantization, knowledge merging, and generational selection.
"""

import logging
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelGeneration:
    """Represents one generation in the evolutionary lineage."""
    generation: int
    parent_id: Optional[int]
    mutation_type: str
    accuracy: float
    size_mb: float
    latency_ms: int
    fitness_score: float
    created_at: float


class EvolutionEngine:
    """
    Evolves self-models through mutation operators to optimize the
    accuracy/speed/size tradeoff.
    """
    
    MUTATION_OPERATORS = [
        "prune_weights",     # Remove low-importance neurons
        "quantize_int8",     # Reduce precision to INT8
        "knowledge_merge",   # Merge multiple specialist models
        "add_adapter",       # Add task-specific adapter layers
        "distill_smaller",   # Distill to smaller architecture
    ]
    
    def __init__(
        self,
        checkpoints_dir: str = "models/checkpoints",
        generations_dir: str = "models/generations",
        max_generations: int = 10,
    ):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.generations_dir = Path(generations_dir)
        self.max_generations = max_generations
        
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        
        # Evolution history
        self.lineage: List[ModelGeneration] = []
        self.current_generation = 0
    
    def compute_fitness(
        self,
        accuracy: float,
        latency_ms: int,
        size_mb: float,
        accuracy_weight: float = 0.6,
        speed_weight: float = 0.3,
        size_weight: float = 0.1,
    ) -> float:
        """
        Compute fitness score for multi-objective optimization.
        
        Args:
            accuracy: Model accuracy (0.0 - 1.0)
            latency_ms: Inference latency in milliseconds
            size_mb: Model size in megabytes
            accuracy_weight: Weight for accuracy objective
            speed_weight: Weight for speed objective
            size_weight: Weight for size objective
        
        Returns:
            Fitness score (higher is better)
        """
        # Normalize metrics to 0-1 range
        accuracy_norm = accuracy  # Already 0-1
        speed_norm = max(0, 1 - (latency_ms / 1000))  # Faster = higher score
        size_norm = max(0, 1 - (size_mb / 1000))      # Smaller = higher score
        
        # Weighted sum
        fitness = (
            accuracy_weight * accuracy_norm +
            speed_weight * speed_norm +
            size_weight * size_norm
        )
        
        return fitness
    
    def prune_weights(self, model_path: Path, target_sparsity: float = 0.3) -> Path:
        """
        Prune low-magnitude weights from model.
        
        Args:
            model_path: Path to model checkpoint
            target_sparsity: Fraction of weights to remove (0.3 = 30%)
        
        Returns:
            Path to pruned model
        """
        logger.info(f"Pruning model {model_path.name} (target sparsity: {target_sparsity:.0%})")
        
        # In production, use:
        # - PyTorch: torch.nn.utils.prune
        # - TensorFlow: tensorflow_model_optimization
        
        pruned_path = self.generations_dir / f"{model_path.stem}_pruned.pt"
        
        # Mock pruning (real implementation would load weights, apply mask, save)
        shutil.copy(model_path, pruned_path)
        
        logger.info(f"✓ Pruned model saved to {pruned_path}")
        return pruned_path
    
    def quantize_int8(self, model_path: Path) -> Path:
        """
        Quantize model from FP32 to INT8 precision.
        
        Args:
            model_path: Path to model checkpoint
        
        Returns:
            Path to quantized model
        """
        logger.info(f"Quantizing model {model_path.name} to INT8")
        
        # In production, use:
        # - PyTorch: torch.quantization.quantize_dynamic
        # - ONNX Runtime: quantization API
        # - TensorFlow: TFLite quantization
        
        quantized_path = self.generations_dir / f"{model_path.stem}_int8.pt"
        
        # Mock quantization
        shutil.copy(model_path, quantized_path)
        
        logger.info(f"✓ Quantized model saved to {quantized_path}")
        return quantized_path
    
    def knowledge_merge(self, model_paths: List[Path]) -> Path:
        """
        Merge knowledge from multiple specialist models into one generalist.
        
        Args:
            model_paths: List of model checkpoints to merge
        
        Returns:
            Path to merged model
        """
        logger.info(f"Merging {len(model_paths)} models")
        
        # In production, use:
        # - Weight averaging (simple)
        # - Fisher merging (better)
        # - Model soups (ensemble in weight space)
        
        merged_path = self.generations_dir / f"merged_gen{self.current_generation}.pt"
        
        # Mock merge
        if model_paths:
            shutil.copy(model_paths[0], merged_path)
        
        logger.info(f"✓ Merged model saved to {merged_path}")
        return merged_path
    
    def distill_smaller(self, teacher_path: Path, target_size_mb: float) -> Path:
        """
        Distill a large teacher model into a smaller student.
        
        Args:
            teacher_path: Path to teacher model
            target_size_mb: Target size for student model
        
        Returns:
            Path to distilled student model
        """
        logger.info(f"Distilling {teacher_path.name} to {target_size_mb}MB")
        
        # In production:
        # 1. Initialize smaller architecture
        # 2. Train student to match teacher outputs (soft targets)
        # 3. Use temperature scaling for better knowledge transfer
        
        student_path = self.generations_dir / f"{teacher_path.stem}_distilled.pt"
        
        # Mock distillation
        shutil.copy(teacher_path, student_path)
        
        logger.info(f"✓ Distilled model saved to {student_path}")
        return student_path
    
    def evolve(
        self,
        base_model: Dict,
        evaluation_fn,
        n_mutations: int = 3,
    ) -> List[Dict]:
        """
        Generate and evaluate mutated variants of a base model.
        
        Args:
            base_model: Dict with {path, accuracy, latency_ms, size_mb}
            evaluation_fn: Function to evaluate mutated models
            n_mutations: Number of mutation variants to try
        
        Returns:
            List of evaluated mutant models
        """
        import random
        
        base_path = Path(base_model["path"])
        mutants = []
        
        for _ in range(n_mutations):
            # Select random mutation operator
            mutation = random.choice(self.MUTATION_OPERATORS)
            
            logger.info(f"Applying mutation: {mutation}")
            
            try:
                if mutation == "prune_weights":
                    mutant_path = self.prune_weights(base_path, target_sparsity=0.3)
                elif mutation == "quantize_int8":
                    mutant_path = self.quantize_int8(base_path)
                elif mutation == "distill_smaller":
                    target_size = base_model["size_mb"] * 0.5
                    mutant_path = self.distill_smaller(base_path, target_size)
                else:
                    continue
                
                # Evaluate mutant
                mutant_metrics = evaluation_fn(mutant_path)
                
                fitness = self.compute_fitness(
                    mutant_metrics["accuracy"],
                    mutant_metrics["latency_ms"],
                    mutant_metrics["size_mb"],
                )
                
                mutants.append({
                    "path": mutant_path,
                    "mutation": mutation,
                    "accuracy": mutant_metrics["accuracy"],
                    "latency_ms": mutant_metrics["latency_ms"],
                    "size_mb": mutant_metrics["size_mb"],
                    "fitness": fitness,
                })
                
                logger.info(f"  Mutant fitness: {fitness:.3f}")
            
            except Exception as e:
                logger.error(f"Mutation {mutation} failed: {e}")
        
        return mutants
    
    def select_best(
        self,
        candidates: List[Dict],
        keep_top_n: int = 3,
    ) -> List[Dict]:
        """
        Select the best models from a population based on fitness.
        
        Args:
            candidates: List of evaluated models
            keep_top_n: Number of top models to keep
        
        Returns:
            Top N models sorted by fitness
        """
        sorted_candidates = sorted(candidates, key=lambda x: x["fitness"], reverse=True)
        return sorted_candidates[:keep_top_n]
    
    def archive_generation(self, generation: int, models: List[Dict]):
        """Archive a generation's models and metadata."""
        gen_dir = self.generations_dir / f"gen_{generation}"
        gen_dir.mkdir(exist_ok=True)
        
        for i, model in enumerate(models):
            # Copy model file
            src_path = Path(model["path"])
            dst_path = gen_dir / src_path.name
            shutil.copy(src_path, dst_path)
            
            # Save metadata
            meta_path = dst_path.with_suffix(".json")
            import json
            with open(meta_path, "w") as f:
                json.dump({
                    "generation": generation,
                    "rank": i + 1,
                    "fitness": model["fitness"],
                    "accuracy": model["accuracy"],
                    "latency_ms": model["latency_ms"],
                    "size_mb": model["size_mb"],
                }, f, indent=2)
        
        logger.info(f"Archived generation {generation} with {len(models)} models")
    
    def prune_old_generations(self, keep_n: int = 5):
        """Delete old generation directories to save disk space."""
        gen_dirs = sorted(self.generations_dir.glob("gen_*"))
        
        if len(gen_dirs) > keep_n:
            for old_dir in gen_dirs[:-keep_n]:
                shutil.rmtree(old_dir)
                logger.info(f"Pruned old generation: {old_dir.name}")
    
    def stats(self) -> Dict:
        """Return evolution statistics."""
        return {
            "current_generation": self.current_generation,
            "total_generations": len(self.lineage),
            "avg_fitness": (
                sum(g.fitness_score for g in self.lineage) / len(self.lineage)
                if self.lineage else 0.0
            ),
            "best_fitness": max((g.fitness_score for g in self.lineage), default=0.0),
        }
