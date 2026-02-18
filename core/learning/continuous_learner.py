"""
Continuous Learning Engine — Knowledge Distillation from External LLMs

Learns from every user interaction to progressively build self-models that
can replace external LLM calls for common tasks.
"""

import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example extracted from user interaction."""
    input_text: str
    output_text: str
    task_vector: np.ndarray  # embedding
    external_model: str
    reward: float
    timestamp: float
    cluster_id: Optional[int] = None


class ContinuousLearner:
    """
    Collects training data from every request and triggers model updates
    when sufficient high-quality examples accumulate.
    """
    
    def __init__(
        self,
        corpus_path: str = "data/training_corpus.db",
        min_quality_threshold: float = 0.7,
        batch_size: int = 1000,
        embedding_dim: int = 384,
    ):
        self.corpus_path = Path(corpus_path)
        self.min_quality_threshold = min_quality_threshold
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        
        # In-memory buffer (flush every N examples)
        self.buffer: List[TrainingExample] = []
        self.buffer_size = 100
        
        # Statistics
        self.total_collected = 0
        self.total_high_quality = 0
        
        self._init_storage()
        self._load_embedder()
    
    def _init_storage(self):
        """Initialize SQLite tables for training corpus."""
        import sqlite3
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.corpus_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_corpus (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                task_vector     BLOB NOT NULL,
                input_text      TEXT NOT NULL,
                output_text     TEXT NOT NULL,
                external_model  TEXT NOT NULL,
                reward          REAL NOT NULL,
                cluster_id      INTEGER,
                used_for_training BOOLEAN DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reward ON training_corpus(reward)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cluster ON training_corpus(cluster_id)")
        conn.commit()
        conn.close()
    
    def _load_embedder(self):
        """Load lightweight embedding model for task clustering."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embedding model: all-MiniLM-L6-v2 (384 dim)")
        except ImportError:
            logger.warning("sentence-transformers not installed - using random embeddings")
            self.embedder = None
    
    def embed_task(self, text: str) -> np.ndarray:
        """Generate task embedding for clustering."""
        if self.embedder:
            return self.embedder.encode(text, convert_to_numpy=True)
        else:
            # Fallback: hash-based pseudo-embedding
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return np.frombuffer(h[:self.embedding_dim * 4], dtype=np.float32)[:self.embedding_dim]
    
    def collect(
        self,
        input_text: str,
        output_text: str,
        external_model: str,
        reward: float,
    ) -> None:
        """
        Collect one training example from a user interaction.
        
        Args:
            input_text: User's original request
            output_text: LLM's response
            external_model: Which external model generated the response
            reward: Quality score (0.0 - 1.0) computed by reward function
        """
        # Filter low-quality examples
        if reward < self.min_quality_threshold:
            return
        
        # Generate task embedding
        task_vector = self.embed_task(input_text)
        
        example = TrainingExample(
            input_text=input_text,
            output_text=output_text,
            task_vector=task_vector,
            external_model=external_model,
            reward=reward,
            timestamp=time.time(),
        )
        
        self.buffer.append(example)
        self.total_collected += 1
        self.total_high_quality += 1
        
        # Flush buffer periodically
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffered examples to SQLite."""
        if not self.buffer:
            return
        
        import sqlite3
        conn = sqlite3.connect(self.corpus_path)
        
        for ex in self.buffer:
            conn.execute("""
                INSERT INTO training_corpus 
                (timestamp, task_vector, input_text, output_text, external_model, reward)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ex.timestamp)),
                ex.task_vector.tobytes(),
                ex.input_text,
                ex.output_text,
                ex.external_model,
                ex.reward,
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Flushed {len(self.buffer)} training examples to corpus")
        self.buffer.clear()
    
    def get_corpus_size(self) -> int:
        """Return total number of training examples collected."""
        import sqlite3
        conn = sqlite3.connect(self.corpus_path)
        size = conn.execute("SELECT COUNT(*) FROM training_corpus").fetchone()[0]
        conn.close()
        return size
    
    def get_training_batch(self, cluster_id: Optional[int] = None, limit: int = 1000) -> List[TrainingExample]:
        """
        Retrieve a batch of training examples.
        
        Args:
            cluster_id: If specified, only return examples from this cluster
            limit: Maximum number of examples to return
        
        Returns:
            List of TrainingExample objects
        """
        import sqlite3
        conn = sqlite3.connect(self.corpus_path)
        
        if cluster_id is not None:
            query = """
                SELECT task_vector, input_text, output_text, external_model, reward, timestamp
                FROM training_corpus 
                WHERE cluster_id = ?
                ORDER BY reward DESC
                LIMIT ?
            """
            rows = conn.execute(query, (cluster_id, limit)).fetchall()
        else:
            query = """
                SELECT task_vector, input_text, output_text, external_model, reward, timestamp
                FROM training_corpus 
                ORDER BY reward DESC
                LIMIT ?
            """
            rows = conn.execute(query, (limit,)).fetchall()
        
        conn.close()
        
        examples = []
        for row in rows:
            vector_bytes, input_text, output_text, ext_model, reward, ts = row
            task_vector = np.frombuffer(vector_bytes, dtype=np.float32)
            examples.append(TrainingExample(
                input_text=input_text,
                output_text=output_text,
                task_vector=task_vector,
                external_model=ext_model,
                reward=reward,
                timestamp=time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S")),
            ))
        
        return examples
    
    def compute_clusters(self, n_clusters: int = 20, min_cluster_size: int = 50):
        """
        Cluster all training examples into task families using K-means.
        
        Args:
            n_clusters: Number of task clusters to create
            min_cluster_size: Minimum examples per cluster
        """
        import sqlite3
        from sklearn.cluster import MiniBatchKMeans
        
        conn = sqlite3.connect(self.corpus_path)
        
        # Load all task vectors
        rows = conn.execute("SELECT id, task_vector FROM training_corpus").fetchall()
        if len(rows) < n_clusters * min_cluster_size:
            logger.warning(f"Insufficient data for clustering: {len(rows)} examples")
            conn.close()
            return
        
        ids = [row[0] for row in rows]
        vectors = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        
        # Cluster
        logger.info(f"Clustering {len(vectors)} examples into {n_clusters} task families...")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
        labels = kmeans.fit_predict(vectors)
        
        # Update database
        for ex_id, label in zip(ids, labels):
            conn.execute("UPDATE training_corpus SET cluster_id = ? WHERE id = ?", (int(label), ex_id))
        
        conn.commit()
        
        # Report cluster sizes
        cluster_sizes = {}
        for label in range(n_clusters):
            size = conn.execute("SELECT COUNT(*) FROM training_corpus WHERE cluster_id = ?", (label,)).fetchone()[0]
            cluster_sizes[label] = size
        
        conn.close()
        
        logger.info(f"Clustering complete. Cluster sizes: {cluster_sizes}")
        return cluster_sizes
    
    def stats(self) -> Dict:
        """Return learning statistics."""
        corpus_size = self.get_corpus_size()
        return {
            "total_collected": self.total_collected,
            "high_quality_kept": self.total_high_quality,
            "corpus_size_disk": corpus_size,
            "buffer_size": len(self.buffer),
            "quality_threshold": self.min_quality_threshold,
        }
