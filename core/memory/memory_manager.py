"""
Memory & Context Management System

Provides long-term memory for conversations, user preferences, and semantic search
over past interactions. Makes SentinelMesh stateful like a true AI OS.
"""

import time
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry with embedding."""
    id: str
    user_id: str
    session_id: str
    timestamp: float
    prompt: str
    response: str
    metadata: Dict[str, Any]


@dataclass
class ConversationContext:
    """Context for a conversation session."""
    session_id: str
    user_id: str
    messages: List[Dict[str, str]]
    started_at: float
    last_active: float
    preferences: Dict[str, Any]


class VectorStore:
    """
    Vector storage using SQLite + numpy embeddings.
    Provides semantic search over conversation history.
    """
    
    def __init__(self, storage_path: str = "data/memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "memories.db"
        self._init_db()
        
        # Load embedder
        self._load_embedder()
    
    def _init_db(self):
        """Initialize SQLite database for memories."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                session_id      TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                prompt          TEXT NOT NULL,
                response        TEXT NOT NULL,
                embedding       BLOB NOT NULL,
                metadata        TEXT,
                
                INDEX idx_user (user_id),
                INDEX idx_session (session_id),
                INDEX idx_timestamp (timestamp DESC)
            )
        """)
        conn.commit()
        conn.close()
    
    def _load_embedder(self):
        """Load sentence transformer for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
            logger.info("Loaded sentence-transformers embedder")
        except ImportError:
            logger.warning("sentence-transformers not installed, using fallback embeddings")
            self.embedder = None
            self.embedding_dim = 384
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self.embedder:
            return self.embedder.encode(text, convert_to_numpy=True)
        else:
            # Fallback: simple hash-based embedding
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            arr = np.frombuffer(h, dtype=np.uint8)[:self.embedding_dim]
            return arr.astype(np.float32) / 255.0
    
    def add(self, entry: MemoryEntry):
        """Add a new memory entry."""
        # Generate embedding
        combined_text = f"{entry.prompt} {entry.response}"
        embedding = self.embed_text(combined_text)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO memories (id, user_id, session_id, timestamp, prompt, response, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.user_id,
            entry.session_id,
            entry.timestamp,
            entry.prompt,
            entry.response,
            embedding.tobytes(),
            json.dumps(entry.metadata)
        ))
        conn.commit()
        conn.close()
    
    def search(
        self,
        query_text: str,
        user_id: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[MemoryEntry]:
        """Search for similar memories using cosine similarity."""
        query_embedding = self.embed_text(query_text)
        
        conn = sqlite3.connect(self.db_path)
        
        # Get all user memories
        rows = conn.execute("""
            SELECT id, user_id, session_id, timestamp, prompt, response, embedding, metadata
            FROM memories
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1000
        """, (user_id,)).fetchall()
        
        conn.close()
        
        if not rows:
            return []
        
        # Compute similarities
        results = []
        for row in rows:
            embedding = np.frombuffer(row[6], dtype=np.float32)
            
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
            )
            
            if similarity >= min_similarity:
                metadata = json.loads(row[7]) if row[7] else {}
                results.append((similarity, MemoryEntry(
                    id=row[0],
                    user_id=row[1],
                    session_id=row[2],
                    timestamp=row[3],
                    prompt=row[4],
                    response=row[5],
                    metadata=metadata
                )))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:limit]]
    
    def get_recent(self, user_id: str, limit: int = 10) -> List[MemoryEntry]:
        """Get recent memories for a user."""
        conn = sqlite3.connect(self.db_path)
        
        rows = conn.execute("""
            SELECT id, user_id, session_id, timestamp, prompt, response, metadata
            FROM memories
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
        
        conn.close()
        
        return [
            MemoryEntry(
                id=row[0],
                user_id=row[1],
                session_id=row[2],
                timestamp=row[3],
                prompt=row[4],
                response=row[5],
                metadata=json.loads(row[6]) if row[6] else {}
            )
            for row in rows
        ]


class MemoryManager:
    """
    Long-term memory management for conversations and user preferences.
    Provides semantic search, conversation history, and preference learning.
    """
    
    def __init__(self, storage_path: str = "data/memory"):
        self.vector_store = VectorStore(storage_path)
        self.storage_path = Path(storage_path)
        
        # Active conversations
        self.conversations: Dict[str, ConversationContext] = {}
        
        # User preferences
        self.preferences: Dict[str, Dict] = {}
        self._load_preferences()
    
    def _load_preferences(self):
        """Load saved user preferences."""
        prefs_path = self.storage_path / "preferences.json"
        if prefs_path.exists():
            try:
                with open(prefs_path) as f:
                    self.preferences = json.load(f)
                logger.info(f"Loaded preferences for {len(self.preferences)} users")
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}")
    
    def _save_preferences(self):
        """Save user preferences."""
        prefs_path = self.storage_path / "preferences.json"
        try:
            with open(prefs_path, "w") as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
    
    async def store_interaction(
        self,
        user_id: str,
        session_id: str,
        prompt: str,
        response: str,
        metadata: Optional[Dict] = None
    ):
        """Store an interaction in long-term memory."""
        entry = MemoryEntry(
            id=f"{session_id}_{int(time.time()*1000)}",
            user_id=user_id,
            session_id=session_id,
            timestamp=time.time(),
            prompt=prompt,
            response=response,
            metadata=metadata or {}
        )
        
        # Store in vector store
        self.vector_store.add(entry)
        
        # Update conversation context
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                messages=[],
                started_at=time.time(),
                last_active=time.time(),
                preferences=self.preferences.get(user_id, {})
            )
        
        ctx = self.conversations[session_id]
        ctx.messages.append({"role": "user", "content": prompt})
        ctx.messages.append({"role": "assistant", "content": response})
        ctx.last_active = time.time()
        
        # Learn preferences
        self._learn_preferences(user_id, prompt, response, metadata)
    
    def _learn_preferences(
        self,
        user_id: str,
        prompt: str,
        response: str,
        metadata: Optional[Dict]
    ):
        """Learn user preferences from interactions."""
        if user_id not in self.preferences:
            self.preferences[user_id] = {
                "response_length": "medium",
                "technical_level": "medium",
                "format_preference": "prose",
                "interaction_count": 0
            }
        
        prefs = self.preferences[user_id]
        prefs["interaction_count"] = prefs.get("interaction_count", 0) + 1
        
        # Detect response length preference
        if len(response) < 200:
            prefs["response_length"] = self._update_preference(
                prefs.get("response_length", "medium"), "short", 0.2
            )
        elif len(response) > 1000:
            prefs["response_length"] = self._update_preference(
                prefs.get("response_length", "medium"), "long", 0.2
            )
        
        # Detect technical level
        technical_words = ["algorithm", "implementation", "technical", "code", "function"]
        if any(word in prompt.lower() for word in technical_words):
            prefs["technical_level"] = self._update_preference(
                prefs.get("technical_level", "medium"), "high", 0.2
            )
        
        # Save periodically
        if prefs["interaction_count"] % 5 == 0:
            self._save_preferences()
    
    def _update_preference(self, current: str, new: str, weight: float) -> str:
        """Update preference with weighted average."""
        levels = {"short": 0, "medium": 1, "long": 2, "low": 0, "high": 2}
        
        if current not in levels or new not in levels:
            return current
        
        current_val = levels.get(current, 1)
        new_val = levels.get(new, 1)
        updated_val = (1 - weight) * current_val + weight * new_val
        
        reverse = {0: "short", 1: "medium", 2: "long"} if "short" in [current, new] else {0: "low", 1: "medium", 2: "high"}
        return reverse[round(updated_val)]
    
    async def recall_context(
        self,
        user_id: str,
        current_prompt: str,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from memory.
        
        Returns dict with:
        - memories: List of relevant past interactions
        - preferences: User preferences
        - memory_count: Number of memories found
        """
        # Search for similar past interactions
        relevant_memories = self.vector_store.search(
            query_text=current_prompt,
            user_id=user_id,
            limit=k,
            min_similarity=0.7
        )
        
        # Format memories
        memory_texts = []
        for mem in relevant_memories:
            memory_texts.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M", time.localtime(mem.timestamp)),
                "prompt": mem.prompt,
                "response": mem.response[:200] + "..." if len(mem.response) > 200 else mem.response
            })
        
        return {
            "memories": memory_texts,
            "preferences": self.preferences.get(user_id, {}),
            "memory_count": len(relevant_memories)
        }
    
    def get_conversation_history(
        self,
        session_id: str,
        last_n: int = 10
    ) -> List[Dict[str, str]]:
        """Get recent conversation history for a session."""
        if session_id not in self.conversations:
            return []
        
        messages = self.conversations[session_id].messages
        return messages[-last_n:] if len(messages) > last_n else messages
    
    def clear_session(self, session_id: str):
        """Clear a conversation session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        return self.preferences.get(user_id, {})
    
    def set_user_preference(self, user_id: str, key: str, value: Any):
        """Manually set a user preference."""
        if user_id not in self.preferences:
            self.preferences[user_id] = {}
        
        self.preferences[user_id][key] = value
        self._save_preferences()
    
    def stats(self) -> Dict:
        """Return memory system statistics."""
        conn = sqlite3.connect(self.vector_store.db_path)
        total_memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        
        return {
            "total_memories": total_memories,
            "active_sessions": len(self.conversations),
            "users_with_preferences": len(self.preferences),
            "storage_path": str(self.storage_path)
        }
