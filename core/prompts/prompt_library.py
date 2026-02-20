"""
Prompt Library System

Save, version, and manage prompt templates with variable substitution.
Enables A/B testing and prompt optimization.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Prompt template with variables."""
    id: str
    name: str
    template: str
    variables: List[str]
    description: str
    category: str
    version: int
    created_at: float
    updated_at: float
    usage_count: int
    avg_rating: float
    metadata: Dict[str, Any]


@dataclass
class PromptVersion:
    """Version of a prompt template."""
    template_id: str
    version: int
    template: str
    created_at: float
    created_by: str
    change_notes: str


class PromptLibrary:
    """
    Library for managing prompt templates with versioning.
    """
    
    def __init__(self, storage_path: str = "data/prompts"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "prompt_library.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize database for prompt library."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_templates (
                id              TEXT PRIMARY KEY,
                name            TEXT UNIQUE NOT NULL,
                template        TEXT NOT NULL,
                variables       TEXT NOT NULL,
                description     TEXT,
                category        TEXT,
                version         INTEGER DEFAULT 1,
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL,
                usage_count     INTEGER DEFAULT 0,
                avg_rating      REAL DEFAULT 0.0,
                metadata        TEXT,
                
                INDEX idx_category (category),
                INDEX idx_name (name)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_versions (
                template_id     TEXT NOT NULL,
                version         INTEGER NOT NULL,
                template        TEXT NOT NULL,
                created_at      REAL NOT NULL,
                created_by      TEXT,
                change_notes    TEXT,
                
                PRIMARY KEY (template_id, version),
                FOREIGN KEY (template_id) REFERENCES prompt_templates(id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_ratings (
                template_id     TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                rating          INTEGER NOT NULL,
                comment         TEXT,
                created_at      REAL NOT NULL,
                
                PRIMARY KEY (template_id, user_id),
                FOREIGN KEY (template_id) REFERENCES prompt_templates(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template."""
        import re
        return list(set(re.findall(r'\{(\w+)\}', template)))
    
    def save(
        self,
        name: str,
        template: str,
        description: str = "",
        category: str = "general",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a new prompt template.
        
        Args:
            name: Unique template name
            template: Template string with {variables}
            description: Template description
            category: Template category
            metadata: Optional metadata
        
        Returns:
            Template ID
        """
        import hashlib
        template_id = hashlib.sha256(f"{name}{time.time()}".encode()).hexdigest()[:16]
        
        variables = self._extract_variables(template)
        now = time.time()
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            conn.execute("""
                INSERT INTO prompt_templates 
                (id, name, template, variables, description, category, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                template_id,
                name,
                template,
                json.dumps(variables),
                description,
                category,
                now,
                now,
                json.dumps(metadata) if metadata else None
            ))
            
            # Save first version
            conn.execute("""
                INSERT INTO prompt_versions
                (template_id, version, template, created_at, created_by, change_notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (template_id, 1, template, now, "system", "Initial version"))
            
            conn.commit()
            logger.info(f"Saved prompt template: {name}")
            return template_id
        
        except sqlite3.IntegrityError:
            conn.close()
            raise ValueError(f"Template with name '{name}' already exists")
        
        finally:
            conn.close()
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get prompt template by name."""
        conn = sqlite3.connect(self.db_path)
        
        row = conn.execute("""
            SELECT id, name, template, variables, description, category, version,
                   created_at, updated_at, usage_count, avg_rating, metadata
            FROM prompt_templates
            WHERE name = ?
        """, (name,)).fetchone()
        
        conn.close()
        
        if not row:
            return None
        
        return PromptTemplate(
            id=row[0],
            name=row[1],
            template=row[2],
            variables=json.loads(row[3]),
            description=row[4] or "",
            category=row[5] or "general",
            version=row[6],
            created_at=row[7],
            updated_at=row[8],
            usage_count=row[9],
            avg_rating=row[10],
            metadata=json.loads(row[11]) if row[11] else {}
        )
    
    def render(self, name: str, **variables) -> str:
        """
        Render a template with variables.
        
        Args:
            name: Template name
            **variables: Variable values
        
        Returns:
            Rendered prompt
        """
        template = self.get(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        # Increment usage count
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE prompt_templates
            SET usage_count = usage_count + 1
            WHERE name = ?
        """, (name,))
        conn.commit()
        conn.close()
        
        # Render template
        try:
            return template.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
    
    def update(
        self,
        name: str,
        template: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        change_notes: str = "",
        updated_by: str = "system"
    ):
        """Update an existing template (creates new version)."""
        current = self.get(name)
        if not current:
            raise ValueError(f"Template '{name}' not found")
        
        conn = sqlite3.connect(self.db_path)
        
        # Update main record
        updates = []
        params = []
        
        if template:
            updates.append("template = ?")
            params.append(template)
            updates.append("variables = ?")
            params.append(json.dumps(self._extract_variables(template)))
            updates.append("version = version + 1")
        
        if description:
            updates.append("description = ?")
            params.append(description)
        
        if category:
            updates.append("category = ?")
            params.append(category)
        
        updates.append("updated_at = ?")
        params.append(time.time())
        
        params.append(name)
        
        conn.execute(f"""
            UPDATE prompt_templates
            SET {", ".join(updates)}
            WHERE name = ?
        """, params)
        
        # Create new version
        if template:
            new_version = current.version + 1
            conn.execute("""
                INSERT INTO prompt_versions
                (template_id, version, template, created_at, created_by, change_notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (current.id, new_version, template, time.time(), updated_by, change_notes))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated prompt template: {name}")
    
    def list_templates(
        self,
        category: Optional[str] = None
    ) -> List[PromptTemplate]:
        """List all templates, optionally filtered by category."""
        conn = sqlite3.connect(self.db_path)
        
        if category:
            rows = conn.execute("""
                SELECT id, name, template, variables, description, category, version,
                       created_at, updated_at, usage_count, avg_rating, metadata
                FROM prompt_templates
                WHERE category = ?
                ORDER BY name
            """, (category,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, name, template, variables, description, category, version,
                       created_at, updated_at, usage_count, avg_rating, metadata
                FROM prompt_templates
                ORDER BY name
            """).fetchall()
        
        conn.close()
        
        return [
            PromptTemplate(
                id=row[0],
                name=row[1],
                template=row[2],
                variables=json.loads(row[3]),
                description=row[4] or "",
                category=row[5] or "general",
                version=row[6],
                created_at=row[7],
                updated_at=row[8],
                usage_count=row[9],
                avg_rating=row[10],
                metadata=json.loads(row[11]) if row[11] else {}
            )
            for row in rows
        ]
    
    def get_version(self, name: str, version: int) -> Optional[str]:
        """Get a specific version of a template."""
        template = self.get(name)
        if not template:
            return None
        
        conn = sqlite3.connect(self.db_path)
        
        row = conn.execute("""
            SELECT template
            FROM prompt_versions
            WHERE template_id = ? AND version = ?
        """, (template.id, version)).fetchone()
        
        conn.close()
        
        return row[0] if row else None
    
    def rate(self, name: str, user_id: str, rating: int, comment: str = ""):
        """Rate a template (1-5 stars)."""
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        template = self.get(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        conn = sqlite3.connect(self.db_path)
        
        # Insert or update rating
        conn.execute("""
            INSERT INTO prompt_ratings (template_id, user_id, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(template_id, user_id) DO UPDATE SET
                rating = excluded.rating,
                comment = excluded.comment,
                created_at = excluded.created_at
        """, (template.id, user_id, rating, comment, time.time()))
        
        # Recalculate average rating
        avg_rating = conn.execute("""
            SELECT AVG(rating) FROM prompt_ratings WHERE template_id = ?
        """, (template.id,)).fetchone()[0]
        
        conn.execute("""
            UPDATE prompt_templates SET avg_rating = ? WHERE id = ?
        """, (avg_rating, template.id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Rated template {name}: {rating}/5")
    
    def delete(self, name: str):
        """Delete a template and all its versions."""
        template = self.get(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("DELETE FROM prompt_versions WHERE template_id = ?", (template.id,))
        conn.execute("DELETE FROM prompt_ratings WHERE template_id = ?", (template.id,))
        conn.execute("DELETE FROM prompt_templates WHERE id = ?", (template.id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted template: {name}")
