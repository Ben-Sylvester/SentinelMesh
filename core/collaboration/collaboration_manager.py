"""
Collaboration System

Team workspaces, shared memory, RBAC, and activity feeds.
Enterprise collaboration features for AI workflows.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """User roles in workspace."""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Workspace permissions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class Workspace:
    """Team workspace."""
    id: str
    name: str
    description: str
    owner_id: str
    created_at: float
    members: List[str]
    settings: Dict[str, Any]


@dataclass
class WorkspaceMember:
    """Workspace member."""
    user_id: str
    workspace_id: str
    role: Role
    joined_at: float
    permissions: List[Permission]


@dataclass
class Activity:
    """Activity log entry."""
    id: str
    workspace_id: str
    user_id: str
    action: str
    details: Dict[str, Any]
    timestamp: float


class CollaborationManager:
    """
    Manages team workspaces, permissions, and shared resources.
    """
    
    def __init__(self, storage_path: str = "data/collaboration"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "collaboration.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize collaboration database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                description     TEXT,
                owner_id        TEXT NOT NULL,
                created_at      REAL NOT NULL,
                settings        TEXT,
                
                INDEX idx_owner (owner_id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workspace_members (
                workspace_id    TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                role            TEXT NOT NULL,
                joined_at       REAL NOT NULL,
                permissions     TEXT NOT NULL,
                
                PRIMARY KEY (workspace_id, user_id),
                FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_memory (
                workspace_id    TEXT NOT NULL,
                key             TEXT NOT NULL,
                value           TEXT NOT NULL,
                created_by      TEXT NOT NULL,
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL,
                
                PRIMARY KEY (workspace_id, key),
                FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id              TEXT PRIMARY KEY,
                workspace_id    TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                action          TEXT NOT NULL,
                details         TEXT,
                timestamp       REAL NOT NULL,
                
                INDEX idx_workspace (workspace_id, timestamp DESC),
                FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_workspace(
        self,
        name: str,
        owner_id: str,
        description: str = "",
        settings: Optional[Dict] = None
    ) -> str:
        """Create a new workspace."""
        import uuid
        workspace_id = str(uuid.uuid4())[:16]
        
        conn = sqlite3.connect(self.db_path)
        
        # Create workspace
        conn.execute("""
            INSERT INTO workspaces
            (id, name, description, owner_id, created_at, settings)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            workspace_id,
            name,
            description,
            owner_id,
            time.time(),
            json.dumps(settings or {})
        ))
        
        # Add owner as member
        conn.execute("""
            INSERT INTO workspace_members
            (workspace_id, user_id, role, joined_at, permissions)
            VALUES (?, ?, ?, ?, ?)
        """, (
            workspace_id,
            owner_id,
            Role.OWNER.value,
            time.time(),
            json.dumps([p.value for p in Permission])
        ))
        
        conn.commit()
        conn.close()
        
        # Log activity
        self._log_activity(
            workspace_id,
            owner_id,
            "workspace_created",
            {"name": name}
        )
        
        logger.info(f"Created workspace: {name} ({workspace_id})")
        return workspace_id
    
    def add_member(
        self,
        workspace_id: str,
        user_id: str,
        role: Role = Role.MEMBER,
        added_by: str = None
    ):
        """Add a member to workspace."""
        # Check if user already member
        conn = sqlite3.connect(self.db_path)
        existing = conn.execute("""
            SELECT user_id FROM workspace_members
            WHERE workspace_id = ? AND user_id = ?
        """, (workspace_id, user_id)).fetchone()
        
        if existing:
            conn.close()
            raise ValueError("User already in workspace")
        
        # Determine permissions based on role
        if role == Role.OWNER or role == Role.ADMIN:
            permissions = [p.value for p in Permission]
        elif role == Role.MEMBER:
            permissions = [Permission.READ.value, Permission.WRITE.value, Permission.EXECUTE.value]
        else:  # VIEWER
            permissions = [Permission.READ.value]
        
        # Add member
        conn.execute("""
            INSERT INTO workspace_members
            (workspace_id, user_id, role, joined_at, permissions)
            VALUES (?, ?, ?, ?, ?)
        """, (workspace_id, user_id, role.value, time.time(), json.dumps(permissions)))
        
        conn.commit()
        conn.close()
        
        # Log activity
        self._log_activity(
            workspace_id,
            added_by or "system",
            "member_added",
            {"user_id": user_id, "role": role.value}
        )
        
        logger.info(f"Added {user_id} to workspace {workspace_id} as {role.value}")
    
    def remove_member(self, workspace_id: str, user_id: str, removed_by: str):
        """Remove a member from workspace."""
        conn = sqlite3.connect(self.db_path)
        
        # Don't allow removing owner
        role = conn.execute("""
            SELECT role FROM workspace_members
            WHERE workspace_id = ? AND user_id = ?
        """, (workspace_id, user_id)).fetchone()
        
        if role and role[0] == Role.OWNER.value:
            conn.close()
            raise ValueError("Cannot remove workspace owner")
        
        conn.execute("""
            DELETE FROM workspace_members
            WHERE workspace_id = ? AND user_id = ?
        """, (workspace_id, user_id))
        
        conn.commit()
        conn.close()
        
        # Log activity
        self._log_activity(
            workspace_id,
            removed_by,
            "member_removed",
            {"user_id": user_id}
        )
        
        logger.info(f"Removed {user_id} from workspace {workspace_id}")
    
    def update_member_role(
        self,
        workspace_id: str,
        user_id: str,
        new_role: Role,
        updated_by: str
    ):
        """Update member role."""
        conn = sqlite3.connect(self.db_path)
        
        # Determine new permissions
        if new_role == Role.OWNER or new_role == Role.ADMIN:
            permissions = [p.value for p in Permission]
        elif new_role == Role.MEMBER:
            permissions = [Permission.READ.value, Permission.WRITE.value, Permission.EXECUTE.value]
        else:
            permissions = [Permission.READ.value]
        
        conn.execute("""
            UPDATE workspace_members
            SET role = ?, permissions = ?
            WHERE workspace_id = ? AND user_id = ?
        """, (new_role.value, json.dumps(permissions), workspace_id, user_id))
        
        conn.commit()
        conn.close()
        
        # Log activity
        self._log_activity(
            workspace_id,
            updated_by,
            "role_updated",
            {"user_id": user_id, "new_role": new_role.value}
        )
    
    def check_permission(
        self,
        workspace_id: str,
        user_id: str,
        required_permission: Permission
    ) -> bool:
        """Check if user has permission in workspace."""
        conn = sqlite3.connect(self.db_path)
        
        row = conn.execute("""
            SELECT permissions FROM workspace_members
            WHERE workspace_id = ? AND user_id = ?
        """, (workspace_id, user_id)).fetchone()
        
        conn.close()
        
        if not row:
            return False
        
        permissions = json.loads(row[0])
        return required_permission.value in permissions
    
    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace details."""
        conn = sqlite3.connect(self.db_path)
        
        row = conn.execute("""
            SELECT id, name, description, owner_id, created_at, settings
            FROM workspaces
            WHERE id = ?
        """, (workspace_id,)).fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Get members
        members = conn.execute("""
            SELECT user_id FROM workspace_members
            WHERE workspace_id = ?
        """, (workspace_id,)).fetchall()
        
        conn.close()
        
        return Workspace(
            id=row[0],
            name=row[1],
            description=row[2],
            owner_id=row[3],
            created_at=row[4],
            members=[m[0] for m in members],
            settings=json.loads(row[5]) if row[5] else {}
        )
    
    def list_user_workspaces(self, user_id: str) -> List[Dict]:
        """List workspaces user is member of."""
        conn = sqlite3.connect(self.db_path)
        
        rows = conn.execute("""
            SELECT w.id, w.name, w.description, wm.role
            FROM workspaces w
            JOIN workspace_members wm ON w.id = wm.workspace_id
            WHERE wm.user_id = ?
            ORDER BY w.created_at DESC
        """, (user_id,)).fetchall()
        
        conn.close()
        
        return [
            {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "role": row[3]
            }
            for row in rows
        ]
    
    def set_shared_data(
        self,
        workspace_id: str,
        key: str,
        value: Any,
        user_id: str
    ):
        """Store shared data in workspace."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO shared_memory
            (workspace_id, key, value, created_by, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            workspace_id,
            key,
            json.dumps(value),
            user_id,
            time.time(),
            time.time()
        ))
        
        conn.commit()
        conn.close()
        
        # Log activity
        self._log_activity(
            workspace_id,
            user_id,
            "data_updated",
            {"key": key}
        )
    
    def get_shared_data(self, workspace_id: str, key: str) -> Optional[Any]:
        """Retrieve shared data from workspace."""
        conn = sqlite3.connect(self.db_path)
        
        row = conn.execute("""
            SELECT value FROM shared_memory
            WHERE workspace_id = ? AND key = ?
        """, (workspace_id, key)).fetchone()
        
        conn.close()
        
        if not row:
            return None
        
        return json.loads(row[0])
    
    def list_shared_data(self, workspace_id: str) -> List[Dict]:
        """List all shared data in workspace."""
        conn = sqlite3.connect(self.db_path)
        
        rows = conn.execute("""
            SELECT key, created_by, created_at, updated_at
            FROM shared_memory
            WHERE workspace_id = ?
            ORDER BY updated_at DESC
        """, (workspace_id,)).fetchall()
        
        conn.close()
        
        return [
            {
                "key": row[0],
                "created_by": row[1],
                "created_at": row[2],
                "updated_at": row[3]
            }
            for row in rows
        ]
    
    def get_activity_feed(
        self,
        workspace_id: str,
        limit: int = 50
    ) -> List[Activity]:
        """Get activity feed for workspace."""
        conn = sqlite3.connect(self.db_path)
        
        rows = conn.execute("""
            SELECT id, workspace_id, user_id, action, details, timestamp
            FROM activity_log
            WHERE workspace_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (workspace_id, limit)).fetchall()
        
        conn.close()
        
        return [
            Activity(
                id=row[0],
                workspace_id=row[1],
                user_id=row[2],
                action=row[3],
                details=json.loads(row[4]) if row[4] else {},
                timestamp=row[5]
            )
            for row in rows
        ]
    
    def _log_activity(
        self,
        workspace_id: str,
        user_id: str,
        action: str,
        details: Dict[str, Any]
    ):
        """Log activity to workspace feed."""
        import uuid
        activity_id = str(uuid.uuid4())[:16]
        
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO activity_log
            (id, workspace_id, user_id, action, details, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            activity_id,
            workspace_id,
            user_id,
            action,
            json.dumps(details),
            time.time()
        ))
        
        conn.commit()
        conn.close()
