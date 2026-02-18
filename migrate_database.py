"""
Database Schema Migration Script

Run this script ONCE to migrate your database from the old schema to the new schema.

Usage:
    python migrate_database.py
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

def migrate_database():
    """Migrate all database schemas to the latest version."""
    
    db_path = Path("data/learning_state.db")
    
    # Create data directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If database doesn't exist, it will be created with correct schema automatically
    if not db_path.exists():
        print("✓ No existing database found - will be created with correct schema on first run")
        return
    
    print(f"Migrating database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # ═══════════════════════════════════════════════════════════════
    # 1. Migrate bandit_state table
    # ═══════════════════════════════════════════════════════════════
    print("\n[1/5] Checking bandit_state table...")
    
    # Check current schema
    cur.execute("PRAGMA table_info(bandit_state)")
    columns = {row[1]: row[2] for row in cur.fetchall()}
    
    if 'arm' not in columns:
        print("  → Migrating bandit_state (old schema detected)")
        
        # Backup old data
        try:
            cur.execute("SELECT * FROM bandit_state")
            old_rows = cur.fetchall()
            
            # Get column names from old schema
            cur.execute("PRAGMA table_info(bandit_state)")
            old_cols = [row[1] for row in cur.fetchall()]
            
            print(f"  → Backing up {len(old_rows)} existing records")
            
            # Drop old table
            cur.execute("DROP TABLE bandit_state")
            
            # Create new table
            cur.execute("""
                CREATE TABLE bandit_state (
                    arm        TEXT PRIMARY KEY,
                    data       TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Migrate data
            if old_rows:
                for row in old_rows:
                    # Old schema was likely (id, state) or (arm_name, state)
                    arm_name = row[0]
                    state_data = row[1]
                    
                    # Ensure state_data is valid JSON
                    if isinstance(state_data, str):
                        try:
                            json.loads(state_data)  # Validate
                            data = state_data
                        except:
                            data = json.dumps({"error": "invalid_state"})
                    else:
                        data = json.dumps(state_data)
                    
                    cur.execute(
                        "INSERT OR IGNORE INTO bandit_state (arm, data) VALUES (?, ?)",
                        (str(arm_name), data)
                    )
                
                print(f"  ✓ Migrated {len(old_rows)} records")
            
        except sqlite3.OperationalError as e:
            print(f"  → Creating new bandit_state table (no old data)")
            cur.execute("DROP TABLE IF EXISTS bandit_state")
            cur.execute("""
                CREATE TABLE bandit_state (
                    arm        TEXT PRIMARY KEY,
                    data       TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    else:
        print("  ✓ bandit_state already has correct schema")
    
    # ═══════════════════════════════════════════════════════════════
    # 2. Ensure rl_qvalues table exists
    # ═══════════════════════════════════════════════════════════════
    print("\n[2/5] Checking rl_qvalues table...")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rl_qvalues (
            state      TEXT NOT NULL,
            action     TEXT NOT NULL,
            value      REAL NOT NULL DEFAULT 0.0,
            count      INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (state, action)
        )
    """)
    print("  ✓ rl_qvalues table ready")
    
    # ═══════════════════════════════════════════════════════════════
    # 3. Ensure world_model table exists
    # ═══════════════════════════════════════════════════════════════
    print("\n[3/5] Checking world_model table...")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS world_model (
            signature  TEXT PRIMARY KEY,
            data       TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("  ✓ world_model table ready")
    
    # ═══════════════════════════════════════════════════════════════
    # 4. Ensure meta_policy table exists
    # ═══════════════════════════════════════════════════════════════
    print("\n[4/5] Checking meta_policy table...")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meta_policy (
            key        TEXT PRIMARY KEY,
            data       TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("  ✓ meta_policy table ready")
    
    # ═══════════════════════════════════════════════════════════════
    # 5. Ensure traces table exists
    # ═══════════════════════════════════════════════════════════════
    print("\n[5/5] Checking traces table...")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL,
            strategy   TEXT,
            data       TEXT NOT NULL
        )
    """)
    print("  ✓ traces table ready")
    
    # ═══════════════════════════════════════════════════════════════
    # Finalize
    # ═══════════════════════════════════════════════════════════════
    conn.commit()
    conn.close()
    
    print("\n" + "="*60)
    print("✅ Database migration complete!")
    print("="*60)
    print("\nYou can now run: uvicorn app:app --reload")


if __name__ == "__main__":
    migrate_database()
