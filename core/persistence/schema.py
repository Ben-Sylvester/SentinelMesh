"""
Database Schema Initialization with Automatic Migration

This module ensures all tables exist with the correct schema.
Handles migration from old schemas automatically.
"""

def init_schema(db):
    """
    Initialize or migrate database schema.
    
    This function is idempotent - safe to call multiple times.
    Automatically migrates from old schemas to new schemas.
    """
    
    cursor = db.cursor()
    
    # ═══════════════════════════════════════════════════════════════
    # Check if we need to migrate bandit_state
    # ═══════════════════════════════════════════════════════════════
    try:
        cursor.execute("PRAGMA table_info(bandit_state)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if columns and 'arm' not in columns:
            # Old schema detected - migrate
            print("Migrating bandit_state table...")
            
            # Backup old data
            cursor.execute("SELECT * FROM bandit_state")
            old_data = cursor.fetchall()
            
            # Drop and recreate
            cursor.execute("DROP TABLE bandit_state")
            cursor.execute("""
                CREATE TABLE bandit_state (
                    arm        TEXT PRIMARY KEY,
                    data       TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Restore data with new schema
            for row in old_data:
                cursor.execute(
                    "INSERT OR IGNORE INTO bandit_state (arm, data) VALUES (?, ?)",
                    (str(row[0]), str(row[1]))
                )
            
            print(f"✓ Migrated {len(old_data)} bandit records")
    except:
        pass
    
    # ═══════════════════════════════════════════════════════════════
    # Create all tables with current schema
    # ═══════════════════════════════════════════════════════════════
    
    # World model beliefs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS world_model (
            signature  TEXT PRIMARY KEY,
            data       TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Bandit state (LinUCB parameters)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bandit_state (
            arm        TEXT PRIMARY KEY,
            data       TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # RL Q-values
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rl_qvalues (
            state      TEXT NOT NULL,
            action     TEXT NOT NULL,
            value      REAL NOT NULL DEFAULT 0.0,
            count      INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (state, action)
        )
    """)
    
    # Meta-policy state
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meta_policy (
            key        TEXT PRIMARY KEY,
            data       TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Request traces
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL,
            strategy   TEXT,
            data       TEXT NOT NULL
        )
    """)
    
    # Create indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_strategy ON traces(strategy)")
    
    db.commit()
