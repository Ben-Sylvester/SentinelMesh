# Troubleshooting Guide

## Database Schema Errors

### Error: `sqlite3.OperationalError: no such column: arm`

**Cause:** Your database was created with an older version of the schema.

**Solution 1: Run Migration Script (Recommended)**
```bash
python migrate_database.py
```

This will automatically:
- Detect old schema
- Backup existing data
- Migrate to new schema
- Restore your data

**Solution 2: Delete and Recreate Database**
If you don't need to preserve existing learning data:

```bash
# Windows
del data\learning_state.db

# Linux/Mac
rm data/learning_state.db
```

Then restart the application. The database will be created with the correct schema.

**Solution 3: Manual Migration**
```python
import sqlite3

conn = sqlite3.connect('data/learning_state.db')
cursor = conn.cursor()

# Check current schema
cursor.execute("PRAGMA table_info(bandit_state)")
print(cursor.fetchall())

# If 'arm' column is missing, run migration
cursor.execute("DROP TABLE IF EXISTS bandit_state")
cursor.execute("""
    CREATE TABLE bandit_state (
        arm        TEXT PRIMARY KEY,
        data       TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

conn.commit()
conn.close()
```

---

## Other Common Issues

### Import Errors

**Error: `ModuleNotFoundError: No module named 'sentence_transformers'`**

**Solution:**
```bash
pip install sentence-transformers scikit-learn
```

### Port Already in Use

**Error: `OSError: [Errno 48] Address already in use`**

**Solution:**
```bash
# Find and kill process on port 8000
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn app:app --port 8001
```

### API Key Not Found

**Error: `KeyError: 'OPENAI_API_KEY'`**

**Solution:**
1. Copy `.env.example` to `.env`
2. Add your API keys:
```bash
OPENAI_API_KEY=sk-...
```
3. Restart the server

### Database Locked

**Error: `sqlite3.OperationalError: database is locked`**

**Solution:**
- Close any SQLite browser/viewer applications
- Ensure only one instance of the app is running
- If persistent, restart the application

---

## Verification Steps

After fixing, verify everything works:

```bash
# 1. Check database schema
python -c "import sqlite3; conn = sqlite3.connect('data/learning_state.db'); print(conn.execute('PRAGMA table_info(bandit_state)').fetchall())"

# Expected output should include: ('arm', 'TEXT', ...)

# 2. Test the API
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'

# Should return a valid response
```

---

## Getting Help

If issues persist:

1. **Check logs** for detailed error messages
2. **Verify Python version** (3.10+ required)
3. **Ensure all dependencies installed**: `pip install -r requirements.txt`
4. **Check file permissions** on `data/` directory
5. **Try fresh installation** in a new virtual environment

For more help, see [README.md](README.md) and [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).
