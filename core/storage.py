"""
REMOVED: core/storage is a legacy file from a previous architecture.

Replacements:
  core/storage.py    → core/persistence/bandit_store.py + core/persistence/db.py
  core/budget.py     → core/tenant_budget.py
  core/persistence.py → core/persistence/ package (db.py, schema.py, *_store.py)

Do NOT import from this file. It will be deleted in the next major version.
"""
raise ImportError(
    "core.storage has been removed. "
    "See the module docstring for the replacement."
)
