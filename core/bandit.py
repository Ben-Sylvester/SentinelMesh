"""
B-22: This file is DEPRECATED.

EpsilonGreedyBandit and UCBPolicy have been replaced by ContextualBandit
(core/contextual_bandit.py) which implements LinUCB — a proper contextual
bandit that uses feature vectors rather than ignoring them.

BanditStore has been replaced by core/persistence/bandit_store.py which
persists via the shared SQLite connection (data/learning_state.db).

This file is retained temporarily to avoid breaking any external imports.
It will be removed in the next major version.
"""
import warnings

warnings.warn(
    "core.bandit is deprecated. Use core.contextual_bandit.ContextualBandit.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backwards compatibility
from core.contextual_bandit import ContextualBandit as EpsilonGreedyBandit  # noqa: F401
