"""
Plugin System

Extensible plugin architecture for custom components.
Hot-reload, sandboxed execution, marketplace ready.
"""

import importlib
import importlib.util
import sys
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
import time

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    entry_point: str
    config_schema: Dict[str, Any]


class PluginContext:
    """Context provided to plugins."""
    
    def __init__(self, router, memory_manager, integration_manager):
        self.router = router
        self.memory = memory_manager
        self.integrations = integration_manager
        self.storage = {}
    
    async def route(self, prompt: str, **kwargs):
        """Route request through main router."""
        return await self.router.route(prompt, **kwargs)
    
    async def store(self, key: str, value: Any):
        """Store plugin data."""
        self.storage[key] = value
    
    async def retrieve(self, key: str) -> Any:
        """Retrieve plugin data."""
        return self.storage.get(key)


class BasePlugin:
    """Base class for all plugins."""
    
    def __init__(self, context: PluginContext):
        self.context = context
        self.name = self.__class__.__name__
        self.enabled = True
        self.config = {}
    
    async def initialize(self, config: Dict[str, Any]):
        """Initialize plugin with configuration."""
        self.config = config
        logger.info(f"Initialized plugin: {self.name}")
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute plugin action."""
        raise NotImplementedError("Plugins must implement execute()")
    
    async def on_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming message (optional hook)."""
        return None
    
    async def on_response(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle outgoing response (optional hook)."""
        return None
    
    async def cleanup(self):
        """Cleanup plugin resources."""
        logger.info(f"Cleaning up plugin: {self.name}")
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name=self.name,
            version="1.0.0",
            description="Plugin description",
            author="Unknown",
            dependencies=[],
            entry_point=self.__class__.__name__,
            config_schema={}
        )


class PluginManager:
    """
    Manages plugin lifecycle, loading, execution, and marketplace.
    """
    
    def __init__(
        self,
        plugins_dir: str = "plugins",
        router=None,
        memory_manager=None,
        integration_manager=None
    ):
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.context = PluginContext(router, memory_manager, integration_manager)
        self.plugins: Dict[str, BasePlugin] = {}
        self.metadata: Dict[str, PluginMetadata] = {}
        
        self.db_path = self.plugins_dir / "plugins.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize plugin database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plugins (
                name            TEXT PRIMARY KEY,
                version         TEXT NOT NULL,
                description     TEXT,
                author          TEXT,
                installed_at    REAL NOT NULL,
                enabled         INTEGER DEFAULT 1,
                config          TEXT,
                metadata        TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plugin_executions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                plugin_name     TEXT NOT NULL,
                action          TEXT NOT NULL,
                executed_at     REAL NOT NULL,
                success         INTEGER NOT NULL,
                duration_ms     INTEGER,
                error           TEXT,
                
                INDEX idx_plugin (plugin_name),
                INDEX idx_executed (executed_at DESC)
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def install_plugin(self, plugin_path: str) -> str:
        """
        Install a plugin from file.
        
        Args:
            plugin_path: Path to plugin .py file or directory
        
        Returns:
            Plugin name
        """
        plugin_path = Path(plugin_path)
        
        # Load plugin module
        spec = importlib.util.spec_from_file_location(
            f"plugins.{plugin_path.stem}",
            plugin_path
        )
        if not spec or not spec.loader:
            raise ValueError(f"Cannot load plugin: {plugin_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        # Get plugin class
        plugin_class = getattr(module, "Plugin", None)
        if not plugin_class:
            raise ValueError("Plugin must define a 'Plugin' class")
        
        # Instantiate plugin
        plugin = plugin_class(self.context)
        metadata = plugin.get_metadata()
        
        # Check dependencies
        missing_deps = [
            dep for dep in metadata.dependencies
            if dep not in self.plugins
        ]
        if missing_deps:
            raise ValueError(f"Missing dependencies: {missing_deps}")
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO plugins
            (name, version, description, author, installed_at, config, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.name,
            metadata.version,
            metadata.description,
            metadata.author,
            time.time(),
            json.dumps({}),
            json.dumps({
                "dependencies": metadata.dependencies,
                "entry_point": metadata.entry_point,
                "config_schema": metadata.config_schema
            })
        ))
        conn.commit()
        conn.close()
        
        # Store plugin
        self.plugins[metadata.name] = plugin
        self.metadata[metadata.name] = metadata
        
        logger.info(f"Installed plugin: {metadata.name} v{metadata.version}")
        return metadata.name
    
    async def load_plugin(self, plugin_name: str, config: Optional[Dict] = None):
        """Load and initialize a plugin."""
        if plugin_name in self.plugins:
            logger.info(f"Plugin already loaded: {plugin_name}")
            return
        
        # Get from database
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("""
            SELECT name, version, config, metadata, enabled
            FROM plugins
            WHERE name = ?
        """, (plugin_name,)).fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"Plugin not installed: {plugin_name}")
        
        if not row[4]:  # enabled
            raise ValueError(f"Plugin disabled: {plugin_name}")
        
        # Load plugin (assume already in plugins directory)
        plugin_file = self.plugins_dir / f"{plugin_name}.py"
        if not plugin_file.exists():
            raise ValueError(f"Plugin file not found: {plugin_file}")
        
        # Import and initialize
        spec = importlib.util.spec_from_file_location(
            f"plugins.{plugin_name}",
            plugin_file
        )
        if not spec or not spec.loader:
            raise ValueError(f"Cannot load plugin: {plugin_name}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        plugin_class = getattr(module, "Plugin")
        plugin = plugin_class(self.context)
        
        # Initialize with config
        saved_config = json.loads(row[2]) if row[2] else {}
        final_config = {**saved_config, **(config or {})}
        await plugin.initialize(final_config)
        
        self.plugins[plugin_name] = plugin
        
        logger.info(f"Loaded plugin: {plugin_name}")
    
    async def execute_plugin(
        self,
        plugin_name: str,
        action: str,
        params: Dict[str, Any]
    ) -> Any:
        """Execute a plugin action."""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not loaded: {plugin_name}")
        
        plugin = self.plugins[plugin_name]
        if not plugin.enabled:
            raise ValueError(f"Plugin disabled: {plugin_name}")
        
        start_time = time.time()
        success = False
        error = None
        result = None
        
        try:
            result = await plugin.execute(action, params)
            success = True
        except Exception as e:
            error = str(e)
            logger.error(f"Plugin execution failed: {plugin_name}.{action} - {e}")
            raise
        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log execution
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO plugin_executions
                (plugin_name, action, executed_at, success, duration_ms, error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (plugin_name, action, time.time(), int(success), duration_ms, error))
            conn.commit()
            conn.close()
        
        return result
    
    async def unload_plugin(self, plugin_name: str):
        """Unload a plugin."""
        if plugin_name not in self.plugins:
            return
        
        plugin = self.plugins[plugin_name]
        await plugin.cleanup()
        
        del self.plugins[plugin_name]
        logger.info(f"Unloaded plugin: {plugin_name}")
    
    async def reload_plugin(self, plugin_name: str):
        """Reload a plugin (hot-reload)."""
        await self.unload_plugin(plugin_name)
        await self.load_plugin(plugin_name)
        logger.info(f"Reloaded plugin: {plugin_name}")
    
    def enable_plugin(self, plugin_name: str):
        """Enable a plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = True
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE plugins SET enabled = 1 WHERE name = ?
        """, (plugin_name,))
        conn.commit()
        conn.close()
        
        logger.info(f"Enabled plugin: {plugin_name}")
    
    def disable_plugin(self, plugin_name: str):
        """Disable a plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = False
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE plugins SET enabled = 0 WHERE name = ?
        """, (plugin_name,))
        conn.commit()
        conn.close()
        
        logger.info(f"Disabled plugin: {plugin_name}")
    
    def list_plugins(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        """List all plugins."""
        conn = sqlite3.connect(self.db_path)
        
        if include_disabled:
            rows = conn.execute("""
                SELECT name, version, description, author, enabled
                FROM plugins
                ORDER BY name
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT name, version, description, author, enabled
                FROM plugins
                WHERE enabled = 1
                ORDER BY name
            """).fetchall()
        
        conn.close()
        
        return [
            {
                "name": row[0],
                "version": row[1],
                "description": row[2],
                "author": row[3],
                "enabled": bool(row[4]),
                "loaded": row[0] in self.plugins
            }
            for row in rows
        ]
    
    def get_plugin_stats(self, plugin_name: str) -> Dict[str, Any]:
        """Get plugin statistics."""
        conn = sqlite3.connect(self.db_path)
        
        row = conn.execute("""
            SELECT
                COUNT(*) as total_executions,
                SUM(success) as successful_executions,
                AVG(duration_ms) as avg_duration_ms,
                MAX(executed_at) as last_execution
            FROM plugin_executions
            WHERE plugin_name = ?
        """, (plugin_name,)).fetchone()
        
        conn.close()
        
        return {
            "total_executions": row[0],
            "successful_executions": row[1],
            "failed_executions": row[0] - row[1],
            "avg_duration_ms": row[2],
            "last_execution": row[3]
        }


# Example plugin implementation
class ExamplePlugin(BasePlugin):
    """Example plugin implementation."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            description="Example plugin for demonstration",
            author="SentinelMesh",
            dependencies=[],
            entry_point="ExamplePlugin",
            config_schema={
                "api_key": {"type": "string", "required": True}
            }
        )
    
    async def execute(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute plugin action."""
        if action == "greet":
            name = params.get("name", "World")
            return {"message": f"Hello, {name}!"}
        
        elif action == "analyze":
            text = params.get("text", "")
            # Use context to route through main system
            result = await self.context.route(f"Analyze: {text}")
            return {"analysis": result.output}
        
        else:
            raise ValueError(f"Unknown action: {action}")
