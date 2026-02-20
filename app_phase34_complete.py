# Complete Phase 3 & 4 Integration
# Add these endpoints to the existing app.py

from core.plugins.plugin_system import PluginManager
from core.collaboration.collaboration_manager import CollaborationManager
from core.voice.voice_manager import VoiceManager

# Initialize Phase 4 systems
plugin_manager = PluginManager(router=router, memory_manager=memory_manager, integration_manager=integration_manager)
collaboration_manager = CollaborationManager()
voice_manager = VoiceManager()
voice_manager.router = router

logger.info("✅ Phase 4 upgrades loaded: Plugins, Collaboration, Voice")


# ══════════════════════════════════════════════════════════════════════════════
# PLUGIN SYSTEM ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/plugins/install")
async def install_plugin(plugin_path: str):
    """Install a plugin from file."""
    try:
        plugin_name = await plugin_manager.install_plugin(plugin_path)
        return {"status": "installed", "name": plugin_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/plugins/{plugin_name}/load")
async def load_plugin(plugin_name: str, config: Optional[Dict] = None):
    """Load and initialize a plugin."""
    try:
        await plugin_manager.load_plugin(plugin_name, config)
        return {"status": "loaded", "name": plugin_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/plugins/{plugin_name}/execute")
async def execute_plugin(
    plugin_name: str,
    action: str,
    params: Dict[str, Any]
):
    """Execute a plugin action."""
    try:
        result = await plugin_manager.execute_plugin(plugin_name, action, params)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/plugins")
async def list_plugins(include_disabled: bool = False):
    """List all plugins."""
    return {"plugins": plugin_manager.list_plugins(include_disabled)}


@app.post("/plugins/{plugin_name}/enable")
async def enable_plugin(plugin_name: str):
    """Enable a plugin."""
    plugin_manager.enable_plugin(plugin_name)
    return {"status": "enabled", "name": plugin_name}


@app.post("/plugins/{plugin_name}/disable")
async def disable_plugin(plugin_name: str):
    """Disable a plugin."""
    plugin_manager.disable_plugin(plugin_name)
    return {"status": "disabled", "name": plugin_name}


@app.get("/plugins/{plugin_name}/stats")
async def get_plugin_stats(plugin_name: str):
    """Get plugin statistics."""
    return plugin_manager.get_plugin_stats(plugin_name)


# ══════════════════════════════════════════════════════════════════════════════
# COLLABORATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/workspaces/create")
async def create_workspace(
    name: str,
    owner_id: str,
    description: str = ""
):
    """Create a new team workspace."""
    workspace_id = collaboration_manager.create_workspace(name, owner_id, description)
    return {"status": "created", "workspace_id": workspace_id, "name": name}


@app.post("/workspaces/{workspace_id}/members")
async def add_workspace_member(
    workspace_id: str,
    user_id: str,
    role: str = "member",
    added_by: str = None
):
    """Add a member to workspace."""
    from core.collaboration.collaboration_manager import Role
    collaboration_manager.add_member(workspace_id, user_id, Role(role), added_by)
    return {"status": "added", "user_id": user_id, "role": role}


@app.delete("/workspaces/{workspace_id}/members/{user_id}")
async def remove_workspace_member(
    workspace_id: str,
    user_id: str,
    removed_by: str
):
    """Remove a member from workspace."""
    collaboration_manager.remove_member(workspace_id, user_id, removed_by)
    return {"status": "removed", "user_id": user_id}


@app.get("/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str):
    """Get workspace details."""
    workspace = collaboration_manager.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@app.get("/users/{user_id}/workspaces")
async def list_user_workspaces(user_id: str):
    """List user's workspaces."""
    return {"workspaces": collaboration_manager.list_user_workspaces(user_id)}


@app.post("/workspaces/{workspace_id}/data")
async def set_shared_data(
    workspace_id: str,
    key: str,
    value: Any,
    user_id: str
):
    """Store shared data in workspace."""
    collaboration_manager.set_shared_data(workspace_id, key, value, user_id)
    return {"status": "stored", "key": key}


@app.get("/workspaces/{workspace_id}/data/{key}")
async def get_shared_data(workspace_id: str, key: str):
    """Get shared data from workspace."""
    data = collaboration_manager.get_shared_data(workspace_id, key)
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
    return {"key": key, "value": data}


@app.get("/workspaces/{workspace_id}/activity")
async def get_workspace_activity(workspace_id: str, limit: int = 50):
    """Get workspace activity feed."""
    activities = collaboration_manager.get_activity_feed(workspace_id, limit)
    return {"activities": [a.__dict__ for a in activities]}


# ══════════════════════════════════════════════════════════════════════════════
# VOICE INTERFACE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/voice/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    provider: Optional[str] = None,
    language: Optional[str] = None
):
    """Transcribe audio to text."""
    audio_data = await audio.read()
    
    from core.voice.voice_manager import VoiceProvider
    provider_enum = VoiceProvider(provider) if provider else None
    
    result = await voice_manager.stt.transcribe(audio_data, provider_enum, language)
    
    return {
        "text": result.text,
        "language": result.language,
        "confidence": result.confidence,
        "duration_ms": result.duration_ms,
        "provider": result.provider
    }


@app.post("/voice/synthesize")
async def synthesize_speech(
    text: str,
    provider: Optional[str] = None,
    voice: Optional[str] = None,
    speed: float = 1.0
):
    """Synthesize text to speech."""
    from core.voice.voice_manager import VoiceProvider
    provider_enum = VoiceProvider(provider) if provider else None
    
    result = await voice_manager.tts.synthesize(text, provider_enum, voice, speed)
    
    return {
        "audio_data": result.audio_data.decode('latin1'),  # For JSON serialization
        "format": result.format,
        "duration_ms": result.duration_ms,
        "provider": result.provider,
        "voice_id": result.voice_id
    }


@app.post("/voice/conversation")
async def voice_conversation(
    audio: UploadFile = File(...),
    language: Optional[str] = None
):
    """
    Process complete voice conversation.
    Transcribe -> AI Process -> Synthesize response.
    """
    audio_data = await audio.read()
    result = await voice_manager.process_voice_input(audio_data, language)
    
    return result


logger.info("✅ All Phase 3 & 4 endpoints registered")
