"""
Workflow Engine

DAG-based workflow automation with conditional branching, error handling,
and scheduled execution. Enterprise-grade automation for AI workflows.
"""

import json
import yaml
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class WorkflowStep:
    """Single step in a workflow."""
    name: str
    function: str
    params: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Python expression
    retry_count: int = 3
    retry_delay: int = 5
    timeout: Optional[int] = None


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    name: str
    description: str
    steps: List[WorkflowStep]
    schedule: Optional[str] = None  # Cron expression
    timeout: Optional[int] = None
    on_failure: Optional[str] = None  # "stop", "continue", "retry"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepExecution:
    """Execution record for a step."""
    step_name: str
    status: StepStatus
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class WorkflowExecution:
    """Execution record for a workflow."""
    id: str
    workflow_name: str
    status: WorkflowStatus
    started_at: float
    completed_at: Optional[float] = None
    steps: Dict[str, StepExecution] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class WorkflowEngine:
    """
    DAG-based workflow execution engine.
    Supports conditional branching, error handling, and scheduling.
    """
    
    def __init__(self, storage_path: str = "data/workflows"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "workflows.db"
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.function_registry: Dict[str, Callable] = {}
        
        self._init_db()
        self._register_builtin_functions()
    
    def _init_db(self):
        """Initialize workflow database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_definitions (
                name            TEXT PRIMARY KEY,
                definition      TEXT NOT NULL,
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_executions (
                id              TEXT PRIMARY KEY,
                workflow_name   TEXT NOT NULL,
                status          TEXT NOT NULL,
                started_at      REAL NOT NULL,
                completed_at    REAL,
                execution_data  TEXT NOT NULL,
                error           TEXT,
                
                INDEX idx_workflow (workflow_name),
                INDEX idx_status (status),
                INDEX idx_started (started_at DESC)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_workflows (
                workflow_name   TEXT PRIMARY KEY,
                schedule        TEXT NOT NULL,
                last_run        REAL,
                next_run        REAL,
                enabled         INTEGER DEFAULT 1,
                
                FOREIGN KEY (workflow_name) REFERENCES workflow_definitions(name)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _register_builtin_functions(self):
        """Register built-in workflow functions."""
        self.register_function("log", self._log)
        self.register_function("sleep", self._sleep)
        self.register_function("http_request", self._http_request)
    
    async def _log(self, message: str, level: str = "info"):
        """Log a message."""
        getattr(logger, level)(f"[Workflow] {message}")
        return {"logged": True, "message": message}
    
    async def _sleep(self, seconds: int):
        """Sleep for specified seconds."""
        await asyncio.sleep(seconds)
        return {"slept": seconds}
    
    async def _http_request(self, url: str, method: str = "GET", **kwargs):
        """Make HTTP request."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as response:
                return {
                    "status": response.status,
                    "data": await response.text()
                }
    
    def register_function(self, name: str, func: Callable):
        """Register a function for use in workflows."""
        self.function_registry[name] = func
        logger.info(f"Registered workflow function: {name}")
    
    def load_workflow(self, yaml_content: str) -> WorkflowDefinition:
        """Load workflow from YAML."""
        data = yaml.safe_load(yaml_content)
        
        steps = [
            WorkflowStep(
                name=step["name"],
                function=step["function"],
                params=step.get("params", {}),
                depends_on=step.get("depends_on", []),
                condition=step.get("condition"),
                retry_count=step.get("retry_count", 3),
                retry_delay=step.get("retry_delay", 5),
                timeout=step.get("timeout")
            )
            for step in data["steps"]
        ]
        
        return WorkflowDefinition(
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            schedule=data.get("schedule"),
            timeout=data.get("timeout"),
            on_failure=data.get("on_failure", "stop"),
            metadata=data.get("metadata", {})
        )
    
    def save_workflow(self, workflow: WorkflowDefinition):
        """Save workflow definition."""
        self.workflows[workflow.name] = workflow
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        now = time.time()
        
        conn.execute("""
            INSERT OR REPLACE INTO workflow_definitions
            (name, definition, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (
            workflow.name,
            json.dumps({
                "name": workflow.name,
                "description": workflow.description,
                "steps": [
                    {
                        "name": s.name,
                        "function": s.function,
                        "params": s.params,
                        "depends_on": s.depends_on,
                        "condition": s.condition,
                        "retry_count": s.retry_count,
                        "retry_delay": s.retry_delay,
                        "timeout": s.timeout
                    }
                    for s in workflow.steps
                ],
                "schedule": workflow.schedule,
                "timeout": workflow.timeout,
                "on_failure": workflow.on_failure,
                "metadata": workflow.metadata
            }),
            now,
            now
        ))
        
        # Schedule if needed
        if workflow.schedule:
            next_run = self._calculate_next_run(workflow.schedule)
            conn.execute("""
                INSERT OR REPLACE INTO scheduled_workflows
                (workflow_name, schedule, next_run, enabled)
                VALUES (?, ?, ?, 1)
            """, (workflow.name, workflow.schedule, next_run))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved workflow: {workflow.name}")
    
    def _calculate_next_run(self, cron_expr: str) -> float:
        """Calculate next run time from cron expression."""
        # Simplified - in production, use croniter library
        return time.time() + 3600  # Default: 1 hour from now
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate step condition."""
        if not condition:
            return True
        
        try:
            # Safe evaluation with limited scope
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False
    
    def _build_dag(self, workflow: WorkflowDefinition) -> Dict[str, List[str]]:
        """Build dependency graph."""
        dag = {}
        for step in workflow.steps:
            dag[step.name] = step.depends_on
        return dag
    
    def _topological_sort(self, dag: Dict[str, List[str]]) -> List[str]:
        """Topologically sort workflow steps."""
        visited = set()
        result = []
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for dep in dag.get(node, []):
                visit(dep)
            result.append(node)
        
        for node in dag:
            visit(node)
        
        return result
    
    async def execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> StepExecution:
        """Execute a single workflow step."""
        execution = StepExecution(
            step_name=step.name,
            status=StepStatus.RUNNING,
            started_at=time.time()
        )
        
        # Check condition
        if not self._evaluate_condition(step.condition, context):
            execution.status = StepStatus.SKIPPED
            execution.completed_at = time.time()
            return execution
        
        # Execute with retry
        for attempt in range(step.retry_count):
            try:
                # Get function
                func = self.function_registry.get(step.function)
                if not func:
                    raise ValueError(f"Function '{step.function}' not found")
                
                # Render parameters with context
                params = self._render_params(step.params, context)
                
                # Execute with timeout
                if step.timeout:
                    result = await asyncio.wait_for(
                        func(**params),
                        timeout=step.timeout
                    )
                else:
                    result = await func(**params)
                
                execution.status = StepStatus.SUCCESS
                execution.result = result
                execution.completed_at = time.time()
                
                # Update context with result
                context[f"steps.{step.name}"] = result
                
                return execution
            
            except Exception as e:
                logger.error(f"Step {step.name} failed (attempt {attempt+1}): {e}")
                execution.retry_count = attempt + 1
                execution.error = str(e)
                
                if attempt < step.retry_count - 1:
                    await asyncio.sleep(step.retry_delay)
                else:
                    execution.status = StepStatus.FAILED
                    execution.completed_at = time.time()
        
        return execution
    
    def _render_params(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Render parameters with context variables."""
        rendered = {}
        for key, value in params.items():
            if isinstance(value, str) and "{{" in value:
                # Simple template rendering
                for ctx_key, ctx_value in context.items():
                    value = value.replace(f"{{{{{ctx_key}}}}}", str(ctx_value))
            rendered[key] = value
        return rendered
    
    async def execute_workflow(
        self,
        workflow_name: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """Execute a workflow."""
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        # Create execution record
        import uuid
        execution_id = str(uuid.uuid4())[:8]
        
        execution = WorkflowExecution(
            id=execution_id,
            workflow_name=workflow_name,
            status=WorkflowStatus.RUNNING,
            started_at=time.time(),
            context=input_data or {}
        )
        
        logger.info(f"Starting workflow execution: {workflow_name} ({execution_id})")
        
        try:
            # Build execution order
            dag = self._build_dag(workflow)
            execution_order = self._topological_sort(dag)
            
            # Execute steps in order
            for step_name in execution_order:
                step = next(s for s in workflow.steps if s.name == step_name)
                
                # Check dependencies
                can_execute = all(
                    execution.steps.get(dep, StepExecution("", StepStatus.PENDING)).status == StepStatus.SUCCESS
                    for dep in step.depends_on
                )
                
                if not can_execute:
                    execution.steps[step_name] = StepExecution(
                        step_name=step_name,
                        status=StepStatus.SKIPPED
                    )
                    continue
                
                # Execute step
                step_execution = await self.execute_step(step, execution.context)
                execution.steps[step_name] = step_execution
                
                # Handle failure
                if step_execution.status == StepStatus.FAILED:
                    if workflow.on_failure == "stop":
                        raise Exception(f"Step {step_name} failed: {step_execution.error}")
                    elif workflow.on_failure == "continue":
                        logger.warning(f"Step {step_name} failed but continuing")
            
            execution.status = WorkflowStatus.SUCCESS
            execution.completed_at = time.time()
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = time.time()
        
        # Save execution record
        self._save_execution(execution)
        
        return execution
    
    def _save_execution(self, execution: WorkflowExecution):
        """Save execution record to database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO workflow_executions
            (id, workflow_name, status, started_at, completed_at, execution_data, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            execution.id,
            execution.workflow_name,
            execution.status.value,
            execution.started_at,
            execution.completed_at,
            json.dumps({
                "steps": {
                    name: {
                        "status": step.status.value,
                        "started_at": step.started_at,
                        "completed_at": step.completed_at,
                        "result": step.result,
                        "error": step.error,
                        "retry_count": step.retry_count
                    }
                    for name, step in execution.steps.items()
                },
                "context": execution.context
            }),
            execution.error
        ))
        
        conn.commit()
        conn.close()
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution by ID."""
        conn = sqlite3.connect(self.db_path)
        
        row = conn.execute("""
            SELECT id, workflow_name, status, started_at, completed_at, execution_data, error
            FROM workflow_executions
            WHERE id = ?
        """, (execution_id,)).fetchone()
        
        conn.close()
        
        if not row:
            return None
        
        data = json.loads(row[5])
        
        return WorkflowExecution(
            id=row[0],
            workflow_name=row[1],
            status=WorkflowStatus(row[2]),
            started_at=row[3],
            completed_at=row[4],
            steps={
                name: StepExecution(
                    step_name=name,
                    status=StepStatus(step_data["status"]),
                    started_at=step_data["started_at"],
                    completed_at=step_data["completed_at"],
                    result=step_data["result"],
                    error=step_data["error"],
                    retry_count=step_data["retry_count"]
                )
                for name, step_data in data["steps"].items()
            },
            context=data["context"],
            error=row[6]
        )
    
    def list_executions(
        self,
        workflow_name: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List workflow executions."""
        conn = sqlite3.connect(self.db_path)
        
        if workflow_name:
            rows = conn.execute("""
                SELECT id, workflow_name, status, started_at, completed_at
                FROM workflow_executions
                WHERE workflow_name = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (workflow_name, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, workflow_name, status, started_at, completed_at
                FROM workflow_executions
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        
        conn.close()
        
        return [
            {
                "id": row[0],
                "workflow_name": row[1],
                "status": row[2],
                "started_at": row[3],
                "completed_at": row[4],
                "duration_seconds": row[4] - row[3] if row[4] else None
            }
            for row in rows
        ]
