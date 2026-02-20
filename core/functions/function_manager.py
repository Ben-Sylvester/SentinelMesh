"""
Enhanced Function Calling System

OpenAI-compatible function calling with JSON schema validation,
automatic parameter extraction, and error handling.
"""

import json
import logging
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ParameterType(str, Enum):
    """JSON Schema types for function parameters."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


@dataclass
class FunctionParameter:
    """Function parameter definition."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None


@dataclass
class FunctionDefinition:
    """Complete function definition for LLM."""
    name: str
    description: str
    parameters: List[FunctionParameter]
    returns: str
    function: Callable


@dataclass
class FunctionCall:
    """Parsed function call from LLM."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class FunctionResult:
    """Result of function execution."""
    name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


class FunctionRegistry:
    """
    Registry for callable functions with schema generation.
    """
    
    def __init__(self):
        self.functions: Dict[str, FunctionDefinition] = {}
    
    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Dict]] = None
    ):
        """
        Decorator to register a function.
        
        Usage:
        @registry.register(
            name="get_weather",
            description="Get current weather",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            }
        )
        async def get_weather(location: str) -> dict:
            return {"temp": 72, "conditions": "sunny"}
        """
        def decorator(func: Callable):
            func_name = name or func.__name__
            func_description = description or (func.__doc__ or "").strip()
            
            # Auto-generate parameters from function signature if not provided
            if parameters is None:
                params = self._extract_parameters(func)
            else:
                params = self._parse_parameters(parameters)
            
            # Get return type description
            return_desc = self._extract_return_description(func)
            
            self.functions[func_name] = FunctionDefinition(
                name=func_name,
                description=func_description,
                parameters=params,
                returns=return_desc,
                function=func
            )
            
            logger.info(f"Registered function: {func_name}")
            return func
        
        return decorator
    
    def _extract_parameters(self, func: Callable) -> List[FunctionParameter]:
        """Extract parameters from function signature."""
        sig = inspect.signature(func)
        params = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'cls']:
                continue
            
            # Infer type from annotation
            param_type = ParameterType.STRING  # Default
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = ParameterType.INTEGER
                elif param.annotation == float:
                    param_type = ParameterType.NUMBER
                elif param.annotation == bool:
                    param_type = ParameterType.BOOLEAN
                elif param.annotation == dict:
                    param_type = ParameterType.OBJECT
                elif param.annotation == list:
                    param_type = ParameterType.ARRAY
            
            params.append(FunctionParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter {param_name}",
                required=param.default == inspect.Parameter.empty
            ))
        
        return params
    
    def _parse_parameters(self, params_dict: Dict) -> List[FunctionParameter]:
        """Parse parameters from dictionary definition."""
        params = []
        for name, spec in params_dict.items():
            params.append(FunctionParameter(
                name=name,
                type=ParameterType(spec.get("type", "string")),
                description=spec.get("description", ""),
                required=spec.get("required", True),
                enum=spec.get("enum"),
                default=spec.get("default")
            ))
        return params
    
    def _extract_return_description(self, func: Callable) -> str:
        """Extract return type description."""
        sig = inspect.signature(func)
        if sig.return_annotation != inspect.Signature.empty:
            return str(sig.return_annotation)
        return "any"
    
    def get_function(self, name: str) -> Optional[FunctionDefinition]:
        """Get function definition by name."""
        return self.functions.get(name)
    
    def list_functions(self) -> List[str]:
        """List all registered function names."""
        return list(self.functions.keys())
    
    def to_openai_format(self) -> List[Dict]:
        """
        Convert registered functions to OpenAI function calling format.
        
        Returns:
            List of function definitions compatible with OpenAI API
        """
        functions = []
        for func_def in self.functions.values():
            # Build parameters schema
            properties = {}
            required = []
            
            for param in func_def.parameters:
                param_schema = {
                    "type": param.type.value,
                    "description": param.description
                }
                
                if param.enum:
                    param_schema["enum"] = param.enum
                
                properties[param.name] = param_schema
                
                if param.required:
                    required.append(param.name)
            
            functions.append({
                "name": func_def.name,
                "description": func_def.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            })
        
        return functions


class FunctionExecutor:
    """
    Executes function calls with validation and error handling.
    """
    
    def __init__(self, registry: FunctionRegistry):
        self.registry = registry
    
    def parse_function_call(self, llm_output: str) -> Optional[FunctionCall]:
        """
        Parse function call from LLM output.
        
        Expected formats:
        - JSON: {"name": "func_name", "arguments": {...}}
        - Text: func_name(arg1="value1", arg2="value2")
        """
        try:
            # Try JSON format first
            data = json.loads(llm_output)
            if "name" in data and "arguments" in data:
                return FunctionCall(
                    name=data["name"],
                    arguments=data["arguments"]
                )
        except json.JSONDecodeError:
            # Try parsing text format
            import re
            match = re.match(r'(\w+)\((.*)\)', llm_output)
            if match:
                func_name = match.group(1)
                args_str = match.group(2)
                
                # Parse arguments
                arguments = {}
                for arg in args_str.split(','):
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        arguments[key.strip()] = value.strip().strip('"\'')
                
                return FunctionCall(name=func_name, arguments=arguments)
        
        return None
    
    def validate_parameters(
        self,
        func_def: FunctionDefinition,
        arguments: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate function parameters."""
        # Check required parameters
        for param in func_def.parameters:
            if param.required and param.name not in arguments:
                return False, f"Missing required parameter: {param.name}"
        
        # Validate types
        for param_name, param_value in arguments.items():
            param_def = next(
                (p for p in func_def.parameters if p.name == param_name),
                None
            )
            if not param_def:
                logger.warning(f"Unknown parameter: {param_name}")
                continue
            
            # Type checking
            if param_def.type == ParameterType.INTEGER:
                if not isinstance(param_value, int):
                    try:
                        arguments[param_name] = int(param_value)
                    except:
                        return False, f"Parameter {param_name} must be integer"
            
            elif param_def.type == ParameterType.NUMBER:
                if not isinstance(param_value, (int, float)):
                    try:
                        arguments[param_name] = float(param_value)
                    except:
                        return False, f"Parameter {param_name} must be number"
            
            elif param_def.type == ParameterType.BOOLEAN:
                if not isinstance(param_value, bool):
                    if param_value in ["true", "True", "1"]:
                        arguments[param_name] = True
                    elif param_value in ["false", "False", "0"]:
                        arguments[param_name] = False
                    else:
                        return False, f"Parameter {param_name} must be boolean"
        
        return True, None
    
    async def execute(self, function_call: FunctionCall) -> FunctionResult:
        """
        Execute a function call with validation and error handling.
        """
        import time
        start = time.time()
        
        # Get function definition
        func_def = self.registry.get_function(function_call.name)
        if not func_def:
            return FunctionResult(
                name=function_call.name,
                success=False,
                result=None,
                error=f"Function '{function_call.name}' not found"
            )
        
        # Validate parameters
        valid, error = self.validate_parameters(func_def, function_call.arguments)
        if not valid:
            return FunctionResult(
                name=function_call.name,
                success=False,
                result=None,
                error=error
            )
        
        # Execute function
        try:
            func = func_def.function
            
            # Handle both sync and async functions
            if inspect.iscoroutinefunction(func):
                result = await func(**function_call.arguments)
            else:
                result = func(**function_call.arguments)
            
            execution_time = int((time.time() - start) * 1000)
            
            return FunctionResult(
                name=function_call.name,
                success=True,
                result=result,
                execution_time_ms=execution_time
            )
        
        except Exception as e:
            logger.error(f"Function execution error: {e}", exc_info=True)
            return FunctionResult(
                name=function_call.name,
                success=False,
                result=None,
                error=str(e)
            )


class FunctionCallingRouter:
    """
    Router wrapper that handles function calling workflow.
    """
    
    def __init__(self, router, registry: FunctionRegistry, max_iterations: int = 5):
        self.router = router
        self.registry = registry
        self.executor = FunctionExecutor(registry)
        self.max_iterations = max_iterations
    
    async def route_with_functions(
        self,
        prompt: str,
        available_functions: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Route request with function calling support.
        
        Workflow:
        1. Send prompt with available functions
        2. Parse LLM response for function calls
        3. Execute functions
        4. Send results back to LLM
        5. Repeat until LLM returns final answer
        """
        # Get available functions
        if available_functions:
            funcs = [self.registry.get_function(name) for name in available_functions]
            funcs = [f for f in funcs if f]
        else:
            funcs = list(self.registry.functions.values())
        
        # Add function definitions to prompt
        func_descriptions = "\n".join([
            f"- {f.name}: {f.description}"
            for f in funcs
        ])
        
        augmented_prompt = f"""{prompt}

Available functions:
{func_descriptions}

To call a function, respond with:
{{"name": "function_name", "arguments": {{"param": "value"}}}}
"""
        
        conversation_history = []
        
        for iteration in range(self.max_iterations):
            # Call router
            result = await self.router.route(augmented_prompt, **kwargs)
            
            conversation_history.append({
                "role": "assistant",
                "content": result.output
            })
            
            # Check for function call
            function_call = self.executor.parse_function_call(result.output or "")
            
            if not function_call:
                # No function call - return final answer
                return result
            
            # Execute function
            func_result = await self.executor.execute(function_call)
            
            conversation_history.append({
                "role": "function",
                "name": func_result.name,
                "content": json.dumps({
                    "success": func_result.success,
                    "result": func_result.result,
                    "error": func_result.error
                })
            })
            
            # Continue conversation with function result
            augmented_prompt = f"""{prompt}

Previous interactions:
{json.dumps(conversation_history, indent=2)}

Continue based on the function result above."""
        
        # Max iterations reached
        return result
