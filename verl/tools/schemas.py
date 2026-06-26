# Stub: tool schemas (placeholder for open-source)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OpenAIFunctionToolCall:
    id: str = ""
    type: str = "function"
    function: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenAIFunctionToolSchema:
    type: str = "function"
    function: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenAIFunctionCallSchema:
    name: str = ""
    arguments: str = ""


@dataclass
class OpenAIFunctionParsedSchema:
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResponse:
    tool_call_id: str = ""
    content: str = ""
    role: str = "tool"
