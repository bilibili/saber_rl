# Stub: base tool (placeholder for open-source)
from typing import Any, Dict


class BaseTool:
    """Base class for tools."""

    name: str = ""
    description: str = ""

    def __init__(self, **kwargs):
        pass

    def execute(self, **kwargs) -> Any:
        raise NotImplementedError
