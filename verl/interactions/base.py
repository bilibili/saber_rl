# Stub: base interaction class
from typing import Any, Dict, List, Optional


class BaseInteraction:
    """Base class for interactions (not used by SABER)."""

    def __init__(self, **kwargs):
        pass

    async def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError
