# autoregressive/__init__.py
"""
Lightweight package init for the autoregressive module.

Avoid importing submodules at import time so the orchestrator can import
`autoregressive.sample` directly to discover `synth`.
"""

__all__ = []
__version__ = "0.1.0"
