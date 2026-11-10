"""API package for serving guardrailed endpoints to benchmarking tools like Garak."""

from .app import app, create_app

__all__ = ["app", "create_app"]
