"""Compatibility wrapper redirecting to src.caching.query_cache."""

from src.caching.query_cache import clear_cache, get_cached, set_cached

__all__ = ["get_cached", "set_cached", "clear_cache"]
