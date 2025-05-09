class RepositoryAuthError(Exception):
    """Raised when repository authentication fails (401/403)."""

class RepositoryNotFoundError(Exception):
    """Raised when repository is not found (404)."""
"""Base class for repository implementations."""

from typing import Any

class RepositoryBase:
    """Abstract base class for repository classes."""
    pass