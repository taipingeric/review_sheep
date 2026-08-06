"""Shared error containment for tools.

A tool that raises kills the agent run, so every tool returns its failures as a
JSON payload instead and lets the model explain what went wrong.
"""

import inspect
import json
from functools import wraps

from github import GithubException


def _repo_argument(func, args, kwargs) -> str:
    """Read the `repo` argument whether it was passed positionally or by name."""
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
    except TypeError:
        return ""
    value = bound.arguments.get("repo", "")
    return value if isinstance(value, str) else ""


def errors_as_data(*, repo_resolver=None):
    """Wrap a tool so failures come back as JSON rather than raising.

    Args:
        repo_resolver: Optional callable turning the tool's `repo` argument into
            the repository actually queried, so the error names the right one.
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except GithubException as error:
                message = (
                    error.data.get("message", str(error))
                    if isinstance(error.data, dict)
                    else str(error)
                )
                payload = {"error": message, "status": error.status}
                if repo_resolver is not None:
                    try:
                        payload["repo"] = repo_resolver(
                            _repo_argument(func, args, kwargs)
                        )
                    except Exception:
                        # Naming the repo is a nicety; never fail while reporting.
                        pass
                return json.dumps(payload)
            except RuntimeError as error:
                # Missing token or model config: not an API failure.
                return json.dumps({"error": str(error)})
            except Exception as error:
                return json.dumps({"error": f"{type(error).__name__}: {error}"})

        return wrapper

    return decorate
