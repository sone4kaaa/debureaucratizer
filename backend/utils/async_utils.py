import asyncio
import functools
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable[[F], F]:
    """Retry-декоратор для async-функций с линейной задержкой."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    if attempt < max_attempts:
                        await asyncio.sleep(delay * attempt)
            raise last_error  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
