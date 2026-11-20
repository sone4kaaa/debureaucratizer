from typing import Any


class RUNGException(Exception):
    """Базовое исключение приложения; разворачивается в JSON-ответ."""

    def __init__(
        self,
        message: str,
        code: str = "internal_error",
        status_code: int = 500,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {"detail": self.message, "code": self.code, "context": self.context}


class ConfigurationError(RUNGException):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, code="configuration_error", status_code=500, **kw)


class UnknownEvalModeError(RUNGException):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, code="unknown_eval_mode", status_code=400, **kw)


class EngineError(RUNGException):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, code="engine_error", status_code=502, **kw)


class EngineUnavailableError(EngineError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.code = "engine_unavailable"
        self.status_code = 503


class EngineFallbackTriggeredError(EngineError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.code = "engine_fallback_triggered"
        self.status_code = 500


class ModelNotLoadedError(EngineError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.code = "model_not_loaded"
        self.status_code = 503


class RetrievalError(RUNGException):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, code="retrieval_error", status_code=502, **kw)


class KnowledgeBaseError(RUNGException):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, code="knowledge_base_error", status_code=500, **kw)


class HITLError(RUNGException):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, code="hitl_error", status_code=500, **kw)


class UncertainExampleNotFoundError(HITLError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.code = "uncertain_example_not_found"
        self.status_code = 404


class WorkerError(RUNGException):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, code="worker_error", status_code=500, **kw)
