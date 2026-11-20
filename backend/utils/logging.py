import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_QUIET_LOGGERS = (
    "sqlalchemy.engine",
    "httpcore",
    "httpx",
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
)


class CustomFormatter(logging.Formatter):
    """Цветной вывод уровня, только когда stdout — TTY."""

    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def __init__(self, fmt: str = LOG_FORMAT, use_color: bool | None = None) -> None:
        super().__init__(fmt)
        self.use_color = sys.stdout.isatty() if use_color is None else use_color

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            color = self._COLORS.get(record.levelname, "")
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = f"{color}{record.levelname}{self._RESET}"
        return super().format(record)


def setup_logging(log_json: bool = False, level: str = "INFO") -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    date_tag = datetime.now().strftime("%Y%m%d")
    main_file = LOG_DIR / f"rung_{date_tag}.log"
    error_file = LOG_DIR / f"errors_{date_tag}.log"

    formatter: dict = {
        "()": "pythonjsonlogger.jsonlogger.JsonFormatter" if log_json else CustomFormatter,
    }
    if not log_json:
        formatter["format"] = LOG_FORMAT

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"main": formatter},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "main",
                    "level": level,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "main",
                    "filename": str(main_file),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 10,
                    "encoding": "utf-8",
                    "level": level,
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "main",
                    "filename": str(error_file),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "encoding": "utf-8",
                    "level": "ERROR",
                },
            },
            "root": {
                "handlers": ["console", "file", "error_file"],
                "level": level,
            },
        }
    )

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("rung")


from backend.core.config import get_settings  # noqa: E402

logger = setup_logging(log_json=get_settings().LOG_JSON)
