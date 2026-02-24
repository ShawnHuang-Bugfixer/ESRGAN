import json
import logging
from typing import Any, Dict


def setup_logging(level: str = 'INFO') -> None:
    normalized = (level or 'INFO').strip().upper()
    resolved_level = getattr(logging, normalized, logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )


def format_log_fields(fields: Dict[str, Any]) -> str:
    normalized = {key: value for key, value in fields.items() if value is not None}
    return json.dumps(normalized, ensure_ascii=False, separators=(',', ':'))
