from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)

    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

 
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:
            pass

    return {
        "__watcherml_unserializable__": True,
        "type": type(value).__qualname__,
        "repr": repr(value)[:500],
    }


def dumps(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False)