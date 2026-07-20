from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Self

import msgspec

from comfyng.core.json_values import freeze_json_value


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 200
    ):
        raise ValueError(f"{name} must be a non-empty trimmed string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid Unicode") from exc
    return value


class EventEnvelope(msgspec.Struct, forbid_unknown_fields=True):
    sequence: int
    stream_sequence: int
    event_type: str
    emitted_at: float
    payload: Mapping[str, Any] = msgspec.field(default_factory=dict)
    job_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("sequence", "stream_sequence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        _identifier(self.event_type, "event_type")
        if "." not in self.event_type:
            raise ValueError("event_type must be namespaced with a dot")
        if (
            isinstance(self.emitted_at, bool)
            or not isinstance(self.emitted_at, (int, float))
            or not math.isfinite(self.emitted_at)
            or self.emitted_at < 0
        ):
            raise ValueError("emitted_at must be a finite non-negative number")
        if self.job_id is not None:
            _identifier(self.job_id, "job_id")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a JSON object")
        self.payload = freeze_json_value(self.payload, path="$.payload")

    def to_json(self) -> bytes:
        return msgspec.json.encode(self)

    @classmethod
    def from_json(cls, value: bytes | bytearray | memoryview | str) -> Self:
        try:
            return msgspec.json.decode(value, type=cls, strict=True)
        except (
            TypeError,
            ValueError,
            msgspec.DecodeError,
            msgspec.ValidationError,
        ) as exc:
            raise ValueError(f"invalid event envelope: {exc}") from exc


__all__ = ["EventEnvelope"]
