from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import fcntl
from pathlib import Path
import re
import threading


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class InvalidDigest(ValueError):
    """Raised when a value cannot safely identify a SHA-256 lock."""


def validate_digest(digest: str) -> str:
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise InvalidDigest("digest must be a lowercase 64-character SHA-256")
    return digest


class DigestLockPool:
    """Thread- and process-safe advisory locks keyed by a SHA-256 digest."""

    _registry_guard = threading.Lock()
    _thread_locks: dict[Path, threading.RLock] = {}
    _local = threading.local()

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _thread_lock(cls, path: Path) -> threading.RLock:
        with cls._registry_guard:
            return cls._thread_locks.setdefault(path, threading.RLock())

    @contextmanager
    def acquire(self, digest: str) -> Iterator[None]:
        digest = validate_digest(digest)
        path = self.root / f"{digest}.lock"
        thread_lock = self._thread_lock(path)
        with thread_lock:
            held = getattr(self._local, "held", None)
            if held is None:
                held = {}
                self._local.held = held
            state = held.get(path)
            if state is not None:
                stream, depth = state
                held[path] = (stream, depth + 1)
                try:
                    yield
                finally:
                    held[path] = (stream, depth)
                return

            stream = path.open("a+b")
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            held[path] = (stream, 1)
            try:
                yield
            finally:
                held.pop(path, None)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                stream.close()

    @contextmanager
    def acquire_many(self, digests: Iterable[str]) -> Iterator[None]:
        ordered = tuple(sorted(validate_digest(digest) for digest in digests))
        contexts = []
        try:
            for digest in ordered:
                context = self.acquire(digest)
                context.__enter__()
                contexts.append(context)
            yield
        finally:
            for context in reversed(contexts):
                context.__exit__(None, None, None)
