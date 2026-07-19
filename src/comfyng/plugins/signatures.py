from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import hmac
import re
from types import MappingProxyType
from typing import Protocol


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SignatureVerificationError(ValueError):
    pass


class SignatureVerifier(Protocol):
    def verify(self, publisher: str, digest: str, signature: str) -> None: ...


class HMACSignatureVerifier:
    """Detached SHA-256 HMAC signatures for configured trusted publishers."""

    def __init__(self, publisher_keys: Mapping[str, bytes]) -> None:
        keys: dict[str, bytes] = {}
        for publisher, key in publisher_keys.items():
            if not isinstance(publisher, str) or not publisher:
                raise ValueError("signature publisher names must be non-empty")
            if not isinstance(key, bytes) or len(key) < 16:
                raise ValueError("signature keys must contain at least 16 bytes")
            keys[publisher] = bytes(key)
        self._keys = MappingProxyType(keys)

    @staticmethod
    def _validate_digest(digest: str) -> None:
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise SignatureVerificationError("signed digest must be lowercase SHA-256")

    def sign(self, publisher: str, digest: str) -> str:
        self._validate_digest(digest)
        try:
            key = self._keys[publisher]
        except KeyError as exc:
            raise SignatureVerificationError(
                f"publisher {publisher!r} is not trusted"
            ) from exc
        return hmac.new(key, digest.encode("ascii"), sha256).hexdigest()

    def verify(self, publisher: str, digest: str, signature: str) -> None:
        self._validate_digest(digest)
        if not isinstance(signature, str) or _SHA256.fullmatch(signature) is None:
            raise SignatureVerificationError("signature must be lowercase SHA-256")
        expected = self.sign(publisher, digest)
        if not hmac.compare_digest(expected, signature):
            raise SignatureVerificationError("plugin signature verification failed")


__all__ = [
    "HMACSignatureVerifier",
    "SignatureVerificationError",
    "SignatureVerifier",
]
