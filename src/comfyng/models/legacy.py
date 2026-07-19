from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .detector import ArchitectureDetection


_MESSAGES = {
    "sd15": "Stable Diffusion 1.5 is not supported.",
    "sd2": "Stable Diffusion 2.x is not supported.",
    "sdxl": "Stable Diffusion XL is not supported.",
}


class UnsupportedModelGeneration(RuntimeError):
    code = "unsupported_model_generation"
    minimum_generation = "FLUX.1"

    def __init__(self, detected_family: str) -> None:
        self.detected_family = detected_family
        self.message = _MESSAGES.get(
            detected_family,
            f"Model family {detected_family} predates the supported generation.",
        )
        super().__init__(self.message)

    def to_payload(self) -> dict[str, dict[str, str]]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "minimum_generation": self.minimum_generation,
                "detected_family": self.detected_family,
            }
        }


T = TypeVar("T", bound="ArchitectureDetection")


def require_modern(detection: T) -> T:
    """Refuse legacy weights before a runtime or GPU loader can be selected."""

    if not detection.supported:
        raise UnsupportedModelGeneration(detection.family)
    return detection


__all__ = ["UnsupportedModelGeneration", "require_modern"]
