"""Dependency-light domain primitives shared by the control plane."""

from .contracts import Contract, decode_contract, register_contract
from .enums import (
    GpuRequirement,
    LifecycleState,
    LoadPolicy,
    NodeLifecycleState,
    RuntimeIsolation,
    TransferPolicy,
    UnloadPolicy,
)
from .errors import ComfyNGError

__all__ = [
    "ComfyNGError",
    "Contract",
    "GpuRequirement",
    "LifecycleState",
    "LoadPolicy",
    "NodeLifecycleState",
    "RuntimeIsolation",
    "TransferPolicy",
    "UnloadPolicy",
    "decode_contract",
    "register_contract",
]
