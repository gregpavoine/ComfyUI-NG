from __future__ import annotations

from pathlib import Path

import pytest

from comfyng.plugins.permissions import (
    PermissionDenied,
    PermissionGuard,
    PermissionSet,
)


def test_permission_set_parses_the_complete_v1_contract_immutably() -> None:
    source = {
        "network": False,
        "filesystem_read": ["models", "input"],
        "filesystem_write": ["output", "temp"],
        "subprocess": False,
        "gpu": True,
        "camera": False,
        "microphone": False,
    }

    permissions = PermissionSet.from_mapping(source)
    source["network"] = True
    source["filesystem_read"].append("secrets")  # type: ignore[union-attr]

    assert permissions.network is False
    assert permissions.filesystem_read == ("input", "models")
    assert permissions.filesystem_write == ("output", "temp")
    assert permissions.gpu is True
    assert permissions.to_mapping() == {
        "network": False,
        "filesystem_read": ("input", "models"),
        "filesystem_write": ("output", "temp"),
        "subprocess": False,
        "gpu": True,
        "camera": False,
        "microphone": False,
    }


@pytest.mark.parametrize(
    "payload",
    (
        {"unknown": True},
        {"network": 1},
        {"filesystem_read": "models"},
        {"filesystem_write": ["../escape"]},
        {"filesystem_read": ["models", "models"]},
    ),
)
def test_permission_set_rejects_unknown_or_unsafe_values(
    payload: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        PermissionSet.from_mapping(payload)


def test_permission_guard_enforces_capabilities_and_contained_paths(
    tmp_path: Path,
) -> None:
    roots = {
        "models": tmp_path / "models",
        "input": tmp_path / "input",
        "output": tmp_path / "output",
        "temp": tmp_path / "temp",
    }
    for root in roots.values():
        root.mkdir()
    readable = roots["models"] / "model.bin"
    readable.write_bytes(b"model")
    guard = PermissionGuard(
        PermissionSet.from_mapping(
            {
                "filesystem_read": ["models", "input"],
                "filesystem_write": ["output", "temp"],
                "gpu": True,
            }
        ),
        roots,
    )

    assert guard.require_read(readable) == readable.resolve()
    assert guard.require_write(roots["output"] / "nested" / "image.png").is_relative_to(
        roots["output"].resolve()
    )
    guard.require_gpu()

    with pytest.raises(PermissionDenied, match="network"):
        guard.require_network()
    with pytest.raises(PermissionDenied, match="subprocess"):
        guard.require_subprocess()
    with pytest.raises(PermissionDenied, match="filesystem_read"):
        guard.require_read(tmp_path / "secret.txt")
    with pytest.raises(PermissionDenied, match="filesystem_write"):
        guard.require_write(tmp_path / "escape.txt")
    with pytest.raises(PermissionDenied, match="camera"):
        guard.require_camera()
    with pytest.raises(PermissionDenied, match="microphone"):
        guard.require_microphone()


def test_permission_guard_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "escape").symlink_to(outside, target_is_directory=True)
    guard = PermissionGuard(
        PermissionSet.from_mapping({"filesystem_write": ["output"]}),
        {"output": allowed},
    )

    with pytest.raises(PermissionDenied, match="filesystem_write"):
        guard.require_write(allowed / "escape" / "payload.bin")
