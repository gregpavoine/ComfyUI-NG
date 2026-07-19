from __future__ import annotations

import asyncio

import pytest

from comfyng.scheduler.cancellation import (
    CancellationCheckpoint,
    CancellationRequested,
    CancellationToken,
)


def test_cancellation_is_idempotent_and_first_reason_wins() -> None:
    token = CancellationToken()

    assert token.cancel("user request") is True
    assert token.cancel("second reason") is False
    assert token.cancelled
    assert token.reason == "user request"


@pytest.mark.parametrize("checkpoint", tuple(CancellationCheckpoint))
def test_every_required_checkpoint_observes_cancellation(
    checkpoint: CancellationCheckpoint,
) -> None:
    token = CancellationToken()
    token.cancel("test")

    with pytest.raises(CancellationRequested) as error:
        token.checkpoint(checkpoint)

    assert error.value.checkpoint is checkpoint
    assert error.value.reason == "test"


def test_wait_is_async_and_checkpoint_history_is_bounded() -> None:
    async def scenario() -> None:
        token = CancellationToken(history_limit=2)
        token.checkpoint(CancellationCheckpoint.SAMPLER_STEP, position=1)
        token.checkpoint(CancellationCheckpoint.SAMPLER_STEP, position=2)
        token.checkpoint(CancellationCheckpoint.BETWEEN_BLOCKS)
        waiter = asyncio.create_task(token.wait())
        await asyncio.sleep(0)
        token.cancel("stop")
        assert await waiter == "stop"
        assert tuple(item.position for item in token.history) == (2, None)

    asyncio.run(scenario())
