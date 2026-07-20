import asyncio
from comfyng.core.jobs import JobSubmission, InMemoryJobRepository
from comfyng.resources.broker import ResourceBroker
from comfyng.resources.hardware import probe_hardware
from comfyng.events.bus import EventBus
from comfyng.events.journal import InMemoryEventJournal
from comfyng.core.cache import InMemoryNodeResultCache
from comfyng.scheduler.retry import RetryPolicy
from comfyng.scheduler.scheduler import Scheduler
from comfyng.api.dispatcher import WorkflowDispatcher

async def main():
    try:
        sub = JobSubmission(
            job_id="job-123",
            queue="normal",
            user_priority=80,
            payload={
                "name": "Test",
                "prompt": "Test prompt",
                "seed": 42,
                "steps": 25,
                "width": 1024,
                "height": 1024,
                "model_name": "flux1-dev.safetensors"
            },
            workflow_id="workflow-1",
            workflow_version_id=1,
        )
        print("JobSubmission instantiated successfully!")
        
        inventory = probe_hardware()
        broker = ResourceBroker(inventory=inventory)
        journal = InMemoryEventJournal()
        events = EventBus(journal)
        repository = InMemoryJobRepository()
        cache = InMemoryNodeResultCache()
        dispatcher = WorkflowDispatcher(artifacts_dir="/tmp")
        
        scheduler = Scheduler(
            repository=repository,
            events=events,
            cache=cache,
            broker=broker,
            dispatcher=dispatcher,
            retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=1),
        )
        
        record = await scheduler.submit(sub)
        print("Scheduler submit successful, record:", record)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
