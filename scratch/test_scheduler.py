import asyncio
import time
from comfyng.api.app import create_app
from comfyng.config import Settings
from comfyng.core.jobs import JobSubmission
from comfyng.api.app import JobSubmissionDTO

async def test():
    settings = Settings.load()
    app = create_app(settings)
    
    # Start ASGI lifespan
    startup_complete = asyncio.Event()
    async def receive():
        return {"type": "lifespan.startup"}
    async def send(message):
        if message["type"] == "lifespan.startup.complete":
            startup_complete.set()
            
    lifespan_task = asyncio.create_task(app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send))
    await startup_complete.wait()
    
    scheduler = app.state.scheduler
    print("Scheduler initialized:", scheduler)
    
    # Submit a job
    submission = JobSubmission(
        job_id="test-job-diag",
        queue="normal",
        user_priority=80,
        payload={
            "prompt": "Test cybernetic space station",
            "seed": 42,
            "steps": 5,
            "width": 1024,
            "height": 1024,
            "model_name": "flux1-dev.safetensors",
        },
        workflow_id="wf-test",
        workflow_version_id=1,
    )
    
    print("Submitting job...")
    record = await scheduler.submit(submission)
    print("Job submitted. Status:", record.status)
    
    # Poll status for 5 seconds
    for i in range(10):
        await asyncio.sleep(0.5)
        updated = await scheduler.repository.get("test-job-diag")
        print(f"[{i*0.5:.1f}s] Job status: {updated.status if updated else 'None'}")
        if updated and updated.status.terminal:
            break

if __name__ == "__main__":
    asyncio.run(test())
