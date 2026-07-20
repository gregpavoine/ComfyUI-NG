import asyncio
from comfyng.api.app import create_app
from comfyng.config import Settings

async def test():
    settings = Settings.load()
    app = create_app(settings)
    
    # Simulate ASGI lifespan startup
    startup_complete = asyncio.Event()
    
    async def receive():
        return {"type": "lifespan.startup"}
        
    async def send(message):
        print("ASGI send message:", message)
        if message["type"] == "lifespan.startup.complete":
            startup_complete.set()

    print("Calling app lifespan...")
    try:
        task = asyncio.create_task(app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send))
        await asyncio.wait_for(startup_complete.wait(), timeout=2.0)
        print("Lifespan startup completed successfully!")
        print("app.state.scheduler:", getattr(app.state, "scheduler", None))
    except Exception as e:
        print("Error during lifespan startup:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
