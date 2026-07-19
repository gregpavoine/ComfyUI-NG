from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import unquote, urlparse

from fastapi import FastAPI


logger = logging.getLogger("comfyng.api.server")


class ASGIHTTPBridge:
    """Lightweight pure-Python asyncio HTTP/1.1 to ASGI 3.0 bridge."""

    def __init__(self, app: Any, host: str, port: int) -> None:
        self.app = app
        self.host = host
        self.port = port

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line_bytes = await reader.readline()
            if not request_line_bytes:
                writer.close()
                await writer.wait_closed()
                return

            request_line = request_line_bytes.decode("latin-1").strip()
            parts = request_line.split(" ")
            if len(parts) < 2:
                writer.close()
                await writer.wait_closed()
                return

            method = parts[0].upper()
            target = parts[1]
            http_version = parts[2] if len(parts) > 2 else "HTTP/1.1"

            # Parse headers
            headers: list[tuple[bytes, bytes]] = []
            content_length = 0
            while True:
                line_bytes = await reader.readline()
                if not line_bytes or line_bytes in (b"\r\n", b"\n"):
                    break
                line = line_bytes.decode("latin-1").strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    key_lower = k.strip().lower().encode("latin-1")
                    val_bytes = v.strip().encode("latin-1")
                    headers.append((key_lower, val_bytes))
                    if key_lower == b"content-length":
                        try:
                            content_length = int(val_bytes)
                        except ValueError:
                            pass

            body_data = b""
            if content_length > 0:
                body_data = await reader.readexactly(content_length)

            # Parse URL path and query string
            parsed_url = urlparse(target)
            raw_path = parsed_url.path or "/"
            path = unquote(raw_path)
            query_string = parsed_url.query.encode("ascii")

            peer = writer.get_extra_info("peername")
            client = (peer[0], peer[1]) if peer and isinstance(peer, tuple) else ("127.0.0.1", 0)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": http_version.replace("HTTP/", ""),
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": raw_path.encode("ascii"),
                "query_string": query_string,
                "root_path": "",
                "headers": headers,
                "client": client,
                "server": (self.host, self.port),
                "state": {},
            }

            body_sent = False

            async def receive() -> dict[str, Any]:
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {
                        "type": "http.request",
                        "body": body_data,
                        "more_body": False,
                    }
                # Keep pending for disconnect if ASGI app calls receive again
                return await asyncio.Future()

            response_started = False

            async def send(message: dict[str, Any]) -> None:
                nonlocal response_started
                msg_type = message.get("type")
                if msg_type == "http.response.start":
                    status_code = message.get("status", 200)
                    resp_headers = message.get("headers", [])
                    status_line = f"HTTP/1.1 {status_code} OK\r\n"
                    writer.write(status_line.encode("latin-1"))
                    for hk, hv in resp_headers:
                        header_line = f"{hk.decode('latin-1')}: {hv.decode('latin-1')}\r\n"
                        writer.write(header_line.encode("latin-1"))
                    writer.write(b"\r\n")
                    await writer.drain()
                    response_started = True
                elif msg_type == "http.response.body":
                    body_chunk = message.get("body", b"")
                    if body_chunk:
                        writer.write(body_chunk)
                        await writer.drain()

            await self.app(scope, receive, send)
        except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            logger.exception("Error handling HTTP request: %s", exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


async def _serve_async(app: FastAPI, host: str, port: int) -> None:
    bridge = ASGIHTTPBridge(app, host, port)
    server = await asyncio.start_server(bridge.handle_client, host, port)
    async with server:
        await server.serve_forever()


def run_server(app: FastAPI, host: str = "127.0.0.1", port: int = 8188) -> None:
    """Run the FastAPI application server on host:port."""

    try:
        import uvicorn  # type: ignore[import-not-found]
        uvicorn.run(app, host=host, port=port, log_level="info")
        return
    except ImportError:
        pass

    try:
        asyncio.run(_serve_async(app, host, port))
    except KeyboardInterrupt:
        pass
