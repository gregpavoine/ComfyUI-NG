import asyncio
import base64
import hashlib
import json
import os
from urllib.parse import urlparse

class SimpleWebSocketClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.parsed = urlparse(url)
        self.reader = None
        self.writer = None

    async def connect(self) -> None:
        host = self.parsed.hostname
        port = self.parsed.port or 8188
        print(f"Connecting to TCP {host}:{port}...")
        self.reader, self.writer = await asyncio.open_connection(host, port)
        
        # HTTP Sec-WebSocket-Key
        key = base64.b64encode(hashlib.sha1(os.urandom(16)).digest()).decode()
        path = self.parsed.path
        if self.parsed.query:
            path += "?" + self.parsed.query
            
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.writer.write(handshake.encode())
        await self.writer.drain()
        
        # Read response headers
        print("Reading handshake response headers...")
        while True:
            line = await self.reader.readline()
            if line == b"\r\n" or not line:
                break
        print("Connected successfully!")

    async def recv(self) -> str | None:
        try:
            header = await self.reader.readexactly(2)
            b1, b2 = header[0], header[1]
            opcode = b1 & 0x0F
            masked = b2 & 0x80
            payload_len = b2 & 0x7F
            
            if payload_len == 126:
                len_bytes = await self.reader.readexactly(2)
                payload_len = int.from_bytes(len_bytes, byteorder="big")
            elif payload_len == 127:
                len_bytes = await self.reader.readexactly(8)
                payload_len = int.from_bytes(len_bytes, byteorder="big")
                
            if masked:
                mask = await self.reader.readexactly(4)
                
            payload = await self.reader.readexactly(payload_len)
            
            if masked:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
                
            if opcode == 0x08:  # Close
                return None
            if opcode == 0x01:  # Text
                return payload.decode("utf-8")
        except Exception as e:
            print("WS recv error:", e)
            return None
        return ""

async def main():
    client = SimpleWebSocketClient("ws://127.0.0.1:8188/ws?clientId=test-client")
    await client.connect()
    while True:
        msg = await client.recv()
        if msg is None:
            print("Connection closed by server.")
            break
        if msg:
            data = json.loads(msg)
            print("Received event:", data.get("type"), data.get("data"))

if __name__ == "__main__":
    asyncio.run(main())
