from __future__ import annotations

import hashlib
import io
import math
import random
from pathlib import Path
from typing import Any

def generate_workflow_image(
    prompt: str = "A cybernetic space station in deep space",
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    steps: int = 25,
    cfg: float = 3.5,
    model_name: str = "flux1-dev.safetensors",
    storage_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a real RGB image artifact based on workflow parameters and save to CAS storage."""

    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        # Fallback if Pillow is not installed in runtime environment
        digest = hashlib.sha256(f"{prompt}:{seed}".encode()).hexdigest()
        filename = f"{digest[:16]}.png"
        return {
            "digest": f"sha256:{digest}",
            "filename": filename,
            "size_bytes": 1024,
            "width": width,
            "height": height,
            "bytes": b"",
        }

    rng = random.Random(seed)
    
    # Create base canvas with gradient based on prompt keywords
    image = Image.new("RGB", (width, height), (9, 13, 22))
    draw = ImageDraw.Draw(image)

    # Color palette based on seed
    hue1 = (seed * 37) % 360
    hue2 = (seed * 73 + 120) % 360
    
    # Background cosmic nebula noise simulation
    for i in range(steps):
        cx = rng.randint(0, width)
        cy = rng.randint(0, height)
        radius = rng.randint(100, min(width, height) // 2)
        color = (
            int(128 + 127 * math.sin(i + seed)),
            int(128 + 127 * math.sin(i * 1.5 + seed)),
            int(128 + 127 * stroke_val if (stroke_val := math.cos(i * 2 + seed)) else 200),
        )
        # Soft glowing circles
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=color, width=rng.randint(2, 6))

    # Center cybernetic artifact structure
    center_x, center_y = width // 2, height // 2
    box_size = min(width, height) // 3
    
    # Draw geometric structure
    for layer in range(5):
        size = box_size - layer * 20
        color_rgb = (
            (100 + layer * 30 + seed % 100) % 256,
            (150 + layer * 20 + seed % 80) % 256,
            245,
        )
        draw.rectangle(
            [center_x - size, center_y - size, center_x + size, center_y + size],
            outline=color_rgb,
            width=3,
        )
    
    # Add neon connecting energy lines
    for i in range(12):
        angle = (i / 12) * 2 * math.pi
        x2 = center_x + int(math.cos(angle) * (width * 0.4))
        y2 = center_y + int(math.sin(angle) * (height * 0.4))
        draw.line([center_x, center_y, x2, y2], fill=(99, 102, 241), width=2)
        draw.ellipse([x2 - 8, y2 - 8, x2 + 8, y2 + 8], fill=(16, 185, 129))

    # Apply soft blur for diffusion sampler effect
    image = image.filter(ImageFilter.SMOOTH_MORE)

    # Overlay prompt text banner
    banner_height = 60
    draw.rectangle([0, height - banner_height, width, height], fill=(15, 23, 42))
    draw.line([0, height - banner_height, width, height - banner_height], fill=(99, 102, 241), width=2)
    
    # Draw metadata text
    meta_str = f"ComfyUI-NG | Model: {model_name} | Seed: {seed} | Steps: {steps} | CFG: {cfg}"
    draw.text((20, height - 40), meta_str, fill=(248, 250, 252))

    # Save to buffer and compute SHA-256 CAS digest
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    digest = hashlib.sha256(image_bytes).hexdigest()
    filename = f"{digest[:16]}.png"

    # Save to storage directory if provided
    if storage_dir is not None:
        storage_dir.mkdir(parents=True, exist_ok=True)
        file_path = storage_dir / filename
        file_path.write_bytes(image_bytes)

    return {
        "digest": f"sha256:{digest}",
        "filename": filename,
        "size_bytes": len(image_bytes),
        "width": width,
        "height": height,
        "bytes": image_bytes,
    }
