from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import httpx

from ..config import get_settings


def vision_configuration() -> dict[str, str]:
    """Use explicit vision settings, or DeepSeek's documented image model."""
    settings = get_settings()
    if settings.vision_api_base and settings.vision_api_key and settings.vision_model:
        return {"base": settings.vision_api_base, "key": settings.vision_api_key,
                "model": settings.vision_model, "source": "explicit"}
    if (settings.llm_api_base.rstrip("/").lower() in
            {"https://api.deepseek.com", "https://api.deepseek.com/v1"}
            and settings.llm_api_key and not settings.llm_api_key.startswith("replace-")):
        return {"base": settings.llm_api_base, "key": settings.llm_api_key,
                "model": "deepseek-flash", "source": "deepseek_default"}
    return {}


def analyze_figure(image_path: Path, *, context: str = "") -> dict:
    """Analyze an extracted figure with explicit vision settings or DeepSeek Flash."""
    config = vision_configuration()
    if not config:
        return {
            "status": "pending_configuration",
            "message": "Configure VISION_API_BASE, VISION_API_KEY, and VISION_MODEL to enable figure analysis.",
        }

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"
    prompt = (
        "Describe the visible research figure faithfully. Identify its type (chart, architecture, "
        "flowchart, or other), labels, relations, and any legible values. Do not infer details that "
        "are not visible. Separate uncertain observations."
    )
    if context.strip():
        prompt += f"\nNearby paper text or caption:\n{context.strip()[:3000]}"

    try:
        response = httpx.post(
            f"{config['base'].rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {config['key']}"},
            json={
                "model": config["model"],
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                "temperature": 0,
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        description = payload["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as exc:
        return {"status": "failed", "model": config["model"],
                "error_type": "HTTPStatusError", "http_status": exc.response.status_code,
                "message": exc.response.text[:300]}
    except Exception as exc:
        return {
            "status": "failed",
            "model": config["model"],
            "error_type": exc.__class__.__name__,
        }
    return {"status": "completed", "model": config["model"], "description": description}
