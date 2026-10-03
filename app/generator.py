import logging
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_URL = os.getenv("GROK_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROK_MODEL = os.getenv("GROK_MODEL", "llama-3.3-70b-versatile")
GROK_TIMEOUT = float(os.getenv("GROK_TIMEOUT", "60"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))


class GrokError(RuntimeError):
    """Raised when the generation API is unreachable or returns an unusable response."""


def call_grok(messages: List[Dict[str, str]]) -> str:
    if not GROK_API_KEY:
        raise GrokError("GROK_API_KEY is not set; cannot generate an answer.")

    payload = {"model": GROK_MODEL, "messages": messages, "max_tokens": 2048, "temperature": 0.2}
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(GROK_API_URL, json=payload, headers=headers, timeout=GROK_TIMEOUT)
    except requests.Timeout as exc:
        raise GrokError(f"Grok API timed out after {GROK_TIMEOUT:.0f}s.") from exc
    except requests.RequestException as exc:
        raise GrokError(f"Could not reach Grok API at {GROK_API_URL}: {exc}") from exc

    if response.status_code >= 400:
        detail = (response.text or "").strip()[:400]
        logger.error("Grok API returned HTTP %s: %s", response.status_code, detail)
        raise GrokError(f"Grok API returned HTTP {response.status_code}. {detail}")

    try:
        data = response.json()
    except ValueError as exc:
        raise GrokError("Grok API returned a non-JSON response.") from exc

    choices = data.get("choices") or []
    if not choices:
        raise GrokError(f"Grok API returned no choices. Response: {str(data)[:400]}")

    content = (choices[0].get("message") or {}).get("content") or ""
    if not content.strip():
        finish_reason = choices[0].get("finish_reason")
        raise GrokError(f"Grok API returned an empty message (finish_reason={finish_reason}).")
    return content


def build_context_block(contexts: List[Dict[str, Any]], budget: int = MAX_CONTEXT_CHARS) -> str:
    """Render retrieved chunks as a labelled context block within a character budget."""
    blocks: List[str] = []
    used = 0
    for i, ctx in enumerate(contexts):
        text = (ctx.get("text") or "").strip()
        if not text:
            continue
        meta = ctx.get("meta") or {}
        header = (
            f"[Context {i+1}] (page {meta.get('page')}, "
            f"source: {meta.get('source', 'unknown')}, type: {meta.get('type', 'text')})"
        )
        block = f"{header}\n{text}\n---"
        if used + len(block) > budget:
            blocks.append(f"{header}\n{text[:max(0, budget - used)]}\n[truncated]")
            break
        blocks.append(block)
        used += len(block)
    return "\n".join(blocks)


def rag_answer(
    question: str,
    contexts: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Answer `question` from `contexts` via Grok. Raises GrokError on any failure."""
    system_prompt = (
        "You are a macroeconomic research assistant. Answer using only the provided context. "
        "If the context does not contain the answer, say so explicitly rather than guessing. "
        "When quoting figures from a table, name the table and page. Keep answers concise."
    )
    context_block = build_context_block(contexts)
    if context_block:
        system_prompt += f"\n\nContext:\n{context_block}"

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in (history or [])[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    return call_grok(messages)