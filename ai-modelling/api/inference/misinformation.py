"""
Misinformation scoring: pure functions over a ``LoadedModel`` bundle.

Routers call this after resolving ``model_id`` via ``model_loader``.
"""
from typing import Any

from api.model_loader import LoadedModel
from src.models.misinformation.deberta import classify_multitask, classify_text


def predict_misinformation(post: dict[str, Any], bundle: LoadedModel) -> dict[str, Any]:
    """
    Classify a single social post across every task the bundle exposes.
    Required keys: ``id``, ``content``.
    """
    if "id" not in post or "content" not in post:
        raise KeyError("post must include 'id' and 'content'")

    if bundle.kind == "deberta_multitask":
        tasks = classify_multitask(
            str(post["content"]),
            tokenizer=bundle.tokenizer,
            model=bundle.model,
            device=bundle.device,
            max_len=bundle.max_len,
        )
        if "misinfo" not in tasks:
            raise RuntimeError(
                f"multitask model '{bundle.model_id}' does not expose a 'misinfo' head"
            )
    else:
        tasks = {
            "misinfo": classify_text(
                str(post["content"]),
                tokenizer=bundle.tokenizer,
                model=bundle.model,
                device=bundle.device,
                max_len=bundle.max_len,
            )
        }

    return {
        "model_id": bundle.model_id,
        "domain": bundle.domain,
        "id": post["id"],
        "author_name": post.get("author_name"),
        "platform": post.get("platform"),
        "content": post["content"],
        "share_count": post.get("share_count"),
        "ts": post.get("ts"),
        "post_url": post.get("post_url"),
        "tasks": tasks,
        "checkpoint": str(bundle.checkpoint_path),
    }
