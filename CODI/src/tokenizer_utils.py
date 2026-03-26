import logging

from transformers import AutoTokenizer


def load_tokenizer_with_fallback(pretrained_model_name_or_path, *args, use_fast=False, **kwargs):
    """Load the requested tokenizer and retry with the fast backend if needed.

    Qwen3 local checkpoints expose a fast tokenizer that works, while the slow
    tokenizer path can fail because the converted tokenizer assets are
    incomplete. Keep the current slow-tokenizer preference for existing models,
    but transparently retry with the fast backend when the slow path breaks.
    """

    try:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            use_fast=use_fast,
            **kwargs,
        )
    except Exception as exc:
        if use_fast:
            raise

        logging.warning(
            "Falling back to the fast tokenizer for %s because the slow tokenizer "
            "failed to load: %s",
            pretrained_model_name_or_path,
            exc,
        )
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            use_fast=True,
            **kwargs,
        )
