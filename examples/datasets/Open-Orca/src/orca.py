"""
Preprocessing helpers for the Open-Orca/OpenOrca dataset.

The dataset stores each example as three raw text fields: ``system_prompt``,
``question`` and ``response``. This module provides map functions that render
those fields through a Jinja chat template and then either tokenize the result
directly (``orca_map_fn``) or hand the rendered text off to
``block_tokenize_fn`` for sequence packing (``orca_packed_map_fn``).
"""

import logging
import os

import jinja2
import jinja2.sandbox

from forgather.ml.datasets import block_tokenize_fn

logger = logging.getLogger(__name__)


DEFAULT_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def load_chat_template(path: str | None = None) -> jinja2.Template:
    """Compile a Jinja chat template.

    ``path`` may be a filesystem path, an inline template string or ``None``
    (in which case the bundled ChatML default is used). The returned template
    is compiled in an immutable sandbox and is safe to share across workers.
    """
    if path and os.path.exists(path):
        with open(path, "r") as f:
            template_str = f.read()
    elif path:
        template_str = path
    else:
        template_str = DEFAULT_CHATML_TEMPLATE
        logger.info("Using bundled default ChatML chat template")

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.from_string(template_str)


def _render_batch(batch, tokenizer, chat_template, template_args):
    if chat_template is None:
        chat_template = load_chat_template()

    render_args = dict(template_args) if template_args else {}
    if tokenizer is not None:
        if tokenizer.bos_token is not None:
            render_args.setdefault("bos_token", tokenizer.bos_token)
        if tokenizer.eos_token is not None:
            render_args.setdefault("eos_token", tokenizer.eos_token)

    return [
        chat_template.render(
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": question or ""},
                {"role": "assistant", "content": response or ""},
            ],
            **render_args,
        )
        for system, question, response in zip(
            batch["system_prompt"], batch["question"], batch["response"]
        )
    ]


def orca_map_fn(
    batch,
    tokenizer,
    feature=None,
    chat_template=None,
    template_args=None,
    **tokenizer_kwargs,
):
    """Render OpenOrca examples through ``chat_template`` and tokenize them.

    ``feature`` is accepted (and ignored) so that this function is drop-in
    compatible with ``preprocess_dataset``'s ``map_fn`` contract, which always
    supplies ``tokenizer`` and ``feature`` via ``fn_kwargs``. Additional
    keyword arguments are forwarded to the tokenizer call, so pass things
    like ``truncation`` or ``max_length`` through ``preprocess_args``.
    """
    del feature  # OpenOrca reads multiple fields rather than a single feature

    texts = _render_batch(batch, tokenizer, chat_template, template_args)
    if tokenizer is None:
        return {"text": texts}

    outputs = tokenizer(texts, **tokenizer_kwargs)
    return {"input_ids": outputs["input_ids"]}


def orca_packed_map_fn(
    batch,
    tokenizer,
    feature=None,
    chat_template=None,
    template_args=None,
    **block_kwargs,
):
    """Render OpenOrca examples and hand them to ``block_tokenize_fn``.

    Any keyword arguments not consumed by the chat-template rendering step are
    forwarded to ``block_tokenize_fn`` so that packing parameters (e.g.
    ``max_length``, ``packing_strategy``, ``add_bos``) can be bound via
    ``!partial`` in the dataset config.
    """
    del feature

    texts = _render_batch(batch, tokenizer, chat_template, template_args)
    return block_tokenize_fn(
        {"text": texts},
        tokenizer=tokenizer,
        feature="text",
        **block_kwargs,
    )
