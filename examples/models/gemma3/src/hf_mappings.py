"""Parameter name mappings between HuggingFace Gemma-3 and Forgather formats.

Gemma-3 has:
- QK-norm on the attention module (``q_norm``/``k_norm``, size ``head_dim``)
- Four layernorms per decoder layer (pre/post attention, pre/post feedforward)
- No biases on Q/K/V/O projections or on MLP layers

The Forgather side uses the HF layernorm names verbatim so the remapping is
identity for the four decoder-layer norms.
"""

HF_TO_FORGATHER = [
    (r"lm_head\.", r"lm_head.", []),
    (
        r"model\.",
        r"causal_lm.",
        [
            (r"embed_tokens\.", r"input_encoder.embedding.", []),
            (r"norm\.", r"layer_stack.layer_norm.", []),
            (
                r"layers\.(\d+)\.",
                r"layer_stack.layers.\1.",
                [
                    (
                        r"self_attn\.",
                        r"attention.",
                        [
                            (r"q_proj\.", r"query_linear.", []),
                            (r"k_proj\.", r"key_linear.", []),
                            (r"v_proj\.", r"value_linear.", []),
                            (r"o_proj\.", r"output_linear.", []),
                            (r"q_norm\.", r"q_norm.", []),
                            (r"k_norm\.", r"k_norm.", []),
                        ],
                    ),
                    (r"mlp\.", r"feedforward.", []),
                    (r"input_layernorm\.", r"input_layernorm.", []),
                    (
                        r"post_attention_layernorm\.",
                        r"post_attention_layernorm.",
                        [],
                    ),
                    (
                        r"pre_feedforward_layernorm\.",
                        r"pre_feedforward_layernorm.",
                        [],
                    ),
                    (
                        r"post_feedforward_layernorm\.",
                        r"post_feedforward_layernorm.",
                        [],
                    ),
                ],
            ),
        ],
    ),
]

FORGATHER_TO_HF = [
    (r"lm_head\.", r"lm_head.", []),
    (
        r"causal_lm\.",
        r"model.",
        [
            (r"input_encoder\.embedding\.", r"embed_tokens.", []),
            (r"layer_stack\.layer_norm\.", r"norm.", []),
            (
                r"layer_stack\.layers\.(\d+)\.",
                r"layers.\1.",
                [
                    (
                        r"attention\.",
                        r"self_attn.",
                        [
                            (r"query_linear\.", r"q_proj.", []),
                            (r"key_linear\.", r"k_proj.", []),
                            (r"value_linear\.", r"v_proj.", []),
                            (r"output_linear\.", r"o_proj.", []),
                            (r"q_norm\.", r"q_norm.", []),
                            (r"k_norm\.", r"k_norm.", []),
                        ],
                    ),
                    (r"feedforward\.", r"mlp.", []),
                    (r"input_layernorm\.", r"input_layernorm.", []),
                    (
                        r"post_attention_layernorm\.",
                        r"post_attention_layernorm.",
                        [],
                    ),
                    (
                        r"pre_feedforward_layernorm\.",
                        r"pre_feedforward_layernorm.",
                        [],
                    ),
                    (
                        r"post_feedforward_layernorm\.",
                        r"post_feedforward_layernorm.",
                        [],
                    ),
                ],
            ),
        ],
    ),
]
