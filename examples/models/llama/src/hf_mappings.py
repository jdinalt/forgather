"""Parameter name mappings between HuggingFace Llama and Forgather formats."""

# HuggingFace Llama to Forgather Dynamic Llama parameter name mappings
# Format: List of (pattern, replacement, [children]) tuples for recursive regex substitution
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
                        ],
                    ),
                    (r"mlp\.", r"feedforward.", []),
                    (r"input_layernorm\.", r"norm1.", []),
                    (r"post_attention_layernorm\.", r"norm2.", []),
                ],
            ),
        ],
    ),
]

# Forgather Dynamic Llama to HuggingFace Llama parameter name mappings
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
                        ],
                    ),
                    (r"feedforward\.", r"mlp.", []),
                    (r"norm1\.", r"input_layernorm.", []),
                    (r"norm2\.", r"post_attention_layernorm.", []),
                ],
            ),
        ],
    ),
]
