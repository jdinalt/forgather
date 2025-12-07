"""Parameter name mappings between HuggingFace Qwen3 and Forgather formats."""

# HuggingFace Qwen3 to Forgather Dynamic Llama parameter name mappings
# Qwen3 has additional q_norm, k_norm, and biases on Q/K/V projections
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
                            (r"q_proj\.weight", r"query_linear.weight", []),
                            (r"q_proj\.bias", r"query_linear.bias", []),
                            (r"k_proj\.weight", r"key_linear.weight", []),
                            (r"k_proj\.bias", r"key_linear.bias", []),
                            (r"v_proj\.weight", r"value_linear.weight", []),
                            (r"v_proj\.bias", r"value_linear.bias", []),
                            (r"o_proj\.", r"output_linear.", []),
                            (r"q_norm\.", r"q_norm.", []),
                            (r"k_norm\.", r"k_norm.", []),
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

# Forgather Dynamic Llama to HuggingFace Qwen3 parameter name mappings
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
                            (r"query_linear\.weight", r"q_proj.weight", []),
                            (r"query_linear\.bias", r"q_proj.bias", []),
                            (r"key_linear\.weight", r"k_proj.weight", []),
                            (r"key_linear\.bias", r"k_proj.bias", []),
                            (r"value_linear\.weight", r"v_proj.weight", []),
                            (r"value_linear\.bias", r"v_proj.bias", []),
                            (r"output_linear\.", r"o_proj.", []),
                            (r"q_norm\.", r"q_norm.", []),
                            (r"k_norm\.", r"k_norm.", []),
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
