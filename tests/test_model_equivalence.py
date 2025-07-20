#!/usr/bin/env python3
"""
Test script to systematically compare HF Llama and Forgather models
to identify where outputs diverge after weight transfer.

Author's note: this was written by Claude, while troubleshooting weight compatibility between the Forgather Llama
implementation and the Huggingface implementation. Keeping for reference and for ensuring that our implementation
remains compatible.

The issues which were causing compatibility issues:
    1. The input encoder was still configured to scale the embeddings by sqrt(d_model). Llama does not do this.
    2. The complex RoPE implementation seems to produce slightly different results.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import sys
import os

from forgather.ml.remap_params import remap_state_dict, hflamma_to_dllama


def issimilar(a, b, rtol=1e-5, atol=1e-8):
    """Check if two tensors are similar within tolerance."""
    if a.shape != b.shape:
        return False
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def load_models():
    """Load both HF and Forgather models."""
    print("Loading models...")

    # Load HF model
    hf_model_path = "../examples/tutorials/tiny_llama/output_models/hf_llama"
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path, trust_remote_code=True
    )
    hf_model.eval()

    # Load Forgather model
    forgather_model_path = "../examples/tutorials/tiny_llama/output_models/tiny_llama"
    forgather_model = AutoModelForCausalLM.from_pretrained(
        forgather_model_path, trust_remote_code=True
    )
    forgather_model.eval()

    # Transfer weights from HF to Forgather
    print("Transferring weights...")
    output_state_dict = remap_state_dict(hf_model.state_dict(), hflamma_to_dllama)
    mismatched = forgather_model.load_state_dict(output_state_dict, strict=False)
    print(f"Missing keys: {mismatched.missing_keys}")
    print(f"Unexpected keys: {mismatched.unexpected_keys}")

    return hf_model, forgather_model


def create_test_inputs(batch_size=1, seq_len=10, vocab_size=2000):
    """Create test inputs for comparison."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Create hidden states tensor for intermediate testing
    hidden_size = 256
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    return input_ids, position_ids, hidden_states


def test_embeddings(hf_model, forgather_model, input_ids):
    """Test embedding layers."""
    print("\n=== Testing Embeddings ===")

    # HF embeddings
    hf_embeds = hf_model.model.embed_tokens(input_ids)

    # Forgather embeddings
    forgather_embeds = forgather_model.causal_lm.input_encoder.embedding(input_ids)

    similar = issimilar(hf_embeds, forgather_embeds)
    print(f"Embeddings similar: {similar}")
    if not similar:
        print(f"HF shape: {hf_embeds.shape}, Forgather shape: {forgather_embeds.shape}")
        print(f"Max diff: {torch.max(torch.abs(hf_embeds - forgather_embeds)).item()}")

    return hf_embeds, forgather_embeds, similar


def test_rms_norm(hf_model, forgather_model, hidden_states):
    """Test RMSNorm layers."""
    print("\n=== Testing RMSNorm ===")

    # Test final layer norm
    hf_norm_out = hf_model.model.norm(hidden_states)
    forgather_norm_out = forgather_model.causal_lm.layer_stack.layer_norm(hidden_states)

    similar = issimilar(hf_norm_out, forgather_norm_out)
    print(f"Final RMSNorm similar: {similar}")
    if not similar:
        print(
            f"Max diff: {torch.max(torch.abs(hf_norm_out - forgather_norm_out)).item()}"
        )

    return similar


def test_attention_layer(
    hf_model, forgather_model, hidden_states, position_ids, layer_idx=0
):
    """Test a single attention layer."""
    print(f"\n=== Testing Attention Layer {layer_idx} ===")

    try:
        # Get layers
        hf_attn = hf_model.model.layers[layer_idx].self_attn
        forgather_attn = forgather_model.causal_lm.layer_stack.layers[
            layer_idx
        ].attention

        # Get position embeddings from HF model's actual rotary_emb
        cos, sin = hf_model.model.rotary_emb(hidden_states, position_ids)

        # HF attention forward
        hf_results = hf_attn(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
        )
        hf_output = hf_results[0] if isinstance(hf_results, tuple) else hf_results

        # Forgather attention forward
        forgather_output = forgather_attn(qkv=hidden_states)

        similar = issimilar(hf_output, forgather_output)
        print(f"Attention layer {layer_idx} similar: {similar}")
        if not similar:
            print(
                f"Max diff: {torch.max(torch.abs(hf_output - forgather_output)).item()}"
            )
            print(
                f"HF shape: {hf_output.shape}, Forgather shape: {forgather_output.shape}"
            )

        return hf_output, forgather_output, similar

    except Exception as e:
        raise e
        # print(f"Attention layer {layer_idx} test failed with error: {e}")
        return hidden_states, hidden_states, False


def test_mlp_layer(hf_model, forgather_model, hidden_states, layer_idx=0):
    """Test a single MLP/feedforward layer."""
    print(f"\n=== Testing MLP Layer {layer_idx} ===")

    # Get layers
    hf_mlp = hf_model.model.layers[layer_idx].mlp
    forgather_ff = forgather_model.causal_lm.layer_stack.layers[layer_idx].feedforward

    # Forward pass
    hf_output = hf_mlp(hidden_states)
    forgather_output = forgather_ff(hidden_states)

    similar = issimilar(hf_output, forgather_output)
    print(f"MLP layer {layer_idx} similar: {similar}")
    if not similar:
        print(f"Max diff: {torch.max(torch.abs(hf_output - forgather_output)).item()}")

    return hf_output, forgather_output, similar


def test_full_transformer_layer(
    hf_model, forgather_model, hidden_states, position_ids, layer_idx=0
):
    """Test a complete transformer layer (attention + MLP + residuals + norms)."""
    print(f"\n=== Testing Full Transformer Layer {layer_idx} ===")

    try:
        # HF layer forward
        hf_layer = hf_model.model.layers[layer_idx]

        # Get position embeddings from HF model's actual rotary_emb
        cos, sin = hf_model.model.rotary_emb(hidden_states, position_ids)

        hf_results = hf_layer(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=(cos, sin),
        )
        hf_output = hf_results[0] if isinstance(hf_results, tuple) else hf_results

        # Forgather layer forward
        forgather_layer = forgather_model.causal_lm.layer_stack.layers[layer_idx]
        seq_len = hidden_states.shape[1]

        forgather_output = forgather_layer(hidden_states)

        similar = issimilar(hf_output, forgather_output)
        print(f"Full transformer layer {layer_idx} similar: {similar}")
        if not similar:
            print(
                f"Max diff: {torch.max(torch.abs(hf_output - forgather_output)).item()}"
            )

        return hf_output, forgather_output, similar

    except Exception as e:
        print(f"Full transformer layer {layer_idx} test failed with error: {e}")
        return hidden_states, hidden_states, False


def test_full_model(hf_model, forgather_model, input_ids, position_ids):
    """Test the complete model forward pass."""
    print("\n=== Testing Full Model ===")

    with torch.no_grad():
        # HF model forward
        hf_outputs = hf_model(input_ids=input_ids, position_ids=position_ids)
        hf_logits = hf_outputs.logits

        # Forgather model forward
        forgather_outputs = forgather_model(
            input_ids=input_ids, position_ids=position_ids
        )
        # Handle different output formats
        if isinstance(forgather_outputs, dict):
            forgather_logits = forgather_outputs["logits"]
        elif hasattr(forgather_outputs, "logits"):
            forgather_logits = forgather_outputs.logits
        else:
            forgather_logits = (
                forgather_outputs  # Assume it's the logits tensor directly
            )

        similar = issimilar(hf_logits, forgather_logits)
        print(f"Full model logits similar: {similar}")
        if not similar:
            print(
                f"Max diff: {torch.max(torch.abs(hf_logits - forgather_logits)).item()}"
            )
            print(
                f"Mean diff: {torch.mean(torch.abs(hf_logits - forgather_logits)).item()}"
            )
            print(
                f"HF logits range: [{hf_logits.min().item():.6f}, {hf_logits.max().item():.6f}]"
            )
            print(
                f"Forgather logits range: [{forgather_logits.min().item():.6f}, {forgather_logits.max().item():.6f}]"
            )

    return similar


def main():
    """Run all tests systematically."""
    print("=== Model Equivalence Testing ===")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load models
    hf_model, forgather_model = load_models()

    # Create test inputs
    input_ids, position_ids, hidden_states = create_test_inputs()

    print(f"Test input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  hidden_states: {hidden_states.shape}")

    # Run tests in order of increasing complexity
    results = {}

    # 1. Test embeddings
    hf_embeds, forgather_embeds, results["embeddings"] = test_embeddings(
        hf_model, forgather_model, input_ids
    )

    # 2. Test RMSNorm
    results["rms_norm"] = test_rms_norm(hf_model, forgather_model, hidden_states)

    # 3. Test RoPE
    # TODO: Rewrite

    # 4. Test individual components for first layer
    _, _, results["attention_0"] = test_attention_layer(
        hf_model, forgather_model, hf_embeds, position_ids, 0
    )
    _, _, results["mlp_0"] = test_mlp_layer(hf_model, forgather_model, hf_embeds, 0)

    # 5. Test full transformer layers
    current_hf = hf_embeds
    current_forgather = forgather_embeds

    for layer_idx in range(4):  # 4 layers in tiny model
        hf_out, forgather_out, layer_similar = test_full_transformer_layer(
            hf_model, forgather_model, current_hf, position_ids, layer_idx
        )
        results[f"layer_{layer_idx}"] = layer_similar
        current_hf = hf_out
        current_forgather = forgather_out

        if not layer_similar:
            print(f"*** DIVERGENCE DETECTED AT LAYER {layer_idx} ***")
            break

    # 6. Test full model
    results["full_model"] = test_full_model(
        hf_model, forgather_model, input_ids, position_ids
    )

    # Summary
    print("\n=== SUMMARY ===")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

    # Identify first failure point
    failed_tests = [name for name, passed in results.items() if not passed]
    if failed_tests:
        print(f"\n🎯 First failure point: {failed_tests[0]}")
    else:
        print("\n🎉 All tests passed!")


if __name__ == "__main__":
    main()
