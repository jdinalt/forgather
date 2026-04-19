# Tiny Models

Train and compare different small language model architectures on the TinyStories
dataset with a shared 2K-token BPE tokenizer. All models are ~4M parameters,
allowing direct comparison of architecture choices under identical conditions.

## Configurations

**Forgather model implementations:**

| Config | Architecture | Key features |
|--------|-------------|--------------|
| `tiny_causal.yaml` | Vanilla Transformer | Basic decoder-only transformer, loosely based on "Attention is All You Need" |
| `tiny_fg_llama.yaml` | Llama | Pre-layer-norm, RoPE, SiLU activation, GQA |
| `tiny_deepone.yaml` | DeepOne | Post-layer-norm, Deepnet initialization, ALiBi positional encoding |
| `tiny_fg_mistral.yaml` | Mistral | Llama variant with sliding-window attention and GQA |
| `tiny_fg_qwen3.yaml` | Qwen3 | Qwen3 architecture |
| `tiny_llama_canon.yaml` | Llama + Canon | Llama with Canon convolutional layers for local token mixing |
| `tiny_single_head.yaml` | Single-Head ALiBi | Single attention head with ALiBi, using eager attention |

**HuggingFace model implementations** (for comparison with Forgather equivalents):

| Config | Architecture | Notes |
|--------|-------------|-------|
| `tiny_hf_llama.yaml` | HF LlamaForCausalLM | Same architecture as `tiny_fg_llama`, HF implementation |
| `tiny_hf_gpt2.yaml` | HF GPT2LMHeadModel | GPT-2 architecture |

## Usage

```bash
# Train a single model
forgather -t tiny_fg_llama.yaml train

# Train all models for comparison
for cfg in $(forgather ls | grep -oP '\S+\.yaml'); do
    forgather -t $cfg train
done

# Compare loss curves
forgather logs plot --compare output_models/*/runs/*/trainer_logs.json --loss-curves
```

## Adding a new model

1. Create a model project under `examples/models/` (or use an existing one)
2. Add a config in `templates/configs/` that sets `ns.model_project_dir` and
   `ns.model_project_config` to point to your model
3. Run `forgather ls` to verify it parses correctly
