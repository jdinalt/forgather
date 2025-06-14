Llama breakage!

https://github.com/pytorch/pytorch/issues/128394

caused by https://github.com/huggingface/transformers/pull/29285

https://github.com/xuhancn/pytorch/commit/07e356d3ef40ff933e5fc2e9aecdba2e2deed25b

And more breakage!

https://github.com/huggingface/transformers/commit/ad3d157188f434655fe0919bad625088952a4dcf#diff-06392bad3b9e97be9ade60d4ac46f73b6809388f4d507c2ba1384ab872711c51

https://github.com/pytorch/ao/tree/main/torchao/optim

https://pytorch.org/blog/pytorch-native-architecture-optimization/

https://arxiv.org/pdf/2502.10940


When splitting a model with pipeline(), an assert is tripped if the model has a sub-module with a no_grad() context manager. I encountered this when trying to use the API on the Huggingface "meta-llama--Llama-2-7b-hf" model. The issue is triggered by the use of a no_grad() decorator on the forward method of "LlamaRotaryEmbedding"

https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L93

The model had other issues, which I resolved before encountering this one:

An auto cast, which causes export() to fail:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L100

Commented out.

Slicing "self.layers" triggers and infinite recursion.
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L439

Commented out slicing.

I made a copy of the model code, placed it in the model's directory, and modified the config to load the code from the directory, rather than from the HF library.

Modified Config:
```json
{
  "_name_or_path": "meta-llama/Llama-2-7b-hf",
  "architectures": [
    "XLlamaForCausalLM"
  ],
  "auto_map": {
    "AutoConfig": "configuration_llama.LlamaConfig",
    "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM"
  },
...
```

Construct an empty model on the meta device from the config.

```python
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# Make debugging trace easier; reduce the number of layers to 1
model_config.num_hidden_layers = 1

# Construct on the meta device -- with.torch_device("meta") triggers a loading bug, but the accelerate init_empty_weights works.
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    
print(model)
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleDict(
      (0): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```

```python
from torch.distributed.pipelining import SplitPoint, ScheduleGPipe, PipelineStage, pipe_split, pipeline

# Create a single split point
split_spec = {
    f"model.layers.0": SplitPoint.END
}

# Make example input
mb_prompts = (
    "How do you", "I like to",
)  # microbatch size = 2

with torch.device("meta"):
    mb_inputs = tokenizer(mb_prompts, return_tensors="pt", padding=True)
example_input = mb_inputs["input_ids"]

# Try to build pipeline
pipe = pipeline(
    model,
    tuple(example_input,),
    dict(input_ids=example_input),
    split_spec=split_spec,
)
```

### Error

```
I0610 18:40:25.340000 93824 torch/distributed/pipelining/_IR.py:1003] Tracing model ...
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690] class GraphModule(torch.nn.Module):
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]     def forward(self, input_ids):
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         input_ids: "i64[2, 4]"; 
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]     
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         input_ids, = fx_pytree.tree_flatten_spec(([], {'input_ids':input_ids}), self._in_spec)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         # No stacktrace found for following nodes
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_layers_0_input_layernorm_weight: "f32[4096]" = getattr(self.model.layers, "0").input_layernorm.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_layers_0_post_attention_layernorm_weight: "f32[4096]" = getattr(self.model.layers, "0").post_attention_layernorm.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_norm_weight: "f32[4096]" = self.model.norm.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_embed_tokens_weight: "f32[32000, 4096]" = self.model.embed_tokens.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_layers_0_self_attn_q_proj_weight: "f32[4096, 4096]" = getattr(self.model.layers, "0").self_attn.q_proj.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_layers_0_self_attn_k_proj_weight: "f32[4096, 4096]" = getattr(self.model.layers, "0").self_attn.k_proj.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_layers_0_self_attn_v_proj_weight: "f32[4096, 4096]" = getattr(self.model.layers, "0").self_attn.v_proj.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_layers_0_self_attn_o_proj_weight: "f32[4096, 4096]" = getattr(self.model.layers, "0").self_attn.o_proj.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_layers_0_mlp_gate_proj_weight: "f32[11008, 4096]" = getattr(self.model.layers, "0").mlp.gate_proj.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_layers_0_mlp_up_proj_weight: "f32[11008, 4096]" = getattr(self.model.layers, "0").mlp.up_proj.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_layers_0_mlp_down_proj_weight: "f32[4096, 11008]" = getattr(self.model.layers, "0").mlp.down_proj.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         lm_head_weight: "f32[32000, 4096]" = self.lm_head.weight
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         model_rotary_emb_inv_freq: "f32[64]" = self.model.rotary_emb.inv_freq
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:531 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         embedding: "f32[2, 4, 4096]" = torch.ops.aten.embedding.default(model_embed_tokens_weight, input_ids);  model_embed_tokens_weight = input_ids = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:538 in forward, code: cache_position = torch.arange(
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         arange: "i64[4]" = torch.ops.aten.arange.start(0, 4, device = device(type='meta'), pin_memory = False)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:543 in forward, code: position_ids = cache_position.unsqueeze(0)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         unsqueeze: "i64[1, 4]" = torch.ops.aten.unsqueeze.default(arange, 0);  arange = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         # No stacktrace found for following nodes
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         submod_3 = self.submod_1
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         wrap_with_set_grad_enabled = torch.ops.higher_order.wrap_with_set_grad_enabled(False, submod_3, model_rotary_emb_inv_freq, unsqueeze);  submod_3 = model_rotary_emb_inv_freq = unsqueeze = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:123 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         to_5: "f32[1, 4, 128]" = wrap_with_set_grad_enabled[0]
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         to_6: "f32[1, 4, 128]" = wrap_with_set_grad_enabled[1];  wrap_with_set_grad_enabled = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:80 in forward, code: hidden_states = hidden_states.to(torch.float32)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         to_7: "f32[2, 4, 4096]" = torch.ops.aten.to.dtype(embedding, torch.float32);  embedding = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:81 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         pow_1: "f32[2, 4, 4096]" = torch.ops.aten.pow.Tensor_Scalar(to_7, 2)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mean: "f32[2, 4, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:82 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         add: "f32[2, 4, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         rsqrt: "f32[2, 4, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_2: "f32[2, 4, 4096]" = torch.ops.aten.mul.Tensor(to_7, rsqrt);  rsqrt = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:83 in forward, code: return self.weight * hidden_states.to(input_dtype)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         to_8: "f32[2, 4, 4096]" = torch.ops.aten.to.dtype(mul_2, torch.float32);  mul_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_3: "f32[2, 4, 4096]" = torch.ops.aten.mul.Tensor(model_layers_0_input_layernorm_weight, to_8);  model_layers_0_input_layernorm_weight = to_8 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:252 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         linear: "f32[2, 4, 4096]" = torch.ops.aten.linear.default(mul_3, model_layers_0_self_attn_q_proj_weight);  model_layers_0_self_attn_q_proj_weight = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         view: "f32[2, 4, 32, 128]" = torch.ops.aten.view.default(linear, [2, 4, -1, 128]);  linear = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         transpose_1: "f32[2, 32, 4, 128]" = torch.ops.aten.transpose.int(view, 1, 2);  view = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:253 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         linear_1: "f32[2, 4, 4096]" = torch.ops.aten.linear.default(mul_3, model_layers_0_self_attn_k_proj_weight);  model_layers_0_self_attn_k_proj_weight = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         view_1: "f32[2, 4, 32, 128]" = torch.ops.aten.view.default(linear_1, [2, 4, -1, 128]);  linear_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         transpose_2: "f32[2, 32, 4, 128]" = torch.ops.aten.transpose.int(view_1, 1, 2);  view_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:254 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         linear_2: "f32[2, 4, 4096]" = torch.ops.aten.linear.default(mul_3, model_layers_0_self_attn_v_proj_weight);  mul_3 = model_layers_0_self_attn_v_proj_weight = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         view_2: "f32[2, 4, 32, 128]" = torch.ops.aten.view.default(linear_2, [2, 4, -1, 128]);  linear_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         transpose_3: "f32[2, 32, 4, 128]" = torch.ops.aten.transpose.int(view_2, 1, 2);  view_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:153 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         unsqueeze_4: "f32[1, 1, 4, 128]" = torch.ops.aten.unsqueeze.default(to_5, 1);  to_5 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:154 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         unsqueeze_5: "f32[1, 1, 4, 128]" = torch.ops.aten.unsqueeze.default(to_6, 1);  to_6 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:155 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_4: "f32[2, 32, 4, 128]" = torch.ops.aten.mul.Tensor(transpose_1, unsqueeze_4)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:128 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         slice_4: "f32[2, 32, 4, 64]" = torch.ops.aten.slice.Tensor(transpose_1, 3, 0, 64)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:129 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         slice_5: "f32[2, 32, 4, 64]" = torch.ops.aten.slice.Tensor(transpose_1, 3, 64, 9223372036854775807);  transpose_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:130 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         neg: "f32[2, 32, 4, 64]" = torch.ops.aten.neg.default(slice_5);  slice_5 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         cat_1: "f32[2, 32, 4, 128]" = torch.ops.aten.cat.default([neg, slice_4], -1);  neg = slice_4 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:155 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_5: "f32[2, 32, 4, 128]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_5);  cat_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         add_1: "f32[2, 32, 4, 128]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:156 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_6: "f32[2, 32, 4, 128]" = torch.ops.aten.mul.Tensor(transpose_2, unsqueeze_4);  unsqueeze_4 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:128 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         slice_6: "f32[2, 32, 4, 64]" = torch.ops.aten.slice.Tensor(transpose_2, 3, 0, 64)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:129 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         slice_7: "f32[2, 32, 4, 64]" = torch.ops.aten.slice.Tensor(transpose_2, 3, 64, 9223372036854775807);  transpose_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:130 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         neg_1: "f32[2, 32, 4, 64]" = torch.ops.aten.neg.default(slice_7);  slice_7 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         cat_2: "f32[2, 32, 4, 128]" = torch.ops.aten.cat.default([neg_1, slice_6], -1);  neg_1 = slice_6 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:156 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_7: "f32[2, 32, 4, 128]" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_5);  cat_2 = unsqueeze_5 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         add_2: "f32[2, 32, 4, 128]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.local/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py:39 in sdpa_attention_forward, code: query = query.contiguous()
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         contiguous: "f32[2, 32, 4, 128]" = torch.ops.aten.contiguous.default(add_1);  add_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.local/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py:40 in sdpa_attention_forward, code: key = key.contiguous()
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         contiguous_1: "f32[2, 32, 4, 128]" = torch.ops.aten.contiguous.default(add_2);  add_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.local/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py:41 in sdpa_attention_forward, code: value = value.contiguous()
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         contiguous_2: "f32[2, 32, 4, 128]" = torch.ops.aten.contiguous.default(transpose_3);  transpose_3 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.local/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py:54 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         scaled_dot_product_attention: "f32[2, 32, 4, 128]" = torch.ops.aten.scaled_dot_product_attention.default(contiguous, contiguous_1, contiguous_2, None, 0.0, True, scale = 0.08838834764831845);  contiguous = contiguous_1 = contiguous_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.local/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py:63 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         transpose_4: "f32[2, 4, 32, 128]" = torch.ops.aten.transpose.int(scaled_dot_product_attention, 1, 2);  scaled_dot_product_attention = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         contiguous_3: "f32[2, 4, 32, 128]" = torch.ops.aten.contiguous.default(transpose_4);  transpose_4 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:285 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         reshape: "f32[2, 4, 4096]" = torch.ops.aten.reshape.default(contiguous_3, [2, 4, -1]);  contiguous_3 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:286 in forward, code: attn_output = self.o_proj(attn_output)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         linear_3: "f32[2, 4, 4096]" = torch.ops.aten.linear.default(reshape, model_layers_0_self_attn_o_proj_weight);  reshape = model_layers_0_self_attn_o_proj_weight = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:329 in forward, code: hidden_states = residual + hidden_states
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         add_3: "f32[2, 4, 4096]" = torch.ops.aten.add.Tensor(to_7, linear_3);  to_7 = linear_3 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:80 in forward, code: hidden_states = hidden_states.to(torch.float32)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         to_9: "f32[2, 4, 4096]" = torch.ops.aten.to.dtype(add_3, torch.float32);  add_3 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:81 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         pow_2: "f32[2, 4, 4096]" = torch.ops.aten.pow.Tensor_Scalar(to_9, 2)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mean_1: "f32[2, 4, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:82 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         add_4: "f32[2, 4, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         rsqrt_1: "f32[2, 4, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_8: "f32[2, 4, 4096]" = torch.ops.aten.mul.Tensor(to_9, rsqrt_1);  rsqrt_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:83 in forward, code: return self.weight * hidden_states.to(input_dtype)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         to_10: "f32[2, 4, 4096]" = torch.ops.aten.to.dtype(mul_8, torch.float32);  mul_8 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_9: "f32[2, 4, 4096]" = torch.ops.aten.mul.Tensor(model_layers_0_post_attention_layernorm_weight, to_10);  model_layers_0_post_attention_layernorm_weight = to_10 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:172 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         linear_4: "f32[2, 4, 11008]" = torch.ops.aten.linear.default(mul_9, model_layers_0_mlp_gate_proj_weight);  model_layers_0_mlp_gate_proj_weight = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         silu: "f32[2, 4, 11008]" = torch.ops.aten.silu.default(linear_4);  linear_4 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         linear_5: "f32[2, 4, 11008]" = torch.ops.aten.linear.default(mul_9, model_layers_0_mlp_up_proj_weight);  mul_9 = model_layers_0_mlp_up_proj_weight = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_10: "f32[2, 4, 11008]" = torch.ops.aten.mul.Tensor(silu, linear_5);  silu = linear_5 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         linear_6: "f32[2, 4, 4096]" = torch.ops.aten.linear.default(mul_10, model_layers_0_mlp_down_proj_weight);  mul_10 = model_layers_0_mlp_down_proj_weight = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:335 in forward, code: hidden_states = residual + hidden_states
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         add_5: "f32[2, 4, 4096]" = torch.ops.aten.add.Tensor(to_9, linear_6);  to_9 = linear_6 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_IR.py:343 in pipe_split, code: return torch.ops.pippy._pipe_split()
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         _pipe_split = torch.ops.pippy._pipe_split.default();  _pipe_split = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:80 in forward, code: hidden_states = hidden_states.to(torch.float32)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         to_11: "f32[2, 4, 4096]" = torch.ops.aten.to.dtype(add_5, torch.float32);  add_5 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:81 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         pow_3: "f32[2, 4, 4096]" = torch.ops.aten.pow.Tensor_Scalar(to_11, 2)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mean_2: "f32[2, 4, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:82 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         add_6: "f32[2, 4, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-05);  mean_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         rsqrt_2: "f32[2, 4, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_11: "f32[2, 4, 4096]" = torch.ops.aten.mul.Tensor(to_11, rsqrt_2);  to_11 = rsqrt_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:83 in forward, code: return self.weight * hidden_states.to(input_dtype)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         to_12: "f32[2, 4, 4096]" = torch.ops.aten.to.dtype(mul_11, torch.float32);  mul_11 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         mul_12: "f32[2, 4, 4096]" = torch.ops.aten.mul.Tensor(model_norm_weight, to_12);  model_norm_weight = to_12 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:841 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         slice_8: "f32[2, 4, 4096]" = torch.ops.aten.slice.Tensor(mul_12, 0, 0, 9223372036854775807);  mul_12 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         slice_9: "f32[2, 4, 4096]" = torch.ops.aten.slice.Tensor(slice_8, 1, 0, 9223372036854775807);  slice_8 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         slice_10: "f32[2, 4, 4096]" = torch.ops.aten.slice.Tensor(slice_9, 2, 0, 9223372036854775807);  slice_9 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         linear_7: "f32[2, 4, 32000]" = torch.ops.aten.linear.default(slice_10, lm_head_weight);  slice_10 = lm_head_weight = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         return pytree.tree_unflatten((linear_7,), self._out_spec)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]     class submod_1(torch.nn.Module):
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]         def forward(self, b_model_rotary_emb_inv_freq: "f32[64]", unsqueeze: "i64[1, 4]"):
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]              # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:113 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             unsqueeze_1: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(b_model_rotary_emb_inv_freq, 0);  b_model_rotary_emb_inv_freq = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             slice_1: "f32[1, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 1, 0, 9223372036854775807);  unsqueeze_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             unsqueeze_2: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(slice_1, 2);  slice_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             to: "f32[1, 64, 1]" = torch.ops.aten.to.dtype(unsqueeze_2, torch.float32);  unsqueeze_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             expand: "f32[1, 64, 1]" = torch.ops.aten.expand.default(to, [1, -1, 1]);  to = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             to_1: "f32[1, 64, 1]" = torch.ops.aten.to.dtype_layout(expand, dtype = torch.float32, layout = torch.strided, device = device(type='meta'));  expand = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]              # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:114 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             slice_2: "i64[1, 4]" = torch.ops.aten.slice.Tensor(unsqueeze, 0, 0, 9223372036854775807);  unsqueeze = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             unsqueeze_3: "i64[1, 1, 4]" = torch.ops.aten.unsqueeze.default(slice_2, 1);  slice_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             slice_3: "i64[1, 1, 4]" = torch.ops.aten.slice.Tensor(unsqueeze_3, 2, 0, 9223372036854775807);  unsqueeze_3 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             to_2: "f32[1, 1, 4]" = torch.ops.aten.to.dtype(slice_3, torch.float32);  slice_3 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]              # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:118 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             to_3: "f32[1, 64, 1]" = torch.ops.aten.to.dtype(to_1, torch.float32);  to_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             to_4: "f32[1, 1, 4]" = torch.ops.aten.to.dtype(to_2, torch.float32);  to_2 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             matmul: "f32[1, 64, 4]" = torch.ops.aten.matmul.default(to_3, to_4);  to_3 = to_4 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             transpose: "f32[1, 4, 64]" = torch.ops.aten.transpose.int(matmul, 1, 2);  matmul = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]              # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:119 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             cat: "f32[1, 4, 128]" = torch.ops.aten.cat.default([transpose, transpose], -1);  transpose = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]              # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:120 in forward, code: cos = emb.cos() * self.attention_scaling
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             cos: "f32[1, 4, 128]" = torch.ops.aten.cos.default(cat)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             mul: "f32[1, 4, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]              # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:121 in forward, code: sin = emb.sin() * self.attention_scaling
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             sin: "f32[1, 4, 128]" = torch.ops.aten.sin.default(cat);  cat = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             mul_1: "f32[1, 4, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]              # File: /home/dinalt/.cache/huggingface/modules/transformers_modules/meta-llama--Llama-2-7b-hf/modeling_llama.py:123 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             to_5: "f32[1, 4, 128]" = torch.ops.aten.to.dtype(mul, torch.float32);  mul = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             to_6: "f32[1, 4, 128]" = torch.ops.aten.to.dtype(mul_1, torch.float32);  mul_1 = None
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             return (to_5, to_6)
V0610 18:40:25.800000 93824 torch/distributed/pipelining/_IR.py:690]             
V0610 18:40:25.804000 93824 torch/distributed/pipelining/_IR.py:726] Found pipe_split 0
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
Cell In[7], line 2
      1 #model.eval()
----> 2 pipe = build_pipeline(
      3     model,
      4     tuple(),
      5     dict(input_ids=input_ids),
      6     split_spec=split_spec,
      7 )

File ~/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_IR.py:1236, in pipeline(module, mb_args, mb_kwargs, split_spec, split_policy)
   1233 if split_spec is not None:
   1234     # Annotate split points in the module based on user spec
   1235     annotate_split_points(module, split_spec)
-> 1236     return Pipe.from_tracing(
   1237         mod=module,
   1238         example_args=mb_args,
   1239         example_kwargs=mb_kwargs,
   1240     )
   1241 else:
   1242     # Use split policy
   1243     return Pipe.from_tracing(
   1244         mod=module,
   1245         example_args=mb_args,
   1246         example_kwargs=mb_kwargs,
   1247         split_policy=split_policy,
   1248     )

File ~/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_IR.py:1049, in Pipe.from_tracing(mod, example_args, example_kwargs, split_policy)
   1042 # Trace with export
   1043 exported_program = Pipe._trace_with_export(
   1044     mod,
   1045     example_args,
   1046     example_kwargs,
   1047 )
-> 1049 pipe = Pipe._from_traced(
   1050     mod,
   1051     exported_program,
   1052     multi_use_param_spec,
   1053     output_loss_value_spec=output_loss_value_spec,
   1054     split_policy=split_policy,
   1055 )
   1057 # Users want the first pipeline stage to accept kwargs if the original
   1058 # program does. This is controlled by the `_codegen` field of the graph,
   1059 # so we make a copy here. Note: we only want the input spec and not the
   1060 # output spec, because the output spec is for the last stage. Maybe a
   1061 # TODO? Not sure yet.
   1062 split = pipe.split_gm

File ~/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_IR.py:749, in Pipe._from_traced(mod, exported_program, multi_use_param_spec, output_loss_value_spec, split_policy)
    747 for name, submodule in split.named_children():
    748     if isinstance(submodule, fx.GraphModule):
--> 749         new_submod = _outline_submodules(submodule.graph)
    750         # Replace old submod
    751         split.register_module(name, new_submod)

File ~/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_unflatten.py:27, in _outline_submodules(orig_graph)
     13 seen_attrs: dict[str, set[str]] = defaultdict(set)
     14 created_modules: dict[str, torch.nn.Module] = {}
     15 _ModuleFrame(
     16     orig_graph,
     17     tuple(orig_graph.nodes),
     18     seen_nodes,
     19     seen_modules,
     20     seen_attrs,
     21     created_modules,
     22     None,
     23     [("", None, 0)],
     24     "",
     25     {},
     26     module=new_module,
---> 27 ).run_outer()
     28 new_module.graph.lint()
     29 new_module.recompile()

File ~/.local/lib/python3.10/site-packages/torch/export/unflatten.py:1242, in _ModuleFrame.run_outer(self)
   1239     node_idx += 1
   1240     node = self.nodes[node_idx]
-> 1242 self.run_from(node_idx)
   1244 # Copy graph outputs
   1245 for node in self.flat_graph.nodes:

File ~/.local/lib/python3.10/site-packages/torch/export/unflatten.py:1258, in _ModuleFrame.run_from(self, node_idx)
   1256 while node_idx < len(self.nodes):
   1257     node = self.nodes[node_idx]
-> 1258     assert node.op != "placeholder"
   1260     self.print()
   1261     self.print("STEP", node_idx, node.format_node())

AssertionError: 
```

### Diagnostics

```python
# Rough outline of call graph
# Construct pipeline
pipeline()
https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/_IR.py#L1197

    # Apply splitpoint from split_spec
    annotate_split_points()
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/_IR.py#L1173

    # Build pipeline from annotated module
    Pipe.from_tracing()
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/_IR.py#L1020

        # Export module
        Pipe._trace_with_export()
        https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/_IR.py#L1020

            torch.export.export_for_training()

        # Split modules and modify for use-case
        Pipe._from_traced()
        https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/_IR.py#L1047C16-L1047C33

            # Get module from export
            traced = exported_program.module()

            # Split module at pipe_splits
            split = split_module(traced, mod, split_callback)

            # Unflatten
            new_submod = _outline_submodules(submodule.graph)

                https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/_unflatten.py#L8
                    # This does not appear to be a documented interface.
                    _Module_Frame.run_outer()
                    https://github.com/pytorch/pytorch/blob/main/torch/export/unflatten.py#L1259

                        self.run_from()
                        https://github.com/pytorch/pytorch/blob/main/torch/export/unflatten.py#L1282

                            #assertion failure
                            assert node.op != "placeholder"

# Root Cause
# run_outer searches for the last "placeholder" node index by stepping through the graph 
# until it finds a non-placeholder node, then calls run_from() with the given index.
# run_from() then asserts that none of the remaining nodes is a "placeholder," but
# this is clearly not the case.

    def run_outer(self):
        for i, node in enumerate(self.flat_graph.nodes):
            self.print(i, node.meta.get("nn_module_stack"), node.format_node())

        # Copy all graph inputs
        node_idx: int = 0
        node = self.nodes[node_idx]
        while node.op == "placeholder":
            self.copy_node(node)
            node_idx += 1
            node = self.nodes[node_idx]

        self.run_from(node_idx)

        # Copy graph outputs
        for node in self.flat_graph.nodes:
            if node.op == "output":
                self.copy_node(node)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run_from(self, node_idx):
        module_idx = 0
        # Walk through the graph, building up a new graph with the right submodules
        while node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            assert node.op != "placeholder"
```


With the Huggingface Llama model, this is what "split_module(traced, mod, split_callback)" returned:

```
split.submod_0.graph.print_tabular()

opcode         name                                            target                                          args                                                                               kwargs
-------------  ----------------------------------------------  ----------------------------------------------  ---------------------------------------------------------------------------------  ----------------------------------------------------
placeholder    model_embed_tokens_weight                       model_embed_tokens_weight                       ()                                                                                 {}
placeholder    input_ids                                       input_ids                                       ()                                                                                 {}

# There is a get_attr interleaved with the placeholders.
# With the above algorithm, it will stop iterating when this node is reached, but the call to
# run_from() will find another placeholder immediatly after it!
get_attr       submod_1                                        submod_1                                        ()                                                                                 {}
placeholder    model_rotary_emb_inv_freq                       model_rotary_emb_inv_freq                       ()                                                                                 {}
placeholder    model_layers_0_input_layernorm_weight           model_layers_0_input_layernorm_weight           ()                                                                                 {}
placeholder    model_layers_0_self_attn_q_proj_weight          model_layers_0_self_attn_q_proj_weight          ()                                                                                 {}
placeholder    model_layers_0_self_attn_k_proj_weight          model_layers_0_self_attn_k_proj_weight          ()                                                                                 {}
placeholder    model_layers_0_self_attn_v_proj_weight          model_layers_0_self_attn_v_proj_weight          ()                                                                                 {}
placeholder    model_layers_0_self_attn_o_proj_weight          model_layers_0_self_attn_o_proj_weight          ()                                                                                 {}
placeholder    model_layers_0_post_attention_layernorm_weight  model_layers_0_post_attention_layernorm_weight  ()                                                                                 {}
placeholder    model_layers_0_mlp_gate_proj_weight             model_layers_0_mlp_gate_proj_weight             ()                                                                                 {}
placeholder    model_layers_0_mlp_up_proj_weight               model_layers_0_mlp_up_proj_weight               ()                                                                                 {}
placeholder    model_layers_0_mlp_down_proj_weight             model_layers_0_mlp_down_proj_weight             ()                                                                                 {}
call_function  embedding_default                               aten.embedding.default                          (model_embed_tokens_weight, input_ids)                                             {}
call_function  arange_start                                    aten.arange.start                               (0, 4)                                                                             {'device': device(type='meta'), 'pin_memory': False}
call_function  unsqueeze_default                               aten.unsqueeze.default                          (arange_start, 0)                                                                  {}

# "submod_1" is used to implement the no_grad() context manager.
call_function  wrap_with_set_grad_enabled                      wrap_with_set_grad_enabled                      (False, submod_1, model_rotary_emb_inv_freq, unsqueeze_default)                    {}
...
```

## Minimal Reproduction

```python
import os

# Enabie pipeline debug output
os.environ["TORCH_LOGS"] = "+pp"

# Define a trivial module with a no_grad() sub-module.
import torch

class ConstNoiseModule(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        noise = torch.randn(d_model)
        self.register_buffer("noise", noise)

    # pipeline() works, if the following line is commented out and fails when present.
    @torch.no_grad()
    def forward(self, x):
        return x + self.noise

class Model(torch.nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.noise_module = ConstNoiseModule(d_model)
        self.layers = torch.nn.ModuleDict()
        self.layers = torch.nn.ModuleList([
             torch.nn.Linear(d_model, d_model)
             for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.noise_module(x)
        return x

# Build model and example input on the meta-device.
d_model = 16
n_layers = 2
batch_size = 2

with torch.device("meta"):
    model = Model(d_model, n_layers)
    example_input = torch.randn(batch_size, d_model)

print(model)

from torch.distributed.pipelining import SplitPoint, ScheduleGPipe, PipelineStage, pipe_split, pipeline

split_spec = {
    f"layers.0": SplitPoint.END
}

print(model)
print(example_input)
print(split_spec)

pipe = pipeline(
    model,
    (example_input,),
    split_spec=split_spec,
)
I0610 17:20:43.393000 93273 torch/distributed/pipelining/_IR.py:1003] Tracing model ...
Model(
  (noise_module): ConstNoiseModule()
  (layers): ModuleList(
    (0-1): 2 x Linear(in_features=16, out_features=16, bias=True)
  )
)
tensor(..., device='meta', size=(2, 16))
{'layers.0': <SplitPoint.END: 2>}
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690] class GraphModule(torch.nn.Module):
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]     def forward(self, x):
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         x: "f32[2, 16]"; 
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]     
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         # No stacktrace found for following nodes
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         layers_0_weight: "f32[16, 16]" = getattr(self.layers, "0").weight
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         layers_0_bias: "f32[16]" = getattr(self.layers, "0").bias
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         layers_1_weight: "f32[16, 16]" = getattr(self.layers, "1").weight
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         layers_1_bias: "f32[16]" = getattr(self.layers, "1").bias
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         noise_module_noise: "f32[16]" = self.noise_module.noise
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_IR.py:1170 in _split_after_forward, code: return self._orig_forward(*args, **kwargs)
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         linear: "f32[2, 16]" = torch.ops.aten.linear.default(x, layers_0_weight, layers_0_bias);  x = layers_0_weight = layers_0_bias = None
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]          # File: /home/dinalt/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_IR.py:343 in pipe_split, code: return torch.ops.pippy._pipe_split()
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         _pipe_split = torch.ops.pippy._pipe_split.default();  _pipe_split = None
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         # No stacktrace found for following nodes
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         submod_5 = self.submod_1
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         add = torch.ops.higher_order.wrap_with_set_grad_enabled(False, submod_5, linear, noise_module_noise);  submod_5 = linear = None
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]          # File: /tmp/ipykernel_93273/2789375249.py:13 in forward, code: return x + self.noise
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         getitem: "f32[2, 16]" = add[0];  add = None
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]          # File: /tmp/ipykernel_93273/2789375249.py:27 in forward, code: x = layer(x)
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         linear_1: "f32[2, 16]" = torch.ops.aten.linear.default(getitem, layers_1_weight, layers_1_bias);  getitem = layers_1_weight = layers_1_bias = None
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         # No stacktrace found for following nodes
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         submod_6 = self.submod_3
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         add_1 = torch.ops.higher_order.wrap_with_set_grad_enabled(False, submod_6, linear_1, noise_module_noise);  submod_6 = linear_1 = noise_module_noise = None
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]          # File: /tmp/ipykernel_93273/2789375249.py:13 in forward, code: return x + self.noise
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         getitem_1: "f32[2, 16]" = add_1[0];  add_1 = None
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         return pytree.tree_unflatten((getitem_1,), self._out_spec)
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]     class submod_1(torch.nn.Module):
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         def forward(self, linear: "f32[2, 16]", b_noise_module_noise: "f32[16]"):
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]              # File: /tmp/ipykernel_93273/2789375249.py:13 in forward, code: return x + self.noise
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]             add: "f32[2, 16]" = torch.ops.aten.add.Tensor(linear, b_noise_module_noise);  linear = b_noise_module_noise = None
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]             return (add,)
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]             
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]     class submod_3(torch.nn.Module):
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]         def forward(self, linear_1: "f32[2, 16]", b_noise_module_noise: "f32[16]"):
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]              # File: /tmp/ipykernel_93273/2789375249.py:13 in forward, code: return x + self.noise
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]             add_1: "f32[2, 16]" = torch.ops.aten.add.Tensor(linear_1, b_noise_module_noise);  linear_1 = b_noise_module_noise = None
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]             return (add_1,)
V0610 17:20:43.679000 93273 torch/distributed/pipelining/_IR.py:690]             
V0610 17:20:43.680000 93273 torch/distributed/pipelining/_IR.py:726] Found pipe_split 0
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
Cell In[5], line 11
      8 print(example_input)
      9 print(split_spec)
---> 11 pipe = pipeline(
     12     model,
     13     (example_input,),
     14     split_spec=split_spec,
     15 )

File ~/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_IR.py:1236, in pipeline(module, mb_args, mb_kwargs, split_spec, split_policy)
   1233 if split_spec is not None:
   1234     # Annotate split points in the module based on user spec
   1235     annotate_split_points(module, split_spec)
-> 1236     return Pipe.from_tracing(
   1237         mod=module,
   1238         example_args=mb_args,
   1239         example_kwargs=mb_kwargs,
   1240     )
   1241 else:
   1242     # Use split policy
   1243     return Pipe.from_tracing(
   1244         mod=module,
   1245         example_args=mb_args,
   1246         example_kwargs=mb_kwargs,
   1247         split_policy=split_policy,
   1248     )

File ~/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_IR.py:1049, in Pipe.from_tracing(mod, example_args, example_kwargs, split_policy)
   1042 # Trace with export
   1043 exported_program = Pipe._trace_with_export(
   1044     mod,
   1045     example_args,
   1046     example_kwargs,
   1047 )
-> 1049 pipe = Pipe._from_traced(
   1050     mod,
   1051     exported_program,
   1052     multi_use_param_spec,
   1053     output_loss_value_spec=output_loss_value_spec,
   1054     split_policy=split_policy,
   1055 )
   1057 # Users want the first pipeline stage to accept kwargs if the original
   1058 # program does. This is controlled by the `_codegen` field of the graph,
   1059 # so we make a copy here. Note: we only want the input spec and not the
   1060 # output spec, because the output spec is for the last stage. Maybe a
   1061 # TODO? Not sure yet.
   1062 split = pipe.split_gm

File ~/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_IR.py:749, in Pipe._from_traced(mod, exported_program, multi_use_param_spec, output_loss_value_spec, split_policy)
    747 for name, submodule in split.named_children():
    748     if isinstance(submodule, fx.GraphModule):
--> 749         new_submod = _outline_submodules(submodule.graph)
    750         # Replace old submod
    751         split.register_module(name, new_submod)

File ~/.local/lib/python3.10/site-packages/torch/distributed/pipelining/_unflatten.py:27, in _outline_submodules(orig_graph)
     13 seen_attrs: dict[str, set[str]] = defaultdict(set)
     14 created_modules: dict[str, torch.nn.Module] = {}
     15 _ModuleFrame(
     16     orig_graph,
     17     tuple(orig_graph.nodes),
     18     seen_nodes,
     19     seen_modules,
     20     seen_attrs,
     21     created_modules,
     22     None,
     23     [("", None, 0)],
     24     "",
     25     {},
     26     module=new_module,
---> 27 ).run_outer()
     28 new_module.graph.lint()
     29 new_module.recompile()

File ~/.local/lib/python3.10/site-packages/torch/export/unflatten.py:1242, in _ModuleFrame.run_outer(self)
   1239     node_idx += 1
   1240     node = self.nodes[node_idx]
-> 1242 self.run_from(node_idx)
   1244 # Copy graph outputs
   1245 for node in self.flat_graph.nodes:

File ~/.local/lib/python3.10/site-packages/torch/export/unflatten.py:1258, in _ModuleFrame.run_from(self, node_idx)
   1256 while node_idx < len(self.nodes):
   1257     node = self.nodes[node_idx]
-> 1258     assert node.op != "placeholder"
   1260     self.print()
   1261     self.print("STEP", node_idx, node.format_node())

AssertionError: 
```

Removing the no_grad() decorator resolves the issue.

Reproduce key steps in "pipeline()"

```python
from torch.distributed.pipelining import SplitPoint, ScheduleGPipe, PipelineStage, pipe_split, pipeline
from types import MethodType

def _split_before_forward(self, *args, **kwargs):
    pipe_split()
    return self._orig_forward(*args, **kwargs)


def _split_after_forward(self, *args, **kwargs):
    try:
        return self._orig_forward(*args, **kwargs)
    finally:
        pipe_split()

def annotate_split_points(mod: torch.nn.Module, spec: dict[str, SplitPoint]):
    # TODO: make this implementation out-of-place?
    for qualname, split_type in spec.items():
        atoms = qualname.split(".")
        predecessor_module = mod
        for i, atom in enumerate(atoms[:-1]):
            try:
                predecessor_module = getattr(predecessor_module, atom)
            except AttributeError as e:
                raise AttributeError(
                    f"Specified target {qualname} referenced "
                    f"nonexistent module {'.'.join(atoms[: i + 1])}"
                ) from e

        mod_to_wrap = getattr(predecessor_module, atoms[-1])
        mod_to_wrap._orig_forward = mod_to_wrap.forward
        if split_type == SplitPoint.BEGINNING:
            print("Modified begin")
            mod_to_wrap.forward = MethodType(_split_before_forward, mod_to_wrap)
        elif split_type == SplitPoint.END:
            print("Modified end")
            mod_to_wrap.forward = MethodType(_split_after_forward, mod_to_wrap)
        else:
            raise ValueError("Unknown split point type.")

split_spec = {
    f"layers.0": SplitPoint.END
}

annotate_split_points(model, split_spec)

# Manually follow the steps from https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/_IR.py#L666

# Export the module
ep = torch.export.export_for_training(
    model,
    (example_input,),
)
traced = ep.module()
traced.graph.print_tabular()

opcode         name                target                       args                                             kwargs
-------------  ------------------  ---------------------------  -----------------------------------------------  --------
get_attr       layers_0_weight     layers.0.weight              ()                                               {}
get_attr       layers_0_bias       layers.0.bias                ()                                               {}
get_attr       layers_1_weight     layers.1.weight              ()                                               {}
get_attr       layers_1_bias       layers.1.bias                ()                                               {}
get_attr       noise_module_noise  noise_module.noise           ()                                               {}
placeholder    x                   x                            ()                                               {}
call_function  linear              aten.linear.default          (x, layers_0_weight, layers_0_bias)              {}
call_function  _pipe_split         pippy._pipe_split.default    ()                                               {}
get_attr       submod_5            submod_1                     ()                                               {}
call_function  add                 wrap_with_set_grad_enabled   (False, submod_5, linear, noise_module_noise)    {}
call_function  getitem             <built-in function getitem>  (add, 0)                                         {}
call_function  linear_1            aten.linear.default          (getitem, layers_1_weight, layers_1_bias)        {}
get_attr       submod_6            submod_3                     ()                                               {}
call_function  add_1               wrap_with_set_grad_enabled   (False, submod_6, linear_1, noise_module_noise)  {}
call_function  getitem_1           <built-in function getitem>  (add_1, 0)                                       {}
output         output              output                       ((getitem_1,),)                                  {}
```

```python
from torch.fx.passes.split_module import split_module
aten_pipe_split_alias = torch.ops.pippy._pipe_split.default

part_idx = 0

def split_callback(n):
    global part_idx
    if (n.op, n.target) == (
        "call_function",
        aten_pipe_split_alias,
    ):
        print(f"Found pipe_split {part_idx}")
        part_idx += 1
    return part_idx

split = split_module(traced, model, split_callback)
print("split\n")
split.graph.print_tabular()
print("\nsubmod_0\n")
split.submod_0.graph.print_tabular()
print("\nsubmod_1\n")
split.submod_1.graph.print_tabular()

Found pipe_split 0
split

opcode       name                target              args                                                            kwargs
-----------  ------------------  ------------------  --------------------------------------------------------------  --------
get_attr     layers_0_weight     layers.0.weight     ()                                                              {}
get_attr     layers_0_bias       layers.0.bias       ()                                                              {}
get_attr     layers_1_weight     layers.1.weight     ()                                                              {}
get_attr     layers_1_bias       layers.1.bias       ()                                                              {}
get_attr     noise_module_noise  noise_module.noise  ()                                                              {}
placeholder  x                   x                   ()                                                              {}
get_attr     submod_1            submod_1            ()                                                              {}
get_attr     submod_3            submod_3            ()                                                              {}
call_module  submod_0            submod_0            (x, layers_0_weight, layers_0_bias)                             {}
call_module  submod_2            submod_1            (submod_0, noise_module_noise, layers_1_weight, layers_1_bias)  {}
output       output              output              ((submod_2,),)                                                  {}

submod_0

opcode         name             target               args                                 kwargs
-------------  ---------------  -------------------  -----------------------------------  --------
placeholder    x                x                    ()                                   {}
placeholder    layers_0_weight  layers_0_weight      ()                                   {}
placeholder    layers_0_bias    layers_0_bias        ()                                   {}
call_function  linear_default   aten.linear.default  (x, layers_0_weight, layers_0_bias)  {}
output         output           output               (linear_default,)                    {}

submod_1

opcode         name                          target                       args                                                   kwargs
-------------  ----------------------------  ---------------------------  -----------------------------------------------------  --------
get_attr       submod_1                      submod_1                     ()                                                     {}
placeholder    linear                        linear                       ()                                                     {}
placeholder    noise_module_noise            noise_module_noise           ()                                                     {}
placeholder    layers_1_weight               layers_1_weight              ()                                                     {}
placeholder    layers_1_bias                 layers_1_bias                ()                                                     {}
get_attr       submod_3                      submod_3                     ()                                                     {}
call_function  _pipe_split_default           pippy._pipe_split.default    ()                                                     {}
call_function  wrap_with_set_grad_enabled    wrap_with_set_grad_enabled   (False, submod_1, linear, noise_module_noise)          {}
call_function  getitem                       <built-in function getitem>  (wrap_with_set_grad_enabled, 0)                        {}
call_function  linear_default                aten.linear.default          (getitem, layers_1_weight, layers_1_bias)              {}
call_function  wrap_with_set_grad_enabled_1  wrap_with_set_grad_enabled   (False, submod_3, linear_default, noise_module_noise)  {}
call_function  getitem_1                     <built-in function getitem>  (wrap_with_set_grad_enabled_1, 0)                      {}
output         output                        output                       (getitem_1,)                                           {}
```

In the above output for submod_1, we can see that "split" producted a "get_attr" before all of the "placeholder" ops.

When this reaches:
new_submod = _outline_submodules(submodule.graph)

"run_outer" will stop graph traversal as the first node and call "run_from()". As this is not the last "placeholder" op, the assert is tripped.

I would produce a fix for this myself, except it's not clear who is at fault? Is it expected that "placeholder" nodes always come first, as is assumed by run_outer(), in which case, the implementation of "split_module()" shoud be fixed? Or, is this assumption incorrect, in which case "export/unflatten.py" should be fixed.

It does not help that pipelining is calling a private API of "unflatten," which seems to have next to zero documentation. Perhaps "_IR.py" should not be calling private code in "unflatten," and a solution should be in "_IR.py."

