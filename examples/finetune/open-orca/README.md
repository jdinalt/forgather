# Open-Orca Finetune

Fine-tune a causal language model on the [Open-Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
reasoning dataset using the `finetune_v2` project template and the new
fast-loading packed dataset variant.

Open-Orca is a large collection of augmented FLAN examples distilled through
GPT-4 / GPT-3.5, heavy on chain-of-thought explanations and structured
reasoning prompts. Fine-tuning on it teaches a base LM to produce longer,
more step-by-step responses than a chat-persona dataset would — it's the
natural complement to the [Samantha example](../samantha/README.md), which
teaches conversational style rather than reasoning.

## What's different from the Samantha example

If you've already worked through the Samantha tutorial, the quick tour:

- **Dataset: packed Open-Orca (`openorca-packed.yaml`).** The packed variant
  uses best-fit sequence packing so sequences reach `max_length` without
  relying on padding — substantially better throughput than the padded
  variant whenever the batch size is greater than 1. The dataset was
  recently modernised to use Forgather's fast iterable-dataset loader, so
  initialisation takes seconds rather than the 10+ minutes it took
  historically.
- **Headline experiment is a real run, not a smoke test.** The
  `llama3_1b/4gpu_ddp.yaml` config is sized to burn through ~1 billion
  tokens (roughly half an epoch of packed Open-Orca at seq_len 2048) in
  about 11 hours on the reference 4x 4090 box, leaving plenty of headroom
  inside a 24-hour budget. See *Headline Run* below for the exact command
  and what it produced.
- **Trimmed config set.** Four configs total, one per
  (model-size, execution-mode) combination that actually gets used:
  `llama3_1b/{1gpu_default,4gpu_ddp}` and `llama2_7b/{1gpu_default,4gpu_pp}`.
  No 2-GPU PP variants, no FSDP2 variant — Samantha already covers those.
- **WSD learning-rate schedule.** Inherited from `finetune_v2.yaml`: linear
  warmup, stable LR, then cosine decay over `annealing_tokens`. The decay
  phase doesn't fire automatically in a token-budget-bounded run; use the
  `--start-annealing` flag at launch or `forgather control save-stop` to
  trigger it near the end.

## Setup

You will need:

1. A **base model** in Forgather format. Any of the models used by the
   Samantha tutorial will work — the two I tested against are `fg_llama_7b`
   (a conversion of `meta-llama/Llama-2-7b-hf`) and `fg_llama_1b` (a
   conversion of `meta-llama/Llama-3.2-1B`). Both of these are **raw
   pretrained base models**, not instruction-tuned. Neither has a native
   chat template — ChatML (`<|im_start|>` / `<|im_end|>`) is added
   explicitly during the Forgather conversion step per the instructions in
   the Samantha project, along with a few extra special tokens. This is
   actually what makes Open-Orca interesting as a first fine-tune target:
   the base model has never seen a chat format before, and it has to learn
   the ChatML turn structure *and* the reasoning style in the same run.
   Expect the initial loss to be noticeably higher than it would be on an
   already instruction-tuned checkpoint, and to drop sharply in the first
   few thousand steps as the model picks up the turn structure.
2. The **Forgather repo checked out**; this project's `meta.yaml` expects
   the `forgather_workspace` sibling directory for template search paths.
3. A working **Open-Orca dataset** download path. The dataset lives at
   `examples/datasets/Open-Orca/` inside this repo and is pointed at by
   the base project template automatically.

If you haven't already converted a model to Forgather's format, follow the
*Download a Model / Convert the Model* section of the
[Samantha README](../samantha/README.md#setup) — the tooling and the target
model formats are identical.

```bash
# This tutorial uses environment variables to keep the commands short.
# Adjust to your paths.
export FG_MODEL=/path/to/fg_llama_1b
```

## Configurations

| Config | Model | GPUs | Trainer | seq_len | bs/dev | Intended use |
|---|---|---|---|---|---|---|
| [`llama3_1b/1gpu_default.yaml`](./templates/configs/llama3_1b/1gpu_default.yaml) | Llama3 1B | 1 | basic | 2048 | 4 | Iteration / smoke testing |
| [`llama3_1b/4gpu_ddp.yaml`](./templates/configs/llama3_1b/4gpu_ddp.yaml) | Llama3 1B | 4 | ddp | 2048 | 4 | **Headline run** — real training |
| [`llama2_7b/1gpu_default.yaml`](./templates/configs/llama2_7b/1gpu_default.yaml) | Llama2 7B | 1 | basic | 1536 | 1 | Verification only |
| [`llama2_7b/4gpu_pp.yaml`](./templates/configs/llama2_7b/4gpu_pp.yaml) | Llama2 7B | 4 | pipeline (ZBV) | 2048 | 2 | Real 7B training target |

Each config extends `templates/open_orca.yaml`, which in turn extends
`templatelib/examples/projects/finetune_v2.yaml`. Token budgets, warmup, and
annealing windows are set in the individual configs rather than on the
command line so that a run is fully reproducible from the config template
alone — override them via CLI flags only when you're iterating.

## Smoke Testing Before Committing to a Long Run

The user-visible EOS-token / chat-template failure modes are silent until
you try to generate from the resulting model. Always verify the pipeline
end-to-end with a short training + inference round-trip before launching
anything that takes hours:

```bash
# 1. Train a handful of steps, save a checkpoint.
forgather -t llama3_1b/4gpu_ddp.yaml train \
    --save-strategy steps --max-steps 30 --step-cadence 0.001 \
    -M "${FG_MODEL}" -d 0,1,3,4

# 2. Start the inference server against the saved checkpoint.
forgather inf server -c -m "${FG_MODEL}"

# 3. In another terminal, send a test message.
forgather inf client --message "Hello! What is 2+2?" --max-tokens 100
```

What you're looking for in step 3:

- The model **responds in English** (chat template was applied correctly).
- The response **stops after a few lines** rather than running to the full
  `--max-tokens` budget (EOS token is being respected — the server's
  startup log should show `Stop token IDs: [128256]` for a ChatML model).
- No Python tracebacks in either the training or inference logs.

The "correct answer quality" is not the test here — 30 steps of training
won't fix a 1B base model's math ability. You're verifying that the
plumbing works. If any of the three checks above fail, stop and
investigate before committing to a long run; retraining a 12-hour run to
fix a silent EOS bug is exactly the frustration this section exists to
prevent.

## Headline Run: Llama3 1B on 4 GPUs

The experiment: fine-tune `fg_llama_1b` (Llama-3.2-1B base, pretrained-only,
converted to Forgather's format with ChatML added) on packed Open-Orca for
1 billion tokens — roughly 1/8 of an epoch of the packed dataset — using
DDP across 4 GPUs. 1 billion tokens is enough to get well past the
"learning the chat format" phase and into actually learning content. Target
wall-clock is under 24 hours on the reference 4x 4090 box (each card
power-limited to 250 W, GPU 2 excluded for thermals).

Measured numbers from the first two log intervals of the run that produced
this README:

| Metric | Value |
|---|---|
| Total steps | 32,124 (at `total_tokens = 1000`, `seq_len = 2048`, global batch = 16) |
| Steady-state throughput | **~26,800 tokens/second** across 4 ranks |
| Peak memory | 15.49 GiB per rank (8 GiB headroom below the 24 GiB ceiling) |
| Loss at step 32 | 2.17 |
| Loss at step 64 | 1.92 |
| Projected wall-clock | ~10.5 hours |

That initial 0.24-nat drop between step 32 and step 64 is the base model
eating the chat format: it's never seen `<|im_start|>` / `<|im_end|>` before
this run, and picking up the turn structure is the cheapest loss in
training. Expect the curve to flatten meaningfully once you're past the
first few hundred steps.

**Important gotcha about the token budget.** The `finetune_v2.yaml` base
template defaults `max_steps` to `-1`, which means "train for
`num_train_epochs = 1` worth of data". For a dataset the size of packed
Open-Orca (~8 billion tokens) that's ~87 hours on the reference hardware
— far beyond any 24-hour budget. The `open_orca.yaml` base config in this
project rebinds `max_steps` to `ns.total_steps` so that the `total_tokens`
parameter actually bounds training. If you copy these configs into a
project that extends `finetune_v2.yaml` directly, remember to add the
same override or your "1B token run" will silently become an "8B token
run".

```bash
# One-time setup: stage a clean copy of the base model as the run output
# directory, so the run state (checkpoints, logs) lives separately from
# the source model.
OO_RUN=/home/dinalt/my_first_model/openorca_llama3_1b_run
cp -a "${FG_MODEL}/." "${OO_RUN}/"

# Launch the run. 'nohup ... &' detaches the process so it survives the
# terminal closing; output goes to long_run.log inside the run dir.
nohup forgather -t llama3_1b/4gpu_ddp.yaml train \
    -M "${OO_RUN}" -d 0,1,3,4 \
    > "${OO_RUN}/long_run.log" 2>&1 &
disown
```

Monitoring:

```bash
# Follow the live log
tail -F "${OO_RUN}/long_run.log"

# List running Forgather jobs (uses the trainer control interface)
forgather control list

# Inspect the training metrics JSON
forgather logs summary "${OO_RUN}/runs"/*/trainer_logs.json

# Plot the loss curve
forgather logs plot --loss-curves "${OO_RUN}/runs"/*/trainer_logs.json
```

### Triggering the annealing phase

The config uses `finetune_v2.yaml`'s `WSDScheduler` — linear warmup, then a
stable LR, then a cosine decay to `min_lr` over `annealing_tokens` (200M
in this config). In a token-budget-bounded run the decay phase does not
fire on its own; trigger it explicitly when you want to wind the run down:

```bash
# Option 1: bake it into the launch. Start the decay from step 0 -- use
# this when you know the total token budget up front and just want a
# standard cosine-decay schedule.
forgather -t llama3_1b/4gpu_ddp.yaml train --start-annealing \
    -M "${OO_RUN}" -d 0,1,3,4

# Option 2: trigger it mid-run via the control interface. Do this when
# you want to react to the loss curve -- e.g., when eval loss plateaus,
# tell the running job to save a checkpoint and begin annealing. The
# trainer will finish the decay and stop.
forgather control list              # find the job id
forgather control save JOB_ID       # save a pre-anneal checkpoint (optional)
# ... in practice, option 2 requires a control-callback hook to flip
# start_annealing on the running job. For a simpler path, stop the job
# with `forgather control save-stop`, then resume with --start-annealing
# from the last checkpoint.
```

If you just want the model to finish training at a low LR without thinking
about any of this, launch with `--start-annealing` and let the WSDScheduler
handle the decay over the last 200M tokens of the budget.

## Serving the Fine-Tuned Model

After the headline run finishes, the latest checkpoint lives at
`${OO_RUN}/checkpoints/checkpoint-N/` (where N is the final step count).
Point the inference server at the run directory with `-c` to auto-select
the latest checkpoint:

```bash
forgather inf server -c -m "${OO_RUN}"
```

A reasoning prompt that exercises the kind of response Open-Orca training
should reinforce:

```bash
forgather inf client --message "Think step by step: a farmer has 17 sheep. All but 9 run away. How many sheep are left?" --max-tokens 200
```

The [Samantha README](../samantha/README.md#testing-the-finetuned-model)
documents the full set of inference CLI options and the interactive chat
mode — the same tooling applies here.

## References

- Dataset: <https://huggingface.co/datasets/Open-Orca/OpenOrca>
- Forgather base templates:
  [finetune_v2.yaml](../../../templatelib/examples/projects/finetune_v2.yaml)
  → [lm_training_project.yaml](../../../templatelib/examples/projects/lm_training_project.yaml)
- LM Training Project documentation:
  [docs/project-templates/lm-training-projects.md](../../../docs/project-templates/lm-training-projects.md)
- Chat template: [chat_templates/chatml.jinja](../../../chat_templates/chatml.jinja)
- WSDScheduler theory:
  *Understanding Warmup-Stable-Decay Learning Rates* (Wen et al. 2024),
  <https://arxiv.org/abs/2410.05192>
