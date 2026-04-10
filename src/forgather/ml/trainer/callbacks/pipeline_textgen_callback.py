from typing import List, Optional

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from ..trainer_types import TrainerCallback


class PipelineTextgenCallback(TrainerCallback):
    """
    Textgen callback for PipelineTrainer.

    Periodically generates text samples from a set of prompts during pipeline-parallel
    training. Unlike TextgenCallback, all ranks participate cooperatively so that
    activations can be passed between pipeline stages.

    Only rank 0 tokenizes prompts and logs output to TensorBoard; all ranks call
    trainer.pipeline_generate() together.

    This callback has the same constructor signature as TextgenCallback and can be used
    as a drop-in replacement when training with PipelineTrainer.

    Args:
        summary_writer: TensorBoard SummaryWriter to log generated text to.
        prompts: List of prompt strings, or path to a YAML file containing a list.
        generation_config: Dict of arguments forwarded to pipeline_generate() (e.g.
            do_sample, top_k, temperature, repetition_penalty).
        generation_steps: Minimum number of training steps between generations.
            Defaults to eval_steps if not set.
        max_new_tokens: Maximum number of new tokens to generate per prompt.
    """

    def __init__(
        self,
        summary_writer: SummaryWriter,
        prompts: List[str] | str,
        generation_config: Optional[dict] = None,
        generation_steps: Optional[int] = None,
        max_new_tokens: int = 200,
    ):
        super().__init__()
        self.summary_writer = summary_writer

        if isinstance(prompts, list):
            self.prompts = prompts
        else:
            if not isinstance(prompts, str):
                raise ValueError(
                    f"'prompts' must be List[str] | str, found {type(prompts)}"
                )
            with open(prompts, "r") as f:
                self.prompts = yaml.safe_load(f)
            if not isinstance(self.prompts, list):
                raise ValueError(
                    f"From file {prompts}, expected 'prompts' to be a list but found "
                    f"{type(self.prompts)}"
                )

        for s in self.prompts:
            if not isinstance(s, str):
                raise ValueError(
                    f"Expected all prompts to be strings, but found {type(s)}"
                )

        if generation_config is None:
            self.gen_config_args = dict(
                do_sample=True,
                top_k=20,
                temperature=0.7,
                repetition_penalty=1.15,
            )
        else:
            self.gen_config_args = generation_config

        self.generation_steps = generation_steps
        self.max_new_tokens = max_new_tokens
        self.next_gen_step = 0

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        processing_class = kwargs.get("processing_class")

        if not hasattr(trainer, "pipeline_generate"):
            return  # not a pipeline trainer; silently skip

        if self.generation_steps is None:
            self.generation_steps = args.eval_steps

        # All ranks must agree on whether to generate this step.
        # Only rank 0 has the authoritative next_gen_step counter.
        should_gen = torch.tensor(
            [int(state.is_world_process_zero and state.global_step >= self.next_gen_step)],
            dtype=torch.long,
            device=args.device,
        )
        torch.distributed.broadcast(should_gen, src=0)
        if not should_gen.item():
            return

        if state.is_world_process_zero:
            self.next_gen_step += self.generation_steps

        # Rank 0 tokenizes; broadcast shape and ids to all ranks.
        if state.is_world_process_zero:
            enc = processing_class(
                self.prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                padding_side="left",
            )
            input_ids = enc["input_ids"].to(args.device)
            shape_t = torch.tensor(list(input_ids.shape), device=args.device)
        else:
            shape_t = torch.zeros(2, dtype=torch.long, device=args.device)

        torch.distributed.broadcast(shape_t, src=0)

        if not state.is_world_process_zero:
            input_ids = torch.zeros(
                shape_t.tolist(), dtype=torch.long, device=args.device
            )
        torch.distributed.broadcast(input_ids, src=0)

        # All ranks generate together via the pipeline.
        gen_config = dict(
            eos_token_id=processing_class.eos_token_id,
            pad_token_id=processing_class.pad_token_id or processing_class.eos_token_id,
            **self.gen_config_args,
        )
        generated_ids = trainer.pipeline_generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            **gen_config,
        )

        # Only rank 0 decodes and logs.
        if state.is_world_process_zero:
            texts = processing_class.batch_decode(generated_ids, skip_special_tokens=True)
            body = ""
            for prompt, decoded in zip(self.prompts, texts):
                body += prompt + " [START] " + decoded[len(prompt) + 1:] + "\n\n---\n\n"
            self.summary_writer.add_text("eval-text", body, global_step=state.global_step)
            self.summary_writer.flush()
