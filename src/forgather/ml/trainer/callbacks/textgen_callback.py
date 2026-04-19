import os
from typing import List, Optional

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from transformers import GenerationConfig, StoppingCriteria

from forgather.ml.trainer.logging import format_train_info

from ..trainer_types import TrainerCallback


class TextgenCallback(TrainerCallback):
    """
    Periodically generate and log text from a set of prompts for subjective model evaluation.

    Automatically dispatches between single-rank generation (via model.generate()) and
    pipeline-parallel generation (via trainer.pipeline_generate()) based on whether the
    trainer exposes a pipeline_generate method. The same callback works unchanged with
    SimpleTrainer, AccelTrainer/DDPTrainer, and PipelineTrainer.
    """

    # Stride is the number of steps between text generations
    def __init__(
        self,
        summary_writer: SummaryWriter,
        prompts: List[str] | str,
        generation_config: Optional[dict] = None,
        generation_steps: Optional[int] = None,
        max_new_tokens: int = 200,
    ):
        """
        Periodically generates and logs text from a set a prompts for subjective model evaluation

        This may only trigger on model evaluation steps, which establishes the minimum interval between generations.

        args:
            summary_writer: The Tensor Board SummaryWriter to log to.
            prompts: Either a list of prompts (List[str]) or a path to a YAML file, defining a list of prompts.
            generation_config: A dictionary with arguments to HF GenerationConfig
            generation_steps: The number of steps between generations. If None, it defaults to eval_steps
            max_new_tokens: The maximum new tokens to generate for each prompt.
        """
        super().__init__()
        self.summary_writer = summary_writer
        if isinstance(prompts, list):
            self.prompts = prompts
        else:
            if not isinstance(prompts, str):
                raise ValueError(
                    f"'prompts' must be List[str] | str, found {type(prompts)}"
                )
            with open(prompts, "r") as file:
                self.prompts = yaml.safe_load(file)

            if not isinstance(self.prompts, list):
                raise ValueError(
                    f"From file {prompts}, expected 'prompts' to be a list but found {type(self.prompts)}"
                )

        for s in self.prompts:
            if not isinstance(s, str):
                raise ValueError(
                    f"Expected all prompts to be strings, but found {type(s)}"
                )

        # To construct GenerationConfig, we need token ids from the model or tokenizer
        # We don't have these here, so defer construction until callback.
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
        # Pipeline trainers expose pipeline_generate() and require collective participation
        # from all ranks. Single-rank trainers (including DDP) use model.generate() on rank 0.
        if hasattr(trainer, "pipeline_generate"):
            self._on_evaluate_pipeline(args, state, control, **kwargs)
        else:
            self._on_evaluate_single_rank(args, state, control, **kwargs)

    def _on_evaluate_single_rank(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        processing_class = kwargs.get("processing_class")
        if self.generation_steps is None:
            self.generation_steps = args.eval_steps
        if not state.is_world_process_zero or state.global_step < self.next_gen_step:
            return
        self.next_gen_step += self.generation_steps
        text = ""
        for output in self.generate(args.device, model, processing_class):
            text += output + "\n\n---\n\n"
        self.summary_writer.add_text("eval-text", text, global_step=state.global_step)
        self.summary_writer.flush()

    def _on_evaluate_pipeline(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        processing_class = kwargs.get("processing_class")

        if self.generation_steps is None:
            self.generation_steps = args.eval_steps

        # Coordinate "should we generate this step?" across all ranks.
        # Only rank 0 has the authoritative next_gen_step counter.
        should_gen = torch.tensor(
            [
                int(
                    state.is_world_process_zero
                    and state.global_step >= self.next_gen_step
                )
            ],
            dtype=torch.long,
            device=args.device,
        )
        torch.distributed.broadcast(should_gen, src=0)
        if not should_gen.item():
            return

        if state.is_world_process_zero:
            self.next_gen_step += self.generation_steps

        # Rank 0 tokenizes; broadcast shape then ids to all ranks.
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

        # All ranks generate together.
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
            texts = processing_class.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            body = ""
            for prompt, decoded in zip(self.prompts, texts):
                body += (
                    prompt + " [START] " + decoded[len(prompt) + 1 :] + "\n\n---\n\n"
                )
            self.summary_writer.add_text(
                "eval-text", body, global_step=state.global_step
            )
            self.summary_writer.flush()

    def generate(self, device, model, tokenizer):
        generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            **self.gen_config_args,
        )

        tokenizer_outputs = tokenizer(
            self.prompts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )

        # Temporarily remove torch.compile from the model for generation.
        # The compiled model may use flex_attention + max-autotune which fails
        # during the generate loop (different shapes/cache behavior than training).
        # Saving and restoring _compiled_call_impl reverts to eager for this call.
        compiled_call = getattr(model, "_compiled_call_impl", None)
        if compiled_call is not None:
            model._compiled_call_impl = model._call_impl

        input_ids = tokenizer_outputs["input_ids"].to(device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
                tokenizer=tokenizer,
            )

        # Restore compiled forward for training
        if compiled_call is not None:
            model._compiled_call_impl = compiled_call

        output_text = tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )

        for prompt, y in zip(self.prompts, output_text):
            s = prompt + " [START] " + y[len(prompt) + 1 :]
            yield s
