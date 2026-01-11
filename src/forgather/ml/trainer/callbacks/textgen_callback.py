import os
from typing import List, Optional

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from transformers import GenerationConfig, StoppingCriteria

from forgather.ml.trainer.logging import format_train_info

from ..trainer_types import TrainerCallback


# Stop generation after all batch elements have generated an EOS token.
# Stores the index of the first generated EOS token for each batch element in "self.eos_index,"
# which can be used to slice off whatever extra junk was generated after it.
# Note: This is a stateful object. A new instance should be created for each call to generate().
class EosStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.eos_token = tokenizer.eos_token_id
        self.done = None
        self.eos_index = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        batch_size, seq_len = input_ids.shape

        # Lazy construct a bool state for each batch element
        if self.done == None:
            self.done = torch.zeros(
                batch_size, dtype=torch.bool, device=input_ids.device
            )
            self.eos_index = torch.zeros(
                batch_size, dtype=torch.int, device=input_ids.device
            )

        # Get last token ids in batch
        last_ids = input_ids[:, -1]

        # Create mask of where the last token is EOS
        done_update = self.done | (last_ids == self.eos_token)

        # Store the indices where we stopped at for each sequence in the batch.
        # Where the 'done' state has changed, store the seq_len (last index), else 0
        eos_index_update = torch.where(
            done_update ^ self.done, torch.full_like(self.eos_index, seq_len), 0
        )

        # Add the update to the indices
        self.eos_index += eos_index_update

        # Update the done flags
        self.done = done_update

        # Return True, if all done.
        return self.done.all()


class TextgenCallback(TrainerCallback):
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

    def on_evaluate(self, args, state, control, /, model, processing_class, **kwargs):
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

    def generate(self, device, model, tokenizer):
        generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **self.gen_config_args,
        )

        tokenizer_outputs = tokenizer(
            self.prompts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )

        input_ids = tokenizer_outputs["input_ids"].to(device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
                stopping_criteria=[EosStoppingCriteria(tokenizer)],
                return_dict_in_generate=True,
                output_scores=False,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
            )

        output_text = tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )

        for prompt, y in zip(self.prompts, output_text):
            s = prompt + " [START] " + y[len(prompt) + 1 :]
            yield s
