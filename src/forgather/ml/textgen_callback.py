from typing import List
import os

from transformers import GenerationConfig, StoppingCriteria
from torch.utils.tensorboard import SummaryWriter
import torch
from .utils import format_train_info
from .trainer_types import TrainerCallback


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
        prompts: List[str],
        generation_config: GenerationConfig = None,
        generation_steps: int = None,
        max_new_tokens: int = 200,
    ):
        super().__init__()
        self.summary_writer = summary_writer
        self.prompts = prompts

        # To construct GenerationConfig, we need token ids from the model or tokenizer
        # We don't have these here, so defer construction until callback.
        self.generation_config = None
        if generation_config is not None and isinstance(
            self.generation_config, GenerationConfig
        ):
            self.generation_config = generation_config
        elif generation_config is None:
            self.gen_config_args = dict(
                do_sample=True,
                top_k=20,
                top_p=0.9,
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
        for output in self.generate(args, model, processing_class):
            text += output + "\n\n---\n\n"
        self.summary_writer.add_text("eval-text", text, global_step=state.global_step)
        self.summary_writer.flush()

    def init_gen_config(self, model):
        self.generation_config = GenerationConfig(
            pad_token_id=model.config.pad_token_id,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            **self.gen_config_args,
        )

    def generate(self, args, model, processing_class):
        if self.generation_config is None:
            self.init_gen_config(model)
        for prompt in self.prompts:
            tokenizer_outputs = processing_class(
                [prompt],
                truncation=False,
                return_length=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            lengths = tokenizer_outputs["length"]
            input_ids = tokenizer_outputs["input_ids"].to(args.device)
            attention_mask = tokenizer_outputs["attention_mask"].to(args.device)

            use_cache = False  # getattr(model, "_supports_cache_class", False)
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                stopping_criteria=[EosStoppingCriteria(processing_class)],
                return_dict_in_generate=True,
                use_cache=use_cache,
                past_key_values=None,
                max_new_tokens=self.max_new_tokens,
            )

            output_text = processing_class.decode(
                outputs.sequences[0],
                skip_special_tokens=True,
            )
            s = prompt + " [START] " + output_text[len(prompt) + 1 :]
            yield s
