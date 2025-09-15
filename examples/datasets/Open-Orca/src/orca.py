import jinja2
import logging
import os
from forgather.ml.distributed import main_process_first
from forgather.ml.datasets import to_iterable_dataset_with_length

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

chatml_template = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"""


def to_conversations(batch):
    for system, question, response in zip(
        batch["system_prompt"], batch["question"], batch["response"]
    ):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
        yield messages


def map_function(input_batch, tokenizer, chat_template, template_args, tokenizer_args):
    conversations = [
        chat_template.render(
            messages=messages,
            **template_args,
        )
        for messages in to_conversations(input_batch)
    ]
    if not tokenizer:
        return {"text": conversations}

    outputs = tokenizer(
        conversations,
        **tokenizer_args,
    )
    return {"input_ids": outputs["input_ids"]}


@main_process_first()
def preprocess_orca(
    dataset,
    chat_template=None,
    tokenizer=None,
    tokenizer_args=None,
    map_args=None,
    template_args=None,
    desc="Tokenizing Dataset",
    num_shards=256,
    to_iterable=False,
):
    if tokenizer_args is None:
        tokenizer_args = dict()
    if map_args is None:
        map_args = dict()
    if template_args is None:
        template_args = dict()
    if not chat_template or len(chat_template) == 0:
        if tokenizer and tokenizer.chat_template:
            chat_template = tokenizer.chat_template
            logger.info("Using chat template from tokenizer")
        else:
            chat_template = chatml_template
            logger.warning("Using default chat template (ChatML)")
    else:
        with open(chat_template, "r") as f:
            chat_template = f.read()

    environment = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
    )
    chat_template = environment.from_string(chat_template)

    if to_iterable:
        dataset = to_iterable_dataset_with_length(dataset, num_shards=num_shards)
    else:
        map_args["desc"] = desc

    output_dataset = dataset.map(
        map_function,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            chat_template=chat_template,
            template_args=template_args,
            tokenizer_args=tokenizer_args,
        ),
        **map_args,
    )
    return output_dataset
