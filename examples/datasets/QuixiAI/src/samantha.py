import jinja2
import logging
import os
from forgather.ml.distributed import main_process_first

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

chatml_template = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"""


def samantha_conversation(conversation):
    messages = []
    for human, gpt in zip(conversation["human"], conversation["gpt"]):
        messages.extend(
            [
                {"role": "user", "content": human},
                {"role": "assistant", "content": gpt},
            ]
        )
    return messages


def samantha_map_function(
    input_batch, tokenizer, chat_template, template_args, tokenizer_args
):
    conversations = [
        chat_template.render(
            messages=samantha_conversation(batch),
            **template_args,
        )
        for batch in input_batch["conversations"]
    ]
    if not tokenizer:
        return {"text": conversations}

    outputs = tokenizer(
        conversations,
        **tokenizer_args,
    )
    return {"input_ids": outputs["input_ids"]}

@main_process_first()
def preprocess_samantha(
    dataset,
    chat_template=None,
    tokenizer=None,
    tokenizer_args=None,
    map_args=None,
    template_args=None,
    desc="Tokenizing Dataset",
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
    output_dataset = dataset.map(
        samantha_map_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=desc,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            chat_template=chat_template,
            template_args=template_args,
            tokenizer_args=tokenizer_args,
        ),
        **map_args,
    )
    return output_dataset
