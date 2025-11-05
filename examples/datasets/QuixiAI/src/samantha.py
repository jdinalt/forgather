from typing import Any, Optional, Callable

import jinja2
import logging
import os
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from forgather.ml.distributed import main_process_first

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        return { "text": conversations }

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
    map_kwargs=None,
    template_args=None,
    desc="Tokenizing Dataset",
    map_fn: Callable=None,
    fn_kwargs: Optional[dict[str, Any]]=None,
):
    if tokenizer_args is None:
        tokenizer_args = dict()
    if map_kwargs is None:
        map_kwargs = dict()
    if template_args is None:
        template_args = dict()

    template_args['bos_token'] = tokenizer.bos_token
    template_args['eos_token'] = tokenizer.eos_token
    if not chat_template or len(chat_template) == 0:
        if tokenizer and tokenizer.chat_template:
            chat_template = tokenizer.chat_template
            logger.info("Using chat template from tokenizer")
        else:
            chat_template = chatml_template
            logger.warning("Using default chat template (ChatML)")
    else:
        logger.info(f"Using chat template: {chat_template}")
        with open(chat_template, "r") as f:
            chat_template = f.read()

    environment = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
    )
    chat_template = environment.from_string(chat_template)
    dataset = dataset.map(
        samantha_map_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=desc,
        fn_kwargs=dict(
            tokenizer=tokenizer if not map_fn else None,
            chat_template=chat_template,
            template_args=template_args,
            tokenizer_args=tokenizer_args,
        ),
        **map_kwargs,
    )

    # If map_fn, pipeline with map_fn
    if map_fn is not None:
        if not fn_kwargs:
            fn_kwargs=dict()
        fn_kwargs=dict(
            tokenizer=tokenizer,
            feature="text",
        ) | fn_kwargs
        dataset = dataset.map(
            map_fn,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Applying map function",
            fn_kwargs=fn_kwargs,
        )
    return dataset


@main_process_first()
def load_samantha_dataset_manual(
    cache_dir=None, language="en", repo_id="QuixiAI/samantha-data"
):
    """
    Manually download and load the Samantha dataset from Huggingface Hub.
    This replicates the original dataset loading script that used percentage-based splits.

    Args:
        cache_dir: Directory to cache downloaded files
        language: Language code ('en', 'it', 'km', 'zh')
        repo_id: Huggingface repository ID

    Returns:
        DatasetDict with train/validation/test splits matching original script
    """
    # Determine filename based on language
    if language == "en":
        filename = "samantha-1.1.json"
    else:
        filename = f"samantha-1.1-{language}.json"

    logger.info(f"Downloading Samantha dataset file: {filename}")

    try:
        file_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=cache_dir, repo_type="dataset"
        )
        logger.info(f"Downloaded {filename} to {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        logger.info(f"Loaded {len(data_list)} conversations from {filename}")

        # Convert to the expected format with human/gpt sequences
        processed_data = []
        for data in data_list:
            conversations = data["conversations"]
            human = []
            gpt = []
            for conv_id, conversation in enumerate(conversations):
                value_str = conversation["value"].strip()
                if conv_id % 2 == 0:  # Even index = human
                    human.append(value_str)
                else:  # Odd index = gpt
                    gpt.append(value_str)

            processed_data.append(
                {"id": data["id"], "conversations": {"human": human, "gpt": gpt}}
            )

        # Create splits using exact percentages from original script
        # Train: 0-80%, Validation: 80-95%, Test: 95-100%
        total_len = len(processed_data)
        train_end = int(total_len * 0.80)
        val_end = int(total_len * 0.95)

        train_data = processed_data[0:train_end]
        val_data = processed_data[train_end:val_end]
        test_data = processed_data[val_end:]

        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data),
                "test": Dataset.from_list(test_data),
            }
        )

        logger.info(
            f"Created splits: train={len(train_data)}, validation={len(val_data)}, test={len(test_data)}"
        )

        return dataset_dict

    except Exception as e:
        logger.error(f"Failed to download or process {filename}: {e}")
        raise


def load_samantha_split(split="train", **kwargs):
    """Load a specific split of the Samantha dataset."""
    dataset_dict = load_samantha_dataset_manual(**kwargs)
    return dataset_dict[split]
