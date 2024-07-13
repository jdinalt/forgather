import os


def make_project_tokenizer(force: bool = False):
    """
    This ensures that the tutorial tokenizer has been built

    It tries to avoid loading anything that is not absolutely required.
    """
    if not force and os.path.exists(
        os.path.join("..", "tokenizers", "tiny_stories_2k")
    ):
        return

    from forgather.config import load_config, materialize_config
    from aiws.tokenizer_trainer import TokenizerTrainer
    from aiws.dotdict import DotDict

    metacfg = DotDict(load_config("forgather_config.yaml").config)
    config = DotDict(
        materialize_config(
            metacfg.tokenizer_def,
            whitelist=metacfg.tokenizers_whitelist,
            search_path=metacfg.search_paths,
        ).config
    )

    trainer = TokenizerTrainer(**config.trainer_args)
    trainer.train()
    tokenizer = trainer.as_pretrained_tokenizer_fast(
        **config.pretrained_tokenizer_fast_args
    )
    tokenizer.save_pretrained(metacfg.tokenizer_path)
