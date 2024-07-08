import tokenizers
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

def train_bpe_tokenizer(tokenizer_config, dataset):
    # We can add tokens with special meanings to the tokenizer
    special_tokens={
        "pad": "<|PAD|>",   # Used to pad unused positions in a sequence.
        "mask": "<|MASK|>", # Used with masked-language-modeling to mark a position as having been masked.
        "bos": "<|BOS|>",   # Beginning of Sequence
        "eos": "<|EOS|>",   # End of Sequence
        "unk": "<|UNK|>",   # Unknown
    }
    
    # Create a new BPE tokenizer.
    pretrained_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(
        cache_capacity=16,
        unk_token=special_tokens['unk'],
        byte_fallback=True,
    ))
    
    # The 'normalizer' can be used to transform the characters. For example, they can convert everything to
    # lowercase, remove accent marks, and translate Unicode. We will use the NFC Unicode normalizer, with the 
    # detaisl explained here: https://unicode.org/reports/tr15/
    pretrained_tokenizer.normalizer = tokenizers.normalizers.NFC()
    
    # The decoder is applied when coverting tokens back into text and the ByteLevel decoder
    # is responsible for replacing 'Ġ' character with spaces. 
    pretrained_tokenizer.decoder = tokenizers.decoders.ByteLevel()
    
    # Automatically add Begin Of Sequence (BOS) token to output when 'add_special_tokens' is True
    # This has relevance to causal models, which predict the next token in a sequence. As the first real token lacks
    # a preceeding token, this allows the model to identify where the sequence actually begins.
    #
    # Note: A causal model can still function without a BOS token and the need to include it is debatable.
    pretrained_tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A",
        special_tokens=[
            ("<BOS>", 2),
        ],
    )

    # Attach the pre-tokenizer
    pretrained_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()

    # Create a BPE trainer, which is used to build an optimal set of tokens from
    # a a given dataset.
    tok_trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=tokenizer_config.vocab_size,
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(special_tokens.values()),
        show_progress=True,
    )
    
    # This abstraction is needed for the trainer to iterate over our dataset
    def batch_iterator(dataset, batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]['text']
    
    # Train the tokenizer of the dataset
    # Be patient! This will take a bit of time to complete...
    pretrained_tokenizer.train_from_iterator(batch_iterator(dataset), trainer=tok_trainer, length=len(dataset))
    print("Completed training")

    # This wraps the tokenizer in a Huggingface transformer tokenizer, which
    # is a higher level abstraction
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=pretrained_tokenizer,
        # This should match the model's input length limit, which depends upon the archetecture.
        # If not limit is specified, the default will be a VERY LARGE value.
        model_max_length=tokenizer_config.max_sequence_len,
        pad_token=special_tokens['eos'],
        mask_token=special_tokens['mask'],
        bos_token=special_tokens['bos'],
        eos_token=special_tokens['eos'],
        unk_token=special_tokens['unk'],
        return_special_tokens_mask=False,
    )
    return tokenizer