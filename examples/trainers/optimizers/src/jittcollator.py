
class JITCausalCollator:
    def __init__(self, tokenizer, max_length, tokenizer_args=None, preprocess_f=lambda batch: batch):
        self.tokenizer = tokenizer
        self.args = tokenizer_args
        self.preprocess_f = preprocess_f

    def __call__(self, batch):
        batch = self.preprocess_f(batch)
        
        args = dict(
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
            verbose=True,
        )

        if tokenizer_args is not None:
            args |= tokenizer_args
            
        outputs = self.tokenizer(batch, **args)
        input_ids = outputs["input_ids"]
        labels = torch.where(input_ids == self.tokenizer.pad_token_id, -100, input_ids)
        outputs["labels"] = labels
        return outputs
        
        