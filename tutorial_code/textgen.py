import torch

class TextGenerator:
    def __init__(self, model, tokenizer, device, temperature=1.0, do_sample=False, seed=None):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.device = device
        self.do_sample = do_sample
        self.rand_generator = torch.Generator(device=device)
        self.set_seed(seed)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20):
        for _ in range(max_new_tokens):
            logits = self.model(input_ids=input_ids)
            logits = logits[:, -1, :] / self.temperature
            probabilities = torch.softmax(logits, dim=-1)
            if self.do_sample:
                next_token_id = torch.multinomial(
                    probabilities, num_samples=1, generator=self.rand_generator)
            else:
                _, next_token_id = torch.topk(probabilities, k=1, dim=-1)
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
        return input_ids

    def set_seed(self, seed):
        if seed is None:
            self.rand_generator.seed()
        else:
            self.rand_generator.manual_seed(seed)
    
    # Lazy generation pipeline for simple inference.
    def prompt(self, input_text, max_new_tokens=20):
        input_ids = self.tokenizer(input_text, return_tensors='pt')['input_ids']
        model_output = self.generate(
            input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.decode(model_output.to('cpu')[0])