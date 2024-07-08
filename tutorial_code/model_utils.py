def print_model_size(model):
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

def test_model_forward(model, tokenizer, text, device='cpu'):
    model.train()
    model = model.to(device=device)
    
    input_ids = tokenizer(
        text,
        truncation=True,
        return_tensors='pt',
    )['input_ids'].to(device=device)

    print("input_ids:\n", input_ids)
    labels = input_ids

    loss, logits = model(input_ids=input_ids, labels=labels)
    print(loss)
    
    # Compute gradient
    loss.backward()

    # Reset model gradients
    model.zero_grad()