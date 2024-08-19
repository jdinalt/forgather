import torch

torch.autograd.set_detect_anomaly(True)


# Verify model does not fall over when given input.
def kick_test(
    model,
    batch_size=2,
    seq_len=8,
    pad_probability=0.9,
    ignore_label=-100,
    device="cpu",
    dtype=None,
):
    if dtype is not None:
        model = model.to(dtype)
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters())

    opt.zero_grad()
    # input_ids = torch.arange(1, batch_size * seq_len + 1, dtype=torch.long).view(batch_size, seq_len)
    input_ids = torch.randint(
        1,
        model.config.vocab_size,
        (batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    # Generate fake padding mask
    pad_mask = (
        torch.full(input_ids.shape, pad_probability, dtype=torch.float, device=device)
        .bernoulli()
        .to(dtype=torch.long, device=device)
    )
    pad_mask = pad_mask.sort(-1, descending=True)[0]
    print("mask\n", pad_mask)

    # Replace pad values with pad_id and ignore_label
    labels = input_ids.masked_fill(~pad_mask.to(dtype=torch.bool), ignore_label)
    input_ids = input_ids.masked_fill(
        ~pad_mask.to(dtype=torch.bool), model.config.pad_token_id
    )
    print("input_ids\n", input_ids)
    print("labels\n", labels)

    input_ids = input_ids
    pad_mask = pad_mask
    labels = labels
    outputs = model(
        input_ids=input_ids, attention_mask=pad_mask, labels=labels, return_dict=True
    )
    loss = outputs["loss"]
    logits = outputs["logits"]
    print("logits.shape:", logits.shape)
    print("loss:", loss)

    # Make sure backward pass works.
    print("Computing loss.backward()...")
    loss.backward()

    print("Unused Parameters:")
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)

    print("Performing optimizertor step...")
    opt.step()
    print("Done! Congratulations, your model passed the kick-test!")
