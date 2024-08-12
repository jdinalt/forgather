import torch
import datetime
from tqdm.auto import tqdm

class Trainer:
    def __init__(
        self,
        train_dataloader,
        test_dataloader,
        model,
        loss_fn,
        optimizer,
        epochs,
        batch_size,
        logging_steps,
    ):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.logging_steps = logging_steps

        self.total_steps = len(self.train_dataloader) * self.epochs

    def __repr__(self):
        return (
            f"{type(self).__name__}(train_dataloader={repr(self.train_dataloader)}, "
            f"test_dataloader={repr(self.test_dataloader)}, "
            f"model={repr(self.model)}, "
            f"loss_fn={repr(self.loss_fn)}, "
            f"optimizer={repr(self.optimizer)}, "
            f"epochs={repr(self.epochs)}, "
            f"batch_size={repr(self.batch_size)}, "
            f"logging_steps={repr(self.logging_steps)})"
        )

    def __call__(self):
        self.global_step = 0
        self.train_progress_bar = tqdm(total=self.total_steps, dynamic_ncols=True)
        try:
            for i in range(self.epochs):
                self.train_loop()
                self.eval_loop()

        finally:
            self.train_progress_bar.close()
            self.train_progress_bar = None

    def train_loop(self):
        self.model.train()

        # Train for a  full epoch.
        for inputs, targets in self.train_dataloader:
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            self.train_progress_bar.update()
            if self.global_step % self.logging_steps == 0:
                self.train_progress_bar.write(self._format_train(loss.item()))

    @torch.no_grad()
    def eval_loop(self):
        eval_progress_bar = tqdm(
            total=len(self.test_dataloader),
            leave=self.train_progress_bar is None,
            dynamic_ncols=True,
        )

        self.model.eval()
        correct = 0
        total_loss = 0

        try:
            for inputs, targets in self.test_dataloader:
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)
                total_loss += loss.item()
                correct += (logits.argmax(1) == targets).type(torch.float).sum().item()
                eval_progress_bar.update()
                
        finally:
            mean_loss = total_loss / len(self.test_dataloader)
            accuracy = correct / len(self.test_dataloader.dataset)
            eval_progress_bar.write(self._format_eval(mean_loss, accuracy))
            eval_progress_bar.close()
    
    def _record_header(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch = self.global_step / len(self.train_dataloader)
        s = f"{timestamp:<22}{self.global_step:>10,d}  {round(epoch, 2):<5.3}"
        return s
    
    def _format_train(self, loss):
        header = self._record_header()
        return f"{header} train-loss: {round(loss, 5):<10}"

    def _format_eval(self, eval_loss, accuracy):
        header = self._record_header()
        return f"{header} eval-loss:  {round(eval_loss, 5):<10}accuracy: {(accuracy * 100):>0.1f}"
