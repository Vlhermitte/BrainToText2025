from tqdm import tqdm
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import random

from evaluation import predict_sentence
from data import ascii_ids_to_text

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss does not improve.
    """
    def __init__(self, patience=5, min_delta=1e-3, path="best_model.pt"):
        """
        patience: epochs to wait after last improvement
        min_delta: minimum change to count as an improvement
        path: where to save the best model state_dict
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value, model):
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            if not os.path.exists(os.path.dirname(self.path)):
                os.makedirs(os.path.dirname(self.path))
            # save best weights
            torch.save(model.state_dict(), self.path)
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True


# python
class Trainer:
    """
    Trainer class to handle training and validation of a model.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
        epochs: int = 100,
        blank_id: int = 128,
        early_stop: EarlyStopping = None,
        sample_interval: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.blank_id = blank_id
        self.sample_interval = sample_interval

        self.early_stop = early_stop or EarlyStopping(patience=5, min_delta=1e-3, path="./model/best_model.pt")
        self.best_val = float("inf")

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            x, x_len, y, y_len = batch
            x, x_len, y, y_len = x.to(self.device), x_len.to(self.device), y.to(self.device), y_len.to(self.device)

            logits, in_len_after = self.model(x, x_len)        # (T, B, C)
            log_probs = logits.log_softmax(dim=-1)             # CTC expects log-probs

            loss = self.loss_fn(log_probs, y, in_len_after, y_len)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    @torch.no_grad()
    def predict_sample(self, idx: int = None):
        """
        Predict and print a sample from the validation set.
        If idx is None, a random sample is chosen.
        :param idx:
        :return:
        """
        # TODO: look at this function (especially predict_sentence() function)
        if idx is None:
            idx = random.randrange(len(self.val_loader.dataset))
            print(f"Using random sample index: {idx}")
        self.model.eval()
        with torch.no_grad():
            sample_x, sample_y = self.val_loader.dataset[idx]
            sample_text = ascii_ids_to_text(sample_y.tolist())
            pred_text, pred_ids = predict_sentence(model=self.model, x=sample_x)
            print(f"Selected index: {idx}")
            print(f"Target: {sample_text}")
            #print(f"Predicted IDs: {pred_ids}")
            print(f"Predicted text: {pred_text}")

    @torch.no_grad()
    def run_validation(self) -> float:
        self.model.eval()
        total_loss, total_items = 0.0, 0
        for x, x_len, y, y_len in self.val_loader:
            x, x_len, y, y_len = x.to(self.device), x_len.to(self.device), y.to(self.device), y_len.to(self.device)
            logits, in_len_after = self.model(x, x_len)                 # (T, B, C)
            log_probs = logits.log_softmax(dim=-1)   # (T, B, C)
            loss = self.loss_fn(log_probs, y, in_len_after, y_len)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_items += bs
        return total_loss / max(1, total_items)

    def run(self) -> None:
        """
        Train the model with early stopping based on validation loss.
        """
        for epoch in range(self.epochs):
            self._train_one_epoch(epoch)

            # Validation & early stopping
            val_loss = self.run_validation()
            improved = val_loss < self.best_val - 1e-9
            if improved:
                self.best_val = val_loss
            self.early_stop.step(val_loss, self.model)

            print(f"[Epoch {epoch+1}] val_loss={val_loss:.4f} best_val={self.best_val:.4f} "
                  f"patience_used={self.early_stop.bad_epochs}/{self.early_stop.patience}")

            if self.early_stop.should_stop:
                print("Early stopping triggered. Reloading best weights and exiting training.")
                self.model.load_state_dict(torch.load(self.early_stop.path, map_location=self.device))
                break