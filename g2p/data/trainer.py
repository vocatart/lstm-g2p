import math
import random
import torch
from torch import nn
from torch.utils.data import Dataset
from torchaudio.transforms import RNNTLoss
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import tqdm
import os
import datetime

from utils import BOS_IDX, UNK_IDX, PAD_IDX
from dataset import TSVDataset
from model import LstmG2P
from tabulate import tabulate

class Trainer():
    def __init__(self,
                 run_name: str,
                 device: torch.device,
                 loss_device: torch.device,
                 model: LstmG2P,
                 dataset: TSVDataset,
                 batch_size: int = 256,
                 num_epochs: int = 30,
                 learning_rate: float = 0.005,
                 minimum_learning_rate: float = 5e-5,
                 gamma: float = 0.8,
                 grad_clip: float = 2.0,
                 validation_divide_by: int = 10,
                 dl_workers: int = 0,
                 seed=None) -> None:
        
        self.run_name = run_name
        self.device = device
        self.loss_device = loss_device
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.minimum_learning_rate = minimum_learning_rate
        self.gamma = gamma
        self.validation_divide_by = validation_divide_by
        self.dl_workers = dl_workers
        self.seed = seed
        self.grad_clip = grad_clip

        self.run_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # create artifacts dir
        self.artifacts_dir = f"artifacts/{run_name}"
        os.makedirs(self.artifacts_dir, exist_ok=True)
        print(f"artifacts will be saved to: {self.artifacts_dir}")

        # send model to device
        self.model.to(device)

        # initialize loss, optimizer, and scheduler
        self.loss_function = RNNTLoss(blank=BOS_IDX, clamp=self.grad_clip)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        # calculate split lengths
        valid_set_size = len(dataset) // validation_divide_by
        train_set_size = len(dataset) - valid_set_size

        # create random generator for reproducibility
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        
        # split dataset into training and validation sets
        valid_set, train_set = random_split(dataset, [valid_set_size, train_set_size], generator=generator)

        def collate_fn(batch):
            grapheme_batch, phoneme_batch = [], []

            for word_tensor, phoneme_tensor in batch:
                grapheme_batch.append(word_tensor)
                phoneme_batch.append(phoneme_tensor)

            grapheme_lengths = torch.tensor([len(tensor) for tensor in grapheme_batch], dtype=torch.int32)
            phoneme_lengths = torch.tensor([len(tensor) for tensor in phoneme_batch], dtype=torch.int32)

            grapheme_batch = pad_sequence(grapheme_batch, padding_value=PAD_IDX)
            phoneme_batch = pad_sequence(phoneme_batch, padding_value=PAD_IDX)

            return grapheme_batch.transpose(0, 1).contiguous(), phoneme_batch.transpose(0, 1).contiguous(), grapheme_lengths, phoneme_lengths
        
        # create data loaders for training and validation sets
        self.training_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=dl_workers
        )
        self.validation_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=dl_workers
        )

    def _accuracy(self, logits: Tensor, phoneme_truths: Tensor):
        # logits: (batch_size, seq_len, num_phonemes) or (batch_size, seq_len, ...)
        # phoneme_truths: (batch_size, seq_len)

        # if logits has more than 3 dims, take the diagonal along encoder/decoder axes
        if logits.dim() == 4:
            # (batch, enc_len, dec_len, num_phonemes) -> (batch, seq_len, num_phonemes) by diagonal
            seq_len = min(logits.size(1), logits.size(2))

            # logits: (batch, seq_len, num_phonemes)
            logits = logits[:, torch.arange(seq_len), torch.arange(seq_len), :]

        phoneme_predictions = torch.argmax(logits, dim=-1)

        # align the prediction length with the truth length by truncating the longer tensor
        pred_len = phoneme_predictions.shape[1]
        truth_len = phoneme_truths.shape[1]

        if pred_len < truth_len:
            phoneme_truths = phoneme_truths[:, :pred_len]
        elif truth_len < pred_len:
            phoneme_predictions = phoneme_predictions[:, :truth_len]

        mask = (phoneme_truths != UNK_IDX)
        correct_predictions = (phoneme_predictions[mask] == phoneme_truths[mask]).sum().item()
        total = mask.sum().item()
        return correct_predictions, total
    
    def _train_epoch(self, step: int, total_loss: float, epoch: int, writer: SummaryWriter):
        # set model mode
        self.model.train()
        
        # initialize loss and step counter
        running_loss = 0.0
        batch_count = 0
        pbar = tqdm.tqdm(self.training_loader, desc=f"epoch {epoch}", unit="step")

        for graphemes, phonemes, grapheme_lengths, phoneme_lengths in pbar:
            # teacher forcing: prepend BOS_IDX to phonemes
            phonemes_in = torch.concat([torch.full([phonemes.shape[0], 1], BOS_IDX), phonemes], dim=1)

            # forward pass
            logits = self.model(graphemes.to(self.device), phonemes_in.to(self.device))

            # gradient calculation
            self.optimizer.zero_grad()
            loss = self.loss_function(
                logits.to(self.loss_device),
                phonemes.to(self.loss_device),
                grapheme_lengths.to(self.loss_device),
                phoneme_lengths.to(self.loss_device)
            )
            loss.backward()

            # gradient clipping and optimization step
            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # accumulate loss
            total_loss += loss.item()
            running_loss += loss.item()
            step += 1
            batch_count += 1

            # update progress bar with average loss for this epoch
            avg_loss = running_loss / batch_count
            pbar.set_postfix({'loss': avg_loss})

            # log metrics to tensorboard
            writer.add_scalar('training/loss', loss.item(), step)
            writer.add_scalar('training/lr', self.scheduler.get_last_lr()[0], step)
            writer.flush()

        # return epoch loss, total loss, and steps
        return (total_loss / step), total_loss, step
        
    def _eval(self, step: int, writer: SummaryWriter):
        # set model mode
        self.model.eval()

        total_loss = 0
        num_validation_batches = 0
        total_correct = 0
        total_tokens = 0

        for i, (graphemes, phonemes, grapheme_lengths, phoneme_lengths) in enumerate(self.validation_loader):
            # teacher forcing: prepend BOS_IDX to phonemes
            phonemes_in = torch.concat([torch.full([phonemes.shape[0], 1], BOS_IDX), phonemes], dim=1)

            # forward pass
            logits = self.model(graphemes.to(self.device), phonemes_in.to(self.device))

            # calculate loss
            loss = self.loss_function(
                logits.to(self.loss_device),
                phonemes.to(self.loss_device),
                grapheme_lengths.to(self.loss_device),
                phoneme_lengths.to(self.loss_device)
            )

            # accuracy calculation
            correct, total = self._accuracy(logits, phonemes.to(self.device))
            total_correct += correct
            total_tokens += total

            # metric accumulation
            total_loss += loss.item()
            num_validation_batches += 1

        # log metrics to tensorboard
        writer.add_scalar('validation/loss', total_loss / num_validation_batches, global_step=step)
        if total_tokens > 0:
            ph_accuracy = total_correct / total_tokens
            writer.add_scalar('validation/ph_accuracy', ph_accuracy, global_step=step)
        
        writer.flush()

        return total_loss / num_validation_batches
    
    def _save_state_dict(self, name):
        statedict_path = os.path.join(self.artifacts_dir, f'statedict-{name}.pt')
        model_path = os.path.join(self.artifacts_dir, f'model-{name}.pt')
        torch.save(self.model.state_dict(), statedict_path)
        torch.save(self.model, model_path)

    def _load_state_dict(self, name):
        statedict_path = os.path.join(self.artifacts_dir, f'statedict-{name}.pt')
        self.model.load_state_dict(torch.load(statedict_path))

    def _preview(self, entries) -> list[tuple[str, list[str], list[str]]]:
        validated_entries = []
        for word, phonemes in entries:
            predicted_phonemes = self.model.predict_str(word)
            validated_entries.append((word, predicted_phonemes, phonemes))

        return validated_entries
    
    def train(self):
        preview_entries = []
        writer = SummaryWriter(log_dir='runs/{}'.format(self.run_name))

        for i in range(5):
            idx = random.randrange(len(self.dataset.entries))
            preview_entries.append(self.dataset.entries[idx])

        # initialize steps and loss values
        best_eval_loss = 10000
        global_loss = 0
        steps = 0
        for i in range(self.num_epochs):
            loss, global_loss, steps = self._train_epoch(steps, global_loss, i, writer)
            eval_loss = self._eval(steps, writer)
            learning_rate = self.scheduler.get_last_lr()[0]

            #print('epoch: {} - lr: {:.2e} - loss: {:.4f} - eval_loss: {:.4f}'
            #      .format(i, learning_rate, loss, eval_loss))
            
            if loss is None or eval_loss is None or math.isnan(loss) or math.isnan(eval_loss):
                print('Loss is NaN or None, stopping training.')
                break

            if learning_rate > self.minimum_learning_rate:
                self.scheduler.step()

            if best_eval_loss > eval_loss:
                best_eval_loss = eval_loss
                self._save_state_dict('best')

            if (i + 1) % 20 == 0:
                self._save_state_dict('epoch-{}'.format(i + 1))

            # generate validation previews and push to tensorboard
            test_previews = self._preview(preview_entries)

            table_data = [(p[0], " ".join(p[1]), " ".join(p[2])) for p in test_previews]
            table_str = tabulate(table_data, headers=["Word", "Prediction", "Expected"], tablefmt="pipe")

            writer.add_text('validation/previews', table_str, steps)
            writer.flush()
        
        writer.close()