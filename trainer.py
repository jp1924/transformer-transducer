from argparse import Namespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel


class TorchTraner:
    def __init__(self, model, optimizer, tokenizer, train_data, valid_data, collator, args: Namespace) -> None:
        self.train_data = train_data
        self.valid_data = valid_data
        self.collator = collator
        self.args = args
        self.optimizer = optimizer

        if args.local_rank != -1:
            local_rank = self.args.local_rank
            dist.init_process_group(backend="nccl", rank=local_rank)
            self.model = DistributedDataParallel(model, device_ids=[args.local_rank])
        else:
            self.model = model
            torch.cuda.set_device(1)
        return

    def train(self) -> None:
        self.gradient_accumulate: int = lambda step: (step % self.args.train_gradient_accumulation) == 0
        return

    def train_step(self, step, train_data) -> None:

        outputs = self.model(**train_data)
        loss = outputs.loss

        if self.gradient_accumulate(step):
            loss.backward()
            self.optimizer.step()

        return

    def valid(self) -> None:
        return

    def valid_step(self) -> None:
        return

    def predict(self) -> None:
        return

    def predict_step(self) -> None:
        return

    def set_train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            dataset=self.train_data,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            batch_sampler=None,
            collate_fn=self.collator,
            pin_memory=True,
        )

        return

    def set_valid_datalaoder(self) -> DataLoader:
        valid_dataloader = DataLoader(
            dataset=self.valid_data,
            batch_size=self.args.valid_batch_size,
            shuffle=False,
            batch_sampler=None,
            collate_fn=self.collator,
            pin_memory=True,
        )

        return
