import argparse

import torch
import wandb
from datasets import load_dataset
from setproctitle import setproctitle
from torch import optim
from torch.utils.data import DataLoader
from torchaudio.functional import rnnt_loss
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, HfArgumentParser, trainer

from data import TorchCollator, TorchSampler
from utils import DataArguments, ModelArguments, TrainArguments


def main(parser: HfArgumentParser) -> None:
    train_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle("[JP]torch.ver bert")

    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base", cache_dir=train_args.cache)
    config = BertConfig.from_pretrained(
        "klue/bert-base",
        cache_dir=train_args.cache,
        vocab_size=tokenizer.vocab_size,
        num_labels=model_args.num_labels,
    )
    model = BertForSequenceClassification.from_pretrained("klue/bert-base", cache_dir=train_args.cache, config=config)

    loaded_data = load_dataset("nsmc", cache_dir=train_args.cache)
    train_data = loaded_data["train"]
    valid_data = loaded_data["test"]

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=train_args.train_batch_size,
        shuffle=False,
        batch_sampler=None,
        collate_fn=None,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=train_args.valid_batch_size,
        shuffle=False,
        batch_sampler=None,
        collate_fn=None,
        pin_memory=True,
    )

    optimizer = optim.Adam(model.parameters(), train_args.learning_rate)

    for epoch in range(train_args.train_epochs):
        for step, train_data in enumerate(train_dataloader):

            outputs = model()
            loss = rnnt_loss()
            optimizer.backword(loss)
            with torch.no_grad():
                for step, valid_data in enumerate(range(valid_dataloader)):
                    pass


if "__main__" in __name__:
    parser = HfArgumentParser([TrainArguments, ModelArguments, DataArguments])
    main(parser)
