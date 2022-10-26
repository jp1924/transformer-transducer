import argparse

import torch
import wandb
from datasets import load_dataset
from setproctitle import setproctitle
from torch import optim
from torch.utils.data import DataLoader
from torchaudio.functional import rnnt_loss
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    HfArgumentParser,
    trainer,
    BertTokenizer,
)
import datasets
from data import TorchCollator, TorchSampler
from utils import DataArguments, ModelArguments, TrainArguments
import torch.nn as nn


def main(parser: HfArgumentParser) -> None:
    train_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle("[JP]torch.ver bert")

    def preprocess(input_data: datasets.Dataset) -> dict:
        """각 데이터를 토크나이징 및 별도의 전처리를 적용시키는 함수

        각 데이터의 전처리르 위한 함수 입니다. 이 함수는 datasets으로 부터
        dict형태의 각 데이터를 입력받아 tokenizer로 인코딩 후 dict형식으로 반환합니다.

        Args:
            input_data (datasets.Dataset): Datasets로 부터 건내받은 dict형식의 각 데이터를 전달 받습니다.

        Returns:
            BatchEncoding: Datasets의 각 열의 해당되는 columns를 가지고 전처리 된 데이터를 반환합니다.
        """
        input_text = input_data["document"]
        output_data = tokenizer(input_text, return_attention_mask=False)
        output_data["label"] = input_data["label"]
        return output_data

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

    train_data = train_data.map(preprocess, num_proc=5)
    valid_data = valid_data.map(preprocess, num_proc=5)

    train_data = train_data.remove_columns(["id", "document"])
    valid_data = valid_data.remove_columns(["id", "document"])

    collator = TorchCollator(pad=0)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=train_args.train_batch_size,
        shuffle=False,
        batch_sampler=None,
        collate_fn=collator,
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
    cross_entropy_loss = nn.CrossEntropyLoss()

    for epoch in range(train_args.train_epochs):
        for step, train_data in enumerate(train_dataloader, 1):

            outputs = model(**train_data)
            logits = outputs.logits
            loss = cross_entropy_loss(logits, train_data["labels"])
            # loss = rnnt_loss()
            if (step % train_args.train_gradient_accumulation) == 0:
                loss.backward()
                optimizer.step()

            if (step % train_args.valid_step) == 0:

                with torch.no_grad():
                    for step, valid_data in enumerate(range(valid_dataloader)):
                        pass


if "__main__" in __name__:
    parser = HfArgumentParser([TrainArguments, ModelArguments, DataArguments])
    main(parser)
