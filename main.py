from setproctitle import setproctitle
import argparse
import torch

from utils import TrainArguments, ModelArguments, DataArguments

from transformers import BertForSequenceClassification, BertTokenizerFast, BertConfig, HfArgumentParser
from datasets import load_dataset


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


if "__main__" in __name__:
    parser = HfArgumentParser([TrainArguments, ModelArguments, DataArguments])
    main(parser)
