import argparse

import torch
import wandb
from datasets import load_dataset
from setproctitle import setproctitle
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    HfArgumentParser,
    Wav2Vec2CTCTokenizer,
)
import datasets
from data import TorchCollator
from utils import DataArguments, ModelArguments, TrainArguments, get_concat_dataset
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model import LabelEncoder, TransformerTransducerConfig


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


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

    data_dir = r"/data/jsb193/audio/logmelspect"
    loaded_data = get_concat_dataset([data_dir], "train")

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("test42/wav2vec2-base-4data", use_auth_token=True)
    config = TransformerTransducerConfig(vocab_size=tokenizer.vocab_size)
    model = LabelEncoder(config)

    # tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base", cache_dir=train_args.cache)
    # config = BertConfig.from_pretrained(
    #     "klue/bert-base",
    #     cache_dir=train_args.cache,
    #     vocab_size=tokenizer.vocab_size,
    #     num_labels=model_args.num_labels,
    #     pad_token_id=tokenizer.pad_token_id,
    #     # initializer_range=0.2,
    # )
    # model = BertForSequenceClassification.from_pretrained("klue/bert-base", cache_dir=train_args.cache, config=config)

    if train_args.local_rank != -1:
        torch.cuda.set_device(train_args.local_rank)

        device = torch.device(train_args.local_rank)
        device_count = torch.cuda.device_count()
        torch.cuda.set_device(device.index)
        model = model.to(device)

        model = DistributedDataParallel(
            model,
            device_ids=[train_args.local_rank] if device_count != 0 else None,
            output_device=train_args.local_rank if device_count != 0 else None,
        )

    # loaded_data = load_dataset("nsmc", cache_dir=train_args.cache)
    # train_data = loaded_data["train"]
    # valid_data = loaded_data["test"]

    # train_data = train_data.map(preprocess, num_proc=5)
    # valid_data = valid_data.map(preprocess, num_proc=5)

    # train_data = train_data.remove_columns(["id", "document"])
    # valid_data = valid_data.remove_columns(["id", "document"])

    if train_args.local_rank != -1:
        sampler = DistributedSampler(
            dataset=train_data,
            num_replicas=2,
            rank=train_args.local_rank,
            shuffle=False,
            seed=42,
            drop_last=False,
        )
    generator = torch.Generator()
    generator.manual_seed(42)

    collator = TorchCollator(pad=0, device=train_args.local_rank)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=train_args.train_batch_size,
        shuffle=False,
        # sampler=sampler,
        batch_sampler=None,
        collate_fn=collator,
        pin_memory=True,
        generator=generator,
    )
    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=train_args.valid_batch_size,
        shuffle=False,
        batch_sampler=None,
        collate_fn=None,
        pin_memory=True,
    )

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), train_args.learning_rate)
    model.zero_grad()

    for epoch in range(1, (train_args.train_epochs + 1)):
        # set_epoch는 sampler의 shuffle이 True일 때를 위한 함수다.
        # 만약 set_epoch이 설정되지 않았다면 매 epoch마다 동일한 데이터가 들어갈 수 있다.
        # random_seed를 설정할 때 seed + epoch으로 seed를 설정하기 때문에 shuffle시 중요한 요소
        # train_dataloader.sampler.set_epoch(epoch)
        model.train()
        for step, train_data in enumerate(train_dataloader, 1):
            if train_args.local_rank != -1:
                train_data = {key: data.to(train_args.local_rank) for key, data in train_data.items()}
                with model.no_sync():
                    outputs = model(**train_data)
            else:
                train_data = {key: data for key, data in train_data.items()}
                input_data = train_data["input_ids"]
                attention_mask = train_data["attention_mask"]
                token_type_ids = train_data["token_type_ids"]
                logits = model(input_data, attention_mask, token_type_ids)
            # logits = outputs.logits
            loss = loss_func(logits, train_data["labels"])
            # loss = rnnt_loss()
            loss = loss / train_args.train_accumulate
            loss.backward()
            # scheduler.step()
            if (step % train_args.train_accumulate) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.zero_grad()

                print(f"step: {step*epoch}: {loss}")
                print

            if (step % train_args.valid_step) == 0:

                with torch.no_grad():
                    for step, valid_data in enumerate(range(valid_dataloader)):
                        pass


if "__main__" in __name__:
    parser = HfArgumentParser([TrainArguments, ModelArguments, DataArguments])
    main(parser)
