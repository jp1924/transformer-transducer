import datasets
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from data import TransducerCollator, get_concat_dataset
from datasets import load_dataset
from model import TransformerTranducer, TransformerTransducerConfig
from setproctitle import setproctitle
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import HfArgumentParser, Wav2Vec2CTCTokenizer
from utils import DataArguments, ModelArguments, TrainArguments


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
    setproctitle("[JP]Transformer-Transducer")
    train_data = get_concat_dataset([data_args.data_name], "train")

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("test42/wav2vec2-base-4data", use_auth_token=True)
    config = TransformerTransducerConfig(vocab_size=tokenizer.vocab_size)
    model = TransformerTranducer(config).to("cuda:0")

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

    collator = TransducerCollator(tokenizer)

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
    # valid_dataloader = DataLoader(
    #     dataset=valid_data,
    #     batch_size=train_args.valid_batch_size,
    #     shuffle=False,
    #     batch_sampler=None,
    #     collate_fn=None,
    #     pin_memory=True,
    # )

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
                audio_values = train_data["audio_values"].to("cuda:0")
                label_values = train_data["label_values"].to("cuda:0")
                audio_attention_mask = train_data["audio_attention_mask"].to("cuda:0")
                label_attention_mask = train_data["label_attention_mask"].to("cuda:0")

                loss = model(audio_values, label_values, audio_attention_mask, label_attention_mask)

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
