import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import torch
from data import DataCollatorRNNTWithPadding
from datasets import Dataset, concatenate_datasets, load_dataset
from models import TransformerTransducerForRNNT, TransformerTransducerProcessor
from setproctitle import setproctitle
from utils import TransformerTransducerArguments, default_sentence_norm

from transformers import HfArgumentParser, Trainer, is_torch_xla_available, is_wandb_available, set_seed
from transformers import logging as hf_logging
from transformers.trainer_utils import is_main_process


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")

global GLOBAL_LOGGER
GLOBAL_LOGGER = None

BLK_TOKEN = "<blank>"


def main(train_args: TransformerTransducerArguments) -> None:
    def preprocessor(example: Dict[str, Union[List[Any], List[List[Any]]]]) -> Dict[str, List[Any]]:
        sentence_ls = example[train_args.sentence_column_name]
        sentence_ls = sentence_ls if isinstance(sentence_ls, list) else [sentence_ls]

        audio_ls = example[train_args.audio_column_name]
        audio_ls = audio_ls if isinstance(audio_ls, list) else [audio_ls]
        audio_ls = [audio["array"] for audio in audio_ls]

        finish_data_ls = list()
        for sentence, audio in zip(sentence_ls, audio_ls):
            audio = np.array(audio)
            audio_length = audio.shape[0]
            if not audio.any():
                continue

            if not train_args.min_duration_in_seconds <= audio_length <= train_args.max_duration_in_seconds:
                continue

            sentence = default_sentence_norm(sentence)
            if not sentence:
                continue

            sentence = f"{BLK_TOKEN}{sentence}"
            input_ids = processor(text=sentence, return_attention_mask=False, return_tensors="np")["input_ids"]

            finish_data_ls.append(
                {
                    "input_features": audio,
                    "labels": input_ids[0],
                    train_args.length_column_name: audio_length,
                }
            )

        return pa.Table.from_pylist(finish_data_ls)

    def collect_dataset(prefix_ls: List[str]) -> Optional[Dataset]:
        if not prefix_ls:
            return None

        data_ls = list()
        for prefix in prefix_ls:
            check_key: str = lambda key: (prefix in key)
            filter_data = [
                concatenate_datasets(data_dict.pop(key)) for key in list(data_dict.keys()) if check_key(key)
            ]
            data_ls.extend(filter_data)
        dataset = concatenate_datasets(data_ls)
        dataset.set_format("torch")

        return dataset

    def set_wandb() -> None:
        # TODO: 이건 나중에 args로 바꿀 것
        GLOBAL_LOGGER.run.log_code(
            train_args.wandb_code_log_dir,
            include_fn=lambda path: path.endswith(".py") or path.endswith(".json"),
        )
        # logging args
        combined_dict = {**train_args.to_dict()}
        if hasattr(model, "config") and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}

        GLOBAL_LOGGER.config.update(combined_dict, allow_val_change=True)

        # set default metrics
        if getattr(GLOBAL_LOGGER, "define_metric", None):
            GLOBAL_LOGGER.define_metric("train/global_step")
            GLOBAL_LOGGER.define_metric("*", step_metric="train/global_step", step_sync=True)

        # set model watch
        _watch_model = os.getenv("WANDB_WATCH", "false")
        if not is_torch_xla_available() and _watch_model in ("all", "parameters", "gradients"):
            GLOBAL_LOGGER.watch(model, log=_watch_model, log_freq=max(100, train_args.logging_steps))
        GLOBAL_LOGGER.run._label(code="transformers_trainer")

    model = TransformerTransducerForRNNT.from_pretrained(train_args.model_name_or_path)
    model = model.to("cpu")
    processor = TransformerTransducerProcessor.from_pretrained(train_args.model_name_or_path)

    if GLOBAL_LOGGER and is_main_process(train_args.local_rank):
        set_wandb()

    # load dataset & preprocess
    data_dict = dict()
    for dataset_name in train_args.dataset_repo_ls:
        logger.info(f"load-{dataset_name}")
        dataset = load_dataset(dataset_name)

        # DatasetDict이라서 이런식으로 해줘야 함.
        column_names = set(sum(dataset.column_names.values(), []))
        with train_args.main_process_first(desc="data preprocess"):
            cache_file_name = None
            if train_args.cache_file_name:
                get_cache_path: str = lambda x: os.path.join(
                    train_args.cache_dir,
                    f"{name}-{x}_{train_args.cache_file_name}",
                )
                name = dataset_name.split("/")[-1]
                cache_file_name = {x: get_cache_path(x) for x in dataset}

            dataset = dataset.map(
                preprocessor,
                num_proc=train_args.preprocessing_num_workers,
                load_from_cache_file=True,
                batched=train_args.preprocessing_batched,
                cache_file_names=cache_file_name,
                batch_size=train_args.preprocessing_batch_size,
                remove_columns=column_names,
                desc=f"preprocess-{dataset_name}",
            )

        for data_key in dataset:
            if data_key not in data_dict:
                data_dict[data_key] = []

            specific_dataset = dataset[data_key]

            added_data = [f"{dataset_name}-{data_key}"] * len(specific_dataset)
            specific_dataset = specific_dataset.add_column("dataset_name", added_data)

            data_dict[data_key].append(specific_dataset)

    train_dataset = None
    if train_args.do_train:
        train_dataset = collect_dataset(train_args.train_dataset_prefix)
        if is_main_process(train_args.local_rank) and train_dataset:
            train_total_length = sum(train_dataset["length"])
            logger.info("train_dataset")
            logger.info(train_dataset)
            logger.info(f"train_total_hour: {(train_total_length / 16000) / 60**2:.2f}h")

    valid_dataset = None
    if train_args.do_eval:
        valid_dataset = collect_dataset(train_args.valid_dataset_prefix)

        if is_main_process(train_args.local_rank) and valid_dataset:
            valid_total_length = sum(valid_dataset["length"])
            logger.info("valid_dataset")
            logger.info(valid_dataset)
            logger.info(f"valid_total_hour: {(valid_total_length / 16000) / 60**2:.2f}h")

    test_dataset = None
    if train_args.do_predict:
        test_dataset = collect_dataset(train_args.test_dataset_prefix)
        if is_main_process(train_args.local_rank) and test_dataset:
            test_total_length = sum(test_dataset["length"])
            logger.info("test_dataset")
            logger.info(test_dataset)
            logger.info(f"test_total_hour: {(test_total_length / 16000) / 60**2:.2f}h")

    collator = DataCollatorRNNTWithPadding(
        processor=processor,
        sampling_rate=16000,
    )
    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    # set trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    if train_args.do_train and train_dataset:
        train(trainer)

    if train_args.do_eval and valid_dataset:
        valid(trainer)

    if train_args.do_predict and test_dataset:
        predict(trainer, test_dataset)


def train(trainer: Trainer) -> None:
    train_args: TransformerTransducerProcessor = trainer.args
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)
    trainer.save_metrics(save_dir)


@torch.no_grad()
def valid(trainer: Trainer, valid_datasets: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


@torch.no_grad()
def predict(trainer: Trainer, test_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    test_dataset_dict = dict()
    test_name_ls = test_dataset["dataset_name"]
    for dataset_name in set(test_name_ls):
        part_idx = [idx for idx, x in enumerate(test_name_ls) if x == dataset_name]
        part_dataset = test_dataset.select(part_idx, keep_in_memory=False)

        # 'jp1924/KconfSpeech-validation'
        start = dataset_name.rindex("/") + 1
        end = dataset_name.rindex("-")

        outputs = trainer.predict(part_dataset, metric_key_prefix=f"test/{dataset_name[start:]}")
        # NOTE: trainer.log를 사용하면 train/test 처럼 찍혀서 나와서 wandb로 직접 찍음
        if GLOBAL_LOGGER:
            GLOBAL_LOGGER.log(outputs.metrics)
        test_dataset_dict[dataset_name[start:end]] = part_dataset


if __name__ == "__main__":
    parser = HfArgumentParser([TransformerTransducerArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if is_main_process(train_args.local_rank):
        logger.info(remain_args)

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    check_wandb = ("wandb" in train_args.report_to) and is_main_process(train_args.local_rank)
    if is_wandb_available() and check_wandb:
        import wandb

        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            group=os.getenv("WANDB_RUN_GROUP"),
            name=train_args.run_name,
            save_code=True,
        )
        GLOBAL_LOGGER = wandb

    main(train_args)

    if GLOBAL_LOGGER:
        GLOBAL_LOGGER.finish()
