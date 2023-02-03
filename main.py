import io
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
import soundfile as sf
import torch  # it's for debugging
from data import TransformerTransducerCollator, TransformerTransducerFeatureExtractor, TransformerTransducerTokenizer
from evaluate import load
from model import TransformerTransducerConfig, TransformerTransducerForRNNT
from setproctitle import setproctitle
from torch.optim import AdamW
from transformers import HfArgumentParser, Seq2SeqTrainer, Wav2Vec2CTCTokenizer, set_seed
from transformers.integrations import WandbCallback
from transformers.trainer_utils import EvalPrediction, is_main_process
from transformers.utils import logging
from utils import (
    DataArguments,
    GaussianNoiseCallback,
    ModelArguments,
    TransducerTrainArgument,
    get_tri_stage_scheduler_with_warmup,
)

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


def main(parser: HfArgumentParser) -> None:
    train_args, data_args, model_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle(train_args.run_name)
    set_seed(train_args.seed)

    def metrics(evaluation_result: EvalPrediction) -> Dict[str, float]:
        """doc"""
        result = dict()
        predicts = evaluation_result.predictions
        predicts = np.where(predicts != -100, predicts, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(predicts, skip_special_tokens=True)

        labels = evaluation_result.label_ids
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)

        wer_score = wer._compute(predictions, references)
        cer_score = cer._compute(predictions, references)
        result = {
            "wer": wer_score,
            "cer": cer_score,
        }
        return result

    def preprocessor(dataset: datasets.Dataset) -> Dict[str, Any]:
        """doc"""
        byte_audio = dataset.data["audio"]["bytes"]
        str_txt = dataset.data["text"]
        raw_audio, sr = sf.read(io.BytesIO(byte_audio))

        # [NOTE]: outout is [shape(time_seq, mel_seq), ...]
        log_mel = extractor.log_mel_transform(raw_audio, do_numpy=True)
        log_mel = extractor.mel_compressor(log_mel[0])
        int_txt = tokenizer.encode(str_txt)

        # [NOTE]: length for sampler
        time_len = len(log_mel)

        processing_result = {
            "audio": log_mel,
            "text": int_txt,
            "length": time_len,
        }
        return processing_result

    def data_preprocessing(data_type: str) -> datasets.Dataset:
        """doc"""
        concat_group = [asr_data.pop(data_name) for data_name in list(asr_data) if data_type in data_name]
        data = datasets.concatenate_datasets(concat_group)

        cache_file_name = f"{data_args.data_name}_{data_type}.arrow"
        cache_save_path = os.path.join(model_args.cache_dir, data_args.data_name, data_type, cache_file_name)

        if not os.path.exists(cache_save_path):
            os.makedirs(cache_save_path, exist_ok=True)

        data = data.map(
            preprocessor,
            batch_size=1,
            desc=data_type,
            load_from_cache_file=True,
            num_proc=data_args.num_proc,
            cache_file_name=cache_save_path,
        )

        data = data.rename_column("text", "labels")
        data = data.rename_column("audio", "input_features")
        return data

    load_name = train_args.resume_from_checkpoint or model_args.model_name_or_path
    tokenizer_name = train_args.vocab_path or load_name

    logger.info("\n---- set tokenizer & extractor ----")
    tokenizer = TransformerTransducerTokenizer.from_pretrained(tokenizer_name, cache_dir=model_args.cache_dir)
    extractor = TransformerTransducerFeatureExtractor(
        n_fft=data_args.num_fourier,
        feature_size=data_args.mel_shape,
        hop_length=data_args.hop_length,
        stack=data_args.mel_stack,
        stride=data_args.window_stride,
        sampling_rate=data_args.sampling_rate,
    )
    # [NOTE]: process does not worked, it's not added transformers init
    # processor = TransducerProcessor(feature_extractor=extractor, tokenizer=tokenizer)

    logger.info("\n---- set model & config ----")
    if load_name:
        config = TransformerTransducerConfig.from_pretrained(load_name, cache_dir=model_args.cache_dir)
        model = TransformerTransducerForRNNT.from_pretrained(load_name, cache_dir=model_args.cache_dir, config=config)
    else:
        config = TransformerTransducerConfig(
            vocab_size=tokenizer.vocab_size,
            is_encoder_decoder=True,
            decoder_start_token_id=0,
        )
        model = TransformerTransducerForRNNT(config)

    logger.info("\n---- load data ----")
    asr_data = datasets.load_dataset(data_args.data_name, cache_dir=model_args.cache_dir)

    logger.info("\n---- data preprocessing ----")
    asr_data.pop("validation.other")

    train_data = data_preprocessing("train") if train_args.do_train else None
    valid_data = data_preprocessing("validation.clean") if train_args.do_eval else None
    clean_data = data_preprocessing("clean") if train_args.do_predict else None
    other_data = data_preprocessing("other") if train_args.do_predict else None

    logger.info("\n---- load metrics ----")
    wer = load("evaluate-metric/wer", cache_dir=model_args.cache_dir)
    cer = load("evaluate-metric/cer", cache_dir=model_args.cache_dir)

    logger.info("\n---- set optimizer & scheduler ----")
    if train_args.max_steps != -1:
        optimizer = AdamW(
            params=model.parameters(),
            lr=train_args.learning_rate,
            betas=(train_args.adam_beta1, train_args.adam_beta2),
            eps=train_args.adam_epsilon,
            weight_decay=train_args.weight_decay,
        )
        scheduler = get_tri_stage_scheduler_with_warmup(
            optimizer=optimizer,
            num_training_steps=train_args.max_steps,
            final_lr=model_args.final_learning_rate,
            num_warmup_steps=model_args.ramp_up_step_ratio,
            num_hold_steps=model_args.hold_step_ratio,
            num_decay_steps=model_args.decay_step_ratio,
        )
        optimizers = (optimizer, scheduler)
    else:
        optimizers = (None, None)

    logger.info("\n---- set trainer ----")
    collator = TransformerTransducerCollator(
        tokenizer,
        extractor=extractor,
        blank_id=config.blk_token_id,
    )
    callbacks = [WandbCallback] if os.getenv("WANDB_DISABLED") == "false" else []
    callbacks.append(GaussianNoiseCallback)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizers=optimizers,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=train_args,
        compute_metrics=metrics,
        data_collator=collator,
        callbacks=callbacks,
    )

    if train_args.do_train:
        if train_args.do_fist_predict:
            logger.info("\n---- run fisrt predict ----")
            predict(trainer, clean_data, train_args, "clean")
            predict(trainer, other_data, train_args, "other")

        logger.info("\n---- run train ----")
        train(trainer, train_args)

    if train_args.do_eval:
        logger.info("\n---- run eval ----")
        eval(trainer, valid_data, train_args)

    if train_args.do_predict:
        logger.info("\n---- run predict ----")
        predict(trainer, clean_data, train_args, "clean", save=True)
        predict(trainer, other_data, train_args, "other", save=True)


def train(trainer: Seq2SeqTrainer, args: Namespace) -> None:
    """doc"""
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if is_main_process(args.local_rank):
        model_name = trainer.model.name_or_path
        save_path = os.path.join(args.output_dir, model_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        metrics = outputs.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model(save_path)


def eval(trainer: Seq2SeqTrainer, dataset: datasets.Dataset, args: Namespace) -> None:
    """doc"""
    metrics = trainer.evaluate(dataset)
    if is_main_process(args.local_rank):
        trainer.log_metrics("valid", metrics)
        trainer.save_metrics("valid", metrics)


def predict(
    trainer: Seq2SeqTrainer,
    dataset: datasets.Dataset,
    args: Namespace,
    prefix: Optional[str] = None,
    save: bool = False,
) -> None:
    """doc"""
    outputs = trainer.predict(dataset, metric_key_prefix=prefix)
    metrics = outputs.metrics
    logger.info(f"{prefix}: {metrics}")

    if is_main_process(args.local_rank) and save:
        predictions = metrics.predictions
        references = metrics.label_ids

        model_name = trainer.model.name_or_path
        save_path = os.path.join(args.output_dir, model_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        predictions_save_path = os.path.join(save_path, f"{prefix}_predictions.txt")
        with open(predictions_save_path, "w", encoding="utf-8") as predictions_file:
            predictions_file.writelines(predictions)

        references_save_path = os.path.join(save_path, f"{prefix}_references.txt")
        with open(references_save_path, "w", encoding="utf-8") as references_file:
            references_file(references)

        trainer.log_metrics(prefix, metrics)
        trainer.save_metrics(prefix, metrics)


if "__main__" in __name__:
    parser = HfArgumentParser([TransducerTrainArgument, DataArguments, ModelArguments])
    main(parser)
