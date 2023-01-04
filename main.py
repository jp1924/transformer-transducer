import io
import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional
from unicodedata import normalize

import datasets
import numpy as np
import soundfile as sf
import torch
from torch.optim import AdamW
from data import TransducerCollator, TransducerFeatureExtractor, TransducerTokenizer
from evaluate import load
from model import TransformerTranducerForRNNT, TransformerTransducerConfig
from setproctitle import setproctitle
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    set_seed,
)
from transformers.integrations import WandbCallback
from transformers.trainer_utils import EvalPrediction, is_main_process
from utils import DataArguments, ModelArguments, TransducerTrainArgument, TriStageLRScheduler

from trainer import TransducerTrainer

logger = logging.getLogger(__name__)


def main(parser: HfArgumentParser) -> None:
    train_args, data_args, model_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle(train_args.run_name)
    set_seed(train_args.seed)

    def metrics(evaluation_result: EvalPrediction) -> Dict[str, float]:
        """_metrics_
            evaluation과정에서 모델의 성능을 측정하기 위한 metric을 수행하는 함수 입니다.
            이 함수는 Trainer에 의해 실행되며 Huggingface의 Evaluate 페키로 부터
            각종 metric을 전달받아 계산한 뒤 결과를 반환합니다.
        Args:
            evaluation_result (EvalPrediction): Trainer.evaluation_loop에서 model을 통해 계산된
            logits과 label을 전달받습니다.
        Returns:
            Dict[str, float]: metrics 계산결과를 dict로 반환합니다.
        """

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
        byte_audio = dataset.data["audio"]["bytes"]
        str_txt = dataset.data["text"]
        raw_audio, sr = sf.read(io.BytesIO(byte_audio))

        # outout: [shape(time_seq, mel_seq), ...]
        log_mel = extractor.log_mel_transform(raw_audio, do_numpy=True)
        log_mel = extractor.mel_compressor(log_mel[0])
        int_txt = tokenizer.encode(str_txt)

        time_len = len(log_mel)

        processing_result = {
            "audio": log_mel,
            "text": int_txt,
            "length": time_len,
        }
        return processing_result

    def data_preprocessing(data_type: str) -> datasets.Dataset:
        logger.info(f"------------ {data_type}_data preprocessing ------------")

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

    tokenizer = TransducerTokenizer.from_pretrained(tokenizer_name, cache_dir=model_args.cache_dir)
    extractor = TransducerFeatureExtractor(
        n_fft=data_args.num_fourier,
        feature_size=data_args.mel_shape,
        hop_length=data_args.hop_length,
        stack=data_args.mel_stack,
        stride=data_args.window_stride,
    )
    # [TODO]: 나중에 processor추가하기
    # processor = TransducerProcessor(feature_extractor=extractor, tokenizer=tokenizer)
    if load_name:
        config = TransformerTransducerConfig.from_pretrained(load_name, cache_dir=model_args.cache_dir)
        model = TransformerTranducerForRNNT.from_pretrained(load_name, config=config, cache_dir=model_args.cache_dir)
    else:
        config = TransformerTransducerConfig(vocab_size=tokenizer.vocab_size)
        model = TransformerTranducerForRNNT(config)

    # [NOTE]: data load
    asr_data = datasets.load_dataset(data_args.data_name, cache_dir=model_args.cache_dir)

    # [NOTE]: data processing
    train_data = data_preprocessing("train") if train_args.do_train else None
    valid_data = data_preprocessing("valid") if train_args.do_eval else None
    clean_data = data_preprocessing("clean") if train_args.do_predict else None
    other_data = data_preprocessing("other") if train_args.do_predict else None

    # [NOTE]: set metrics
    wer = load("evaluate-metric/wer", cache_dir=model_args.cache_dir)
    cer = load("evaluate-metric/cer", cache_dir=model_args.cache_dir)

    # [NOTE]: set optimizers & scheduler
    if train_args.max_steps != -1:
        optimizer = AdamW(
            params=model.parameters(),
            lr=train_args.learning_rate,
            betas=(train_args.adam_beta1, train_args.adam_beta2),
            eps=train_args.adam_epsilon,
            weight_decay=train_args.weight_decay,
        )
        tri_stage = TriStageLRScheduler(
            model_args,
            max_steps=train_args.max_steps,
            learning_rate=train_args.learning_rate,
        )
        scheduler = tri_stage.get_tri_stage_scheduler(optimizer)
        optimizers = (optimizer, scheduler)
    else:
        optimizers = (None, None)

    # [NOTE]: set Trainer
    collator = TransducerCollator(
        tokenizer,
        extractor=extractor,
        blank_id=config.blk_token_id,
    )
    callbacks = [WandbCallback] if os.getenv("WANDB_DISABLED") == "false" else None
    trainer = TransducerTrainer(
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
        train(trainer, train_args)
    if train_args.do_eval:
        eval(trainer, valid_data, train_args)
    if train_args.do_predict:
        predict(trainer, clean_data, train_args, "clean")
        predict(trainer, other_data, train_args, "other")


def train(trainer: Seq2SeqTrainer, args: Namespace) -> None:
    """_train_
        Trainer를 전달받아 Trainer.train을 실행시키는 함수입니다.
        학습이 끝난 이후 학습 결과 그리고 최종 모델을 저장하는 기능도 합니다.
        만약 학습을 특정 시점에 재시작 하고 싶다면 Seq2SeqTrainingArgument의
        resume_from_checkpoint을 True혹은 PathLike한 값을 넣어주세요.
        - huggingface.trainer.checkpoint
        https://huggingface.co/docs/transformers/main_classes/trainer#checkpoints
    Args:
        trainer (Seq2SeqTrainer): Huggingface의 torch Seq2SeqTrainer를 전달받습니다.
        args (Namespace): Seq2SeqTrainingArgument를 전달받습니다.
    """
    # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, use_cuda=True) as prof:
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # prof.export_chrome_trace("trace.json")
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
    """_eval_
        Trainer를 전달받아 Trainer.eval을 실행시키는 함수입니다.
    Args:
        trainer (Seq2SeqTrainer): Huggingface의 torch Seq2SeqTrainer를 전달받습니다.
        dataset (Dataset): 검증을 하기 위한 Data를 전달받습니다.
    """
    metrics = trainer.evaluate(dataset)
    if is_main_process(args.local_rank):
        trainer.log_metrics("valid", metrics)
        trainer.save_metrics("valid", metrics)


def predict(trainer: Seq2SeqTrainer, dataset: datasets.Dataset, args: Namespace, prefix: Optional[str] = None) -> None:
    """_predict_
        Trainer를 전달받아 Trainer.predict을 실행시키는 함수입니다.
        이때 Seq2SeqTrainer의 Predict이 실행되며 model.generator를 실행시키기 위해
        arg값의 predict_with_generater값을 강제로 True로 변환시킵니다.
    Args:
        trainer (Seq2SeqTrainer): Huggingface의 torch Seq2SeqTrainer를 전달받습니다.
        dataset (Dataset): 검증을 하기 위한 Data를 전달받습니다.
        gen_kwargs (Dict[str, Any]): model.generator를 위한 값들을 전달받습니다.
    """
    outputs = trainer.predict(dataset, key_prefix=prefix)
    metrics = outputs.metrics

    predictions = metrics.predictions
    references = metrics.label_ids

    if is_main_process(args.local_rank):
        predictions
        references

        trainer.log_metrics(prefix, metrics)
        trainer.save_metrics(prefix, metrics)


if "__main__" in __name__:
    parser = HfArgumentParser([TransducerTrainArgument, DataArguments, ModelArguments])
    main(parser)
