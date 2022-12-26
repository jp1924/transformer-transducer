import os
from argparse import Namespace
from typing import Any, Dict, Tuple, Union
from unicodedata import normalize

import numpy as np
import torch
from data import TransducerCollator, TransducerFeatureExtractor, TransducerTokenizer
import datasets
from evaluate import load
from model import TransformerTranducerForRNNT, TransformerTransducerConfig
from setproctitle import setproctitle
from transformers import HfArgumentParser, Seq2SeqTrainer, Wav2Vec2Tokenizer, set_seed
from transformers.integrations import WandbCallback
from transformers.trainer_utils import EvalPrediction, is_main_process
from utils import DataArguments, ModelArguments, TransducerTrainArgument


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

        result = {"wer": wer_score}

        return result

    def preprocess(dataset: datasets.Dataset) -> Dict[str, Any]:

        return

    tokenizer = TransducerTokenizer.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=model_args.cache_dir,
    )
    config = TransformerTransducerConfig(tokenizer.vocab_size)
    model = TransformerTranducerForRNNT(config)

    asr_data = datasets.load_dataset(data_args.data_name, cache_dir=model_args.cache_dir)

    data_collate: str = lambda key: [asr_data[data_name] for data_name in asr_data if key in data_name]
    train_data = datasets.concatenate_datasets(data_collate("train")) if train_args.do_train else None
    valid_data = datasets.concatenate_datasets(data_collate("valid")) if train_args.do_eval else None
    test_data = datasets.concatenate_datasets(data_collate("test")) if train_args.do_test else None

    extractor = TransducerFeatureExtractor()
    extractor([audio["array"] for audio in train_data[:2]["audio"]])

    wer = load("evaluate-metric/wer", cache_dir=model_args.cache_dir)
    collator = TransducerCollator(
        tokenizer,
        train_args.mel_max_length,
        train_args.mel_stack,
        train_args.window_stride,
        extractor=extractor,
        blank_id=config.blank_id,
    )

    callbacks = [WandbCallback] if os.getenv("WANDB_DISABLED") == "false" else None
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
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
        predict(trainer, test_data)


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
        trainer.args.output_dir = save_path

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model(save_path)


def eval(trainer: Seq2SeqTrainer, eval_data: datasets.Dataset, args: Namespace) -> None:
    """_eval_
        Trainer를 전달받아 Trainer.eval을 실행시키는 함수입니다.
    Args:
        trainer (Seq2SeqTrainer): Huggingface의 torch Seq2SeqTrainer를 전달받습니다.
        eval_data (Dataset): 검증을 하기 위한 Data를 전달받습니다.
    """
    metrics = trainer.evaluate(eval_data)
    if is_main_process(args.local_rank):
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def predict(trainer: Seq2SeqTrainer, test_data: datasets.Dataset) -> None:
    """_predict_
        Trainer를 전달받아 Trainer.predict을 실행시키는 함수입니다.
        이때 Seq2SeqTrainer의 Predict이 실행되며 model.generator를 실행시키기 위해
        arg값의 predict_with_generater값을 강제로 True로 변환시킵니다.
        True로 변환시키면 model.generator에서 BeamSearch를 진행해 더 질이 좋은 결과물을 만들 수 있습니다.
    Args:
        trainer (Seq2SeqTrainer): Huggingface의 torch Seq2SeqTrainer를 전달받습니다.
        test_data (Dataset): 검증을 하기 위한 Data를 전달받습니다.
        gen_kwargs (Dict[str, Any]): model.generator를 위한 값들을 전달받습니다.
    """
    raise NotImplementedError


if "__main__" in __name__:
    parser = HfArgumentParser([TransducerTrainArgument, DataArguments, ModelArguments])
    # https://code.visualstudio.com/docs/python/debugging#_redirectoutput
    main(parser)
