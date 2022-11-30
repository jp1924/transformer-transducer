import os
from argparse import Namespace
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from data import TransducerCollator, TransducerFeatureExtractor, get_concat_dataset
from datasets import Dataset
from evaluate import load
from model import TransformerTranducerForRNNT, TransformerTransducerConfig
from setproctitle import setproctitle
from transformers import HfArgumentParser, Trainer, Wav2Vec2Tokenizer, set_seed
from transformers.integrations import WandbCallback
from transformers.trainer_utils import EvalPrediction, is_main_process
from utils import DataArguments, ModelArguments, TransducerTrainArgument


def main(parser: HfArgumentParser) -> None:
    train_args, data_args, model_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle(train_args.run_name)
    set_seed(train_args.seed)

    def mel_preprocess(dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        label은 이미 encoding 되었다고 가정함.
        """
        mel_feature = dataset["input_values"]
        time_steps, features_dim = mel_feature.shape

        # [NOTE]: for explicitness
        max_length = train_args.mel_max_length
        stack = train_args.mel_stack
        stride = train_args.window_stride

        mel_store = list()
        for step in range(stack):
            indices = [step + idx for idx in range(0, (time_steps - step), stride)]
            features = mel_feature[indices[:max_length]]
            # [TODO]: 이 부분은 extractor의 pad 기능이 하는게 맞지만 임시로 이렇게 만든다.
            pad_width = ((0, max_length - features.shape[0]), (0, 0))
            features = np.pad(features, pad_width)

            mel_store.append(features)

        padded_feature = np.concatenate(mel_store, axis=1)
        padded_feature = padded_feature.transpose()

        dataset["input_values"] = padded_feature
        return dataset

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

    def logits_for_metrics(logits: Union[Tuple, torch.Tensor], _) -> torch.Tensor:
        """_logits_for_metrics_
            Trainer.evaluation_loop에서 사용되는 함수로 logits를 argmax를 이용해
            축소 시켜 공간복잡도를 줄이기 위한 목적으로 작성되었습니다.
        Args:
            logits (Union[Tuple, torch.Tensor]): Model을 거쳐서 나온 3차원 (bch, sqr, hdn)의 logits을 전달받습니다.
            _ : label이 입력되는 부분이지만 사용되지 않기에 하이픈처리 했습니다.
        Returns:
            torch.Tensor: 차원을 축소한 뒤의 torch.Tensor를 반환합니다.
        """
        return_logits = logits.argmax(dim=-1)
        return return_logits

    tokenizer = Wav2Vec2Tokenizer.from_pretrained("test42/kerberus2", use_auth_token=True)
    config = TransformerTransducerConfig(tokenizer.vocab_size)
    extractor = TransducerFeatureExtractor(0, 16000, 0)
    model = TransformerTranducerForRNNT(config)

    # [NOTE]: temp
    train_data = get_concat_dataset([data_args.data_name], "train") if train_args.do_train else None
    valid_data = get_concat_dataset([data_args.data_name], "dev") if train_args.do_eval else None
    test_data = get_concat_dataset([data_args.data_name], "eval_other") if train_args.do_predict else valid_data

    train_data = train_data.rename_column("input_ids", "labels") if train_args.do_train else None
    valid_data = valid_data.rename_column("input_ids", "labels") if train_args.do_eval else None

    wer = load("evaluate-metric/wer")

    collator = TransducerCollator(
        tokenizer,
        train_args.mel_max_length,
        train_args.mel_stack,
        train_args.window_stride,
    )
    callbacks = [WandbCallback] if os.getenv("WANDB_DISABLED") == "false" else None

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=train_args,
        # compute_metrics=metrics,
        data_collator=collator,
        callbacks=callbacks,
        preprocess_logits_for_metrics=logits_for_metrics,
    )
    if train_args.do_train:
        train(trainer, train_args)
    if train_args.do_eval:
        eval(trainer, valid_data, train_args)
    if train_args.do_predict:
        predict(trainer, test_data)


def train(trainer: Trainer, args: Namespace) -> None:
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
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
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


def eval(trainer: Trainer, eval_data: Dataset, args: Namespace) -> None:
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


def predict(trainer: Trainer, test_data: Dataset) -> None:
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
    # # [NOTE]: ---------------------------------------------

    """
        test_values = train_data[25]["input_values"]
        time_steps, features_dim = test_values.shape

        stack = 4
        stride = 3

        max_length = 400

        mel_store = list()
        time_steps, features_dim = test_values.shape
        for step in range(stack):
            indices = [step + idx for idx in range(0, (time_steps - step), stride)]
            features = test_values[indices[:max_length]]
            # [TODO]: 이 부분은 extractor의 pad 기능이 하는게 맞지만 임시로 이렇게 만든다.
            pad_width = ((0, max_length - features.shape[0]), (0, 0))
            features = np.pad(features, pad_width)
            mel_store.append(features)
        np.concatenate(mel_store)

        zero_feature_dim = features_dim * (1 + 3 + 0)
        concated_features = np.zeros(shape=[time_steps, zero_feature_dim], dtype=np.float32)
        concated_features[:, 3 * features_dim : (3 + 1) * features_dim] = test_values
        left_context_width = 4

        for i in range(left_context_width):
            # add left context
            concated_features[
                (i + 1) : time_steps,
                ((left_context_width - i - 1) * features_dim) : ((left_context_width - i) * features_dim),
            ] = test_values[0 : time_steps - i - 1, :]

        interval = 3
        temp_mat = [concated_features[i] for i in range(1, concated_features.shape[0], interval)]
        subsampled_features = np.row_stack(temp_mat)

        plt.imsave("/home/jsb193/subsampled_features.png", subsampled_features[:400])    
    
    """

    # # [NOTE]: ---------------------------------------------
    # plt.imsave("/home/jsb193/check_mel/test_asdasdasd1.png", padded_inputs)
