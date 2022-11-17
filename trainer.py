from argparse import Namespace
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from transformers import Trainer
from transformers.trainer_utils import (
    EvalLoopOutput,
    is_torch_tpu_available,
    has_length,
    is_main_process,
)
from transformers.trainer_pt_utils import find_batch_size
from transformers.utils import logging
from typing import Optional, List


if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


class TrnasducerTrainer(Trainer):
    """
    RNN-R loss를 사용하는 Transcuder개열의 모델들은 infenrence를 다른 방법으로 진행하기 때문에 evaluate를 다른 방법으로 진행해야 한다.
    """

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            from transformers.deepspeed import deepspeed_init

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()


class TorchTraner:
    def __init__(self, model, optimizer, tokenizer, train_data, valid_data, collator, args: Namespace) -> None:
        self.train_data = train_data
        self.valid_data = valid_data
        self.collator = collator
        self.args = args
        self.optimizer = optimizer

        if args.local_rank != -1:
            local_rank = self.args.local_rank
            dist.init_process_group(backend="nccl", rank=local_rank)
            self.model = DistributedDataParallel(model, device_ids=[args.local_rank])
        else:
            self.model = model
            torch.cuda.set_device(1)
        return

    def train(self) -> None:
        self.gradient_accumulate: int = lambda step: (step % self.args.train_gradient_accumulation) == 0
        return

    def train_step(self, step, train_data) -> None:

        outputs = self.model(**train_data)
        loss = outputs.loss

        if self.gradient_accumulate(step):
            loss.backward()
            self.optimizer.step()

        return

    def valid(self) -> None:
        return

    def valid_step(self) -> None:
        return

    def predict(self) -> None:
        return

    def predict_step(self) -> None:
        return

    def set_train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            dataset=self.train_data,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            batch_sampler=None,
            collate_fn=self.collator,
            pin_memory=True,
        )

        return

    def set_valid_datalaoder(self) -> DataLoader:
        valid_dataloader = DataLoader(
            dataset=self.valid_data,
            batch_size=self.args.valid_batch_size,
            shuffle=False,
            batch_sampler=None,
            collate_fn=self.collator,
            pin_memory=True,
        )

        return
