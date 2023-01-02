script_path=""
output_dir=""
data_path=""
cache_dir=""

num_gpu=4

export CUDA_VISIBLE_DEVICES=""
export WANDB_DISABLED=""
export WANDB_PROJECT=""
export WANDB_ENTITY=""
export WANDB_RUN_GROUP=""
export WANDB_TAGS=""
export WANDB_DISABLE_CODE=""
# export WANDB_RESUME=""
# export WANDB_RUN_ID=""
export OMP_NUM_THREADS=4

export TORCH_DISTRIBUTED_DEBUG="DETAIL"

# [NOTE]: if you need resume training from checkpoint, please insert this argument in this code
# --resume_from_checkpoint=$checkpoint_path \
python -m torch.distributed.launch --standalone --nnodes=1 --nproc_per_node=$num_gpu \
    $script_path \
    --output_dir=$output_dir \
    --run_name="" \
    --cache=$cache_dir \
    --data_name=$data_path \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=1 \
    --eval_accumulation_steps=2 \
    --max_steps=200000 \
    --learning_rate=5e-4 \
    --ramp_up_step_ratio=0.06 \
    --hold_step_ratio=0.11 \
    --decay_step_ratio=0.83 \
    --init_learning_rate=0.0 \
    --final_learning_rate=2.5e-6 \
    --logging_strategy="steps" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --save_strategy="steps" \
    --eval_steps=2000 \
    --save_steps=2000 \
    --do_train true \
    --do_eval true \
    --do_predict true \
    --group_by_length false \
    --fp16 true \
    --label_names="labels" \
    --num_proc=2 \
    --disable_tqdm true \
    --predict_with_generate true \