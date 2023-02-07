script_path=""
output_dir=""
data_path=""
cache_dir=""

num_gpu=4

export CUDA_VISIBLE_DEVICES=""
export WANDB_DISABLED="false"
export WANDB_PROJECT=""
export WANDB_ENTITY=""
export WANDB_RUN_GROUP=""
export WANDB_TAGS=""
export WANDB_DISABLE_CODE="false"
# export WANDB_RESUME=""
# export WANDB_RUN_ID=""
export OMP_NUM_THREADS=4

export TORCH_DISTRIBUTED_DEBUG="DETAIL"

python -m torch.distributed.launch --standalone --nnodes=1 --nproc_per_node=$num_gpu \
    $script_path \
    --output_dir=$output_dir \
    --run_name="" \
    --cache=$cache_dir \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=128 \
    --per_device_eval_batch_size=1 \
    --eval_accumulation_steps=2 \
    --max_steps=200000 \
    --learning_rate=2.5e-4 \
    --ramp_up_step_ratio=0.02 \
    --hold_step_ratio=0.15 \
    --decay_step_ratio=0.83 \
    --final_learning_rate=2.5e-6 \
    --logging_strategy="steps" \
    --evaluation_strategy="steps" \
    --save_strategy="steps" \
    --logging_steps=10 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --noise_step=10000 \
    --noise_mean=0.0 \
    --noise_std=0.01 \
    --data_name=$data_path \
    --num_proc=2 \
    --mel_stack=4 \
    --window_stride=3 \
    --num_fourier=512 \
    --mel_shape=128 \
    --hop_length=128 \
    --sampling_rate=16000 \
    --do_fist_predict false \
    --do_train true \
    --do_eval true \
    --do_predict true \
    --group_by_length false \
    --fp16 true \
    --label_names="labels" \
    --disable_tqdm true \
    --predict_with_generate true \