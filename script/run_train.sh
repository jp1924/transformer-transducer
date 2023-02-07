script_path="/data01/jsb193/github/transducer/transformer-transducer/main.py"
output_dir="/data01/jsb193/github/transducer/output_dir"
data_path="librispeech_asr"
cache_dir="/data01/jsb193/github/transducer/.cache"

num_gpu=4

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_DISABLED="false"
export WANDB_PROJECT="TransformerTransducer"
export WANDB_ENTITY="jp_42maru"
export WANDB_RUN_GROUP="train"
export WANDB_TAGS="rnn-t, streaming, transducer, train, librispeech"
export WANDB_DISABLE_CODE="false"
# export WANDB_RESUME=""
# export WANDB_RUN_ID=""
export OMP_NUM_THREADS=4

export TORCH_DISTRIBUTED_DEBUG="DETAIL"

python -m torch.distributed.launch --standalone --nnodes=1 --nproc_per_node=$num_gpu \
    $script_path \
    --output_dir=$output_dir \
    --run_name="[JP]transducer-11" \
    --vocab_path="/data01/jsb193/github/transducer/transformer-transducer/temp_vocab" \
    --cache=$cache_dir \
    --data_name=$data_path \
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
    --noise_steps=10000 \
    --noise_mean=0.0 \
    --noise_std=0.01 \
    --data_name=librispeech_asr \
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