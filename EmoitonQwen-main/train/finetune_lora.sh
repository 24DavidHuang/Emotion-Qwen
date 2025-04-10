#!/bin/bash

GPUS_PER_NODE=3 # if has multi-GPUs ,adjust this
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL_PATH="/path/to/the/model"
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
DATA_JSON="/path/to/the/train.json"
IMAGE_DIR="/path/to/the/video/or/image/folder"
# EVAL_DATA="/home/dell/LLaVA/playground/data/Visual_Instruction_Tuning/llava_v1_5_mix665k_formart.json"
MODEL_MAX_Length=2048 # if conduct video sft, please set MODEL_MAX_Length >= 14336
OUTPUT_MODEL_NAME="EmotionQwen_lora"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# if has multi-GPUs ,set CUDA_VISIBLE_DEVICES=0,1,....,n
CUDA_VISIBLE_DEVICES=0,1 torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL_PATH \
    --image_folder $IMAGE_DIR \
    --data_path $DATA_JSON \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --tune_vision false \
    --tune_llm false \
    --tune_general_compressor false \
    --tune_emotion_compressor false \
    --disable_flash_attn2 false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length $MODEL_MAX_Length \
    --image_max_pixels $((1280 * 28 * 28)) \
    --video_max_pixels $((448 * 448)) \
    --output_dir lora_output/$OUTPUT_MODEL_NAME/model \
    --logging_dir lora_output/$OUTPUT_MODEL_NAME/log \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --dataloader_num_workers 3 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "tensorboard" # wandb
    # --per_device_eval_batch_size 4 \
    # --eval_steps 1000 \
    # --evaluation_strategy "steps" \
    # --max_steps 20000 \
    # --do_eval \
    # --eval_data_path $EVAL_DATA \
