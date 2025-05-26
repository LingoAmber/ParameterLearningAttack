#!/bin/bash

deepspeed --include localhost:3,4,5,6 --master_port=25641 llava/train/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data2/yubowang/llava1.5-7b \
    --version v1 \
    --data_path /data2/yubowang/PaintingForm/train_top_20k.json \
    --image_folder /data2/yubowang/PaintingForm/art_images_data/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data2/yubowang/llava_lora_ckpt3_paintingform_rank16_epoch3 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epochs" \
    --save_only_model True \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    # --report_to tensorboard
