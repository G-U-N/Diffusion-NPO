PREFIX=npov2-xl-1e-7-1k
MODEL_DIR="/mnt2/wangfuyun/models/stable-diffusion-xl-base-1.0"
OUTPUT_DIR="outputs_xl/$PREFIX"
PROJ_NAME="$PREFIX"
accelerate launch --main_process_port 29506  train_xl.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_nam=$PROJ_NAME \
    --pretrained_vae_model_name_or_path="/mnt2/wangfuyun/models/sdxl-vae-fp16-fix" \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --learning_rate=1e-7 --loss_type="huber" --adam_weight_decay=0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=16 \
    --validation_steps=20 \
    --checkpointing_steps=50 --checkpoints_total_limit=20 \
    --train_batch_size=10 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --max_grad_norm=1 \
    --gradient_checkpointing