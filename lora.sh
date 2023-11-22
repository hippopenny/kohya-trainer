accelerate launch --config_file=C:\\hippopenny\\kohya-trainer\\default_config.yaml \
    --num_cpu_threads_per_process=8 \
    train_network.py \
    --pretrained_model_name_or_path="C:\\hippopenny\\sd-scripts\\pretrained\\stable_diffusion_1_5-pruned.safetensors" \
    --train_data_dir="C:\hippopenny\sd-scripts\train_data" \
    --output_name=hppg \
    --output_dir="C:\\hippopenny\\sd-scripts\\output\\LoRA" \
    --log_prefix=hppg \
    --logging_dir="C:\\hippopenny\\sd-scripts\\logs" \
    --sample_prompts="C:\\hippopenny\\sd-scripts\\LoRA\\config\\sample_prompt.txt" \
    --sample_every_n_steps=1 \
    --max_train_epochs=50 \
    --train_batch_size=1 \
    --save_n_epoch_ratio=3 \
    --caption_extension=.txt \
    --prior_loss_weight=1.0 --mixed_precision=fp16 --save_precision=fp16 --save_model_as=safetensors \
    --resolution=512 --enable_bucket --min_bucket_reso=256 --max_bucket_reso=1024 --cache_latents \
    --max_token_length=225 --use_8bit_adam --gradient_accumulation_steps=1 \
    --clip_skip=2 --network_dim=32 --network_alpha=32 --network_module=networks.lora --learning_rate=0.0001 \
    --text_encoder_lr=5e-05 --training_comment=this_comment_will_be_stored_in_the_metadata --lr_scheduler=constant --shuffle_caption --xformers