accelerate launch --main_process_port=29501 --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_spo_sdxl.py --config configs/spo_sdxl_4k-prompts_num-sam-2_3-is_10ep_bs2_gradacc2.py
