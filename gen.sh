
python gen.py --generation_path="results/sd15/dpo/"

python gen.py --generation_path="results/sd15/dpo+npo/" --npo_lora_path="weights/sd15/sd15_beta500_2kiter.safetensors" --merge_weight=0.0
