python gen_xl.py --generation_path="results/sdxl_cfg5/origin/"  --merge_weight=0.0  --cfg=5
python gen_xl.py --generation_path="results/sdxl_cfg5/origin+npo/" --npo_lora_path="weights/sdxl/sdxl_beta2k_2kiter.safetensors" --merge_weight=0.0  --cfg=5


python gen_xl.py --generation_path="results/sdxl_cfg5/dpo/"  --merge_weight=0.0  --cfg=5
python gen_xl.py --generation_path="results/sdxl_cfg5/dpo+npo/" --npo_lora_path="weights/sdxl/sdxl_beta2k_2kiter.safetensors" --merge_weight=0.0  --cfg=5


python gen_xl.py --generation_path="results/sdxl_cfg5/spo/"  --cfg=5 --num_inference_steps=20
python gen_xl.py --generation_path="results/sdxl_cfg5/spo+npo/"  --cfg=5 --num_inference_steps=20 --npo_lora_path="weights/sdxl/spo_npo_10ep.safetensors"
