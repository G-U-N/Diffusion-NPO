from diffusers import UNet2DConditionModel
from pipeline_stable_diffusion_double_unet import StableDiffusionPipeline
import torch
import copy
from datasets import load_dataset
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from safetensors.torch import load_file

num_device = 8
num_processes = 8


def mix_models(module1, module2, alpha=0.5):
    """
    Mix the parameters of two PyTorch modules with a given ratio.

    Args:
        module1 (nn.Module): The first PyTorch module.
        module2 (nn.Module): The second PyTorch module.
        alpha (float): The ratio of mixing, where 0 <= alpha <= 1.
                       alpha=0.5 means equal contribution from both models.

    Returns:
        nn.Module: The mixed PyTorch module (same as module1).
    """
    device = next(module1.parameters()).device
    module2 = module2.to(device)

    for param1, param2 in zip(module1.parameters(), module2.parameters()):
        param1.data = alpha * param1.data + (1 - alpha) * param2.data

    return module1


def load_dpo_ours_pipeline(device, merge_weight=0.0, npo_lora_path=None):
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
    )

    unet_id = "mhdang/dpo-sd1.5-text2image-v1"
    unet = UNet2DConditionModel.from_pretrained(
        unet_id, subfolder="unet", torch_dtype=torch.float16
    )
    pipe.unet = unet
    pipe = pipe.to(device)

    pipe_tmp = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe_tmp = pipe_tmp.to(device)
    pipe_tmp.unet = mix_models(
        pipe_tmp.unet, pipe.unet, merge_weight
    )  # 0.0 to use full dpo weight
    pipe_tmp.load_lora_weights(npo_lora_path)
    pipe_tmp.fuse_lora()

    negative_unet = copy.deepcopy(pipe_tmp.unet)
    del pipe_tmp
    pipe.negative_unet = negative_unet

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


def load_dpo_origin_pipeline(device):
    # load pipeline
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
    )

    # load finetuned model
    unet_id = "mhdang/dpo-sd1.5-text2image-v1"
    unet = UNet2DConditionModel.from_pretrained(
        unet_id, subfolder="unet", torch_dtype=torch.float16
    )
    pipe.unet = unet
    pipe = pipe.to(device)

    pipe_tmp = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe_tmp = pipe_tmp.to(device)
    negative_unet = copy.deepcopy(pipe_tmp.unet)

    del pipe_tmp
    pipe.negative_unet = negative_unet

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


def load_dpo_pipeline(device):
    # load pipeline
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
    )

    # load finetuned model
    unet_id = "mhdang/dpo-sd1.5-text2image-v1"
    unet = UNet2DConditionModel.from_pretrained(
        unet_id, subfolder="unet", torch_dtype=torch.float16
    )
    pipe.unet = unet
    pipe = pipe.to(device)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


def load_dpo_lora_pipeline(device, lora_path=None):
    # load pipeline
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
    )
    pipe.load_lora_weights(lora_path)
    pipe = pipe.to(device)

    pipe_tmp = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe_tmp = pipe_tmp.to(device)
    bad_weight = load_file(
        lora_path,
        "cpu",
    )

    pipe_tmp.load_lora_weights(bad_weight, adapter_name="bad")
    spo_weight = load_file(lora_path)
    pipe_tmp.load_lora_weights(spo_weight, adapter_name="spo")
    pipe_tmp.set_adapters(["bad", "spo"], adapter_weights=[1.0, 1.0])
    negative_unet = copy.deepcopy(pipe_tmp.unet)

    del pipe_tmp

    pipe.negative_unet = negative_unet

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe
    return pipe


def load_origin_pipeline(device):
    # load pipeline
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
    )

    pipe = pipe.to(device)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


def load_dreamshaper_pipeline(device):
    # load pipeline
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
    )

    pipe.unet.load_state_dict(
        torch.load("dreamshaper/diffusion_pytorch_model.bin", "cpu")
    )
    pipe = pipe.to(device)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


def load_dreamshaper_ours_pipeline(device, merge_weight, npo_lora_path=None):

    # load pipeline
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
    )

    pipe.unet.load_state_dict(
        torch.load("dreamshaper/diffusion_pytorch_model.bin", "cpu")
    )
    pipe = pipe.to(device)

    pipe_tmp = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe_tmp = pipe_tmp.to(device)

    pipe_tmp.unet = mix_models(pipe_tmp.unet, pipe.unet, merge_weight)

    pipe_tmp.load_lora_weights(npo_lora_path)
    pipe_tmp.fuse_lora()

    negative_unet = copy.deepcopy(pipe_tmp.unet)
    del pipe_tmp
    pipe.negative_unet = negative_unet

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


def load_origin_ours_pipeline(device, npo_lora_path=None):
    # load pipeline
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
    )

    pipe = pipe.to(device)

    pipe_tmp = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe_tmp = pipe_tmp.to(device)

    pipe_tmp.load_lora_weights(npo_lora_path)
    pipe_tmp.fuse_lora()
    negative_unet = copy.deepcopy(pipe_tmp.unet)
    del pipe_tmp
    pipe.negative_unet = negative_unet

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


def load_spo_pipeline(device):
    # load pipeline
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
    )
    pipe.load_lora_weights(
        "SPO-SD-v1-5_4k-p_10ep_LoRA/spo-sd-v1-5_4k-p_10ep_lora_diffusers.safetensors"
    )
    pipe = pipe.to(device)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


def load_spo_ours_pipeline(device, npo_lora_path):
    # load pipeline
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
    )
    pipe.load_lora_weights(
        "SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep_LoRA",
        "cpu",
    )
    pipe = pipe.to(device)

    pipe_tmp = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe_tmp = pipe_tmp.to(device)
    bad_weight = load_file(npo_lora_path, "cpu")
    pipe_tmp.load_lora_weights(bad_weight, adapter_name="bad")
    spo_weight = load_file(
        "SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep_LoRA",
        "cpu",
    )
    pipe_tmp.load_lora_weights(spo_weight, adapter_name="spo")
    pipe_tmp.set_adapters(["bad", "spo"], adapter_weights=[1.0, 1.0])
    negative_unet = copy.deepcopy(pipe_tmp.unet)

    del pipe_tmp

    pipe.negative_unet = negative_unet

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


def generate_batch_images(
    prompts,
    batch_size,
    resolution,
    pipeline,
    cfg,
    num_inference_steps,
    device,
    device_id,
    weight_dtype,
    seed,
    generation_path,
):

    total_batches = len(prompts) // batch_size + (
        1 if len(prompts) % batch_size != 0 else 0
    )
    for batch_idx in tqdm(range(total_batches)):
        batch_prompts = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        generator = torch.Generator(device=device).manual_seed(seed + batch_idx)

        with torch.autocast("cuda", weight_dtype):
            outputs = pipeline(
                prompt=batch_prompts,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=cfg,
                height=resolution[0],
                width=resolution[1],
            )
            images = outputs.images
        for img_idx, (img, prompt) in enumerate(zip(images, batch_prompts)):
            img_path = os.path.join(
                generation_path,
                f"{device_id}_{batch_idx * batch_size + img_idx:08d}.png",
            )
            img.save(img_path)
            text_path = os.path.join(
                generation_path,
                f"{device_id}_{batch_idx * batch_size + img_idx:08d}.txt",
            )
            with open(text_path, "w") as f:
                f.write(prompt)


def generate_imgs(
    generation_path,
    prompts,
    resolution,
    pipeline,
    cfg,
    num_inference_steps,
    device_id,
    weight_dtype,
    seed,
):

    torch.cuda.set_device(f"cuda:{device_id%num_device}")
    device = torch.device(f"cuda:{device_id%num_device}")

    num_prompts_per_device = len(prompts) // num_processes
    start_idx = device_id * num_prompts_per_device
    end_idx = (
        start_idx + num_prompts_per_device
        if device_id != (num_processes - 1)
        else len(prompts)
    )

    device_prompts = prompts[start_idx:end_idx]

    print(f"Device {device} generating for prompts {start_idx} to {end_idx-1}")

    print("## Prepare generation dataset")
    if isinstance(resolution, int):
        resolution = [resolution, resolution]

    batch_size = 32
    generate_batch_images(
        device_prompts,
        batch_size,
        resolution,
        pipeline,
        cfg,
        num_inference_steps,
        device,
        device_id,
        weight_dtype,
        seed,
        generation_path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_path", default="train_coco")
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_false"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cfg", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=25, type=int)
    parser.add_argument("--npo_lora_path", default=None, type=str)
    parser.add_argument("--merge_weight", default=0.0, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.generation_path, exist_ok=True)

    dataset = load_dataset("yuvalkirstain/pickapic_v1", split="test_unique")
    prompts = dataset["caption"]

    pipelines = []
    for i in range(num_processes):
        if "dpo+npo/" in args.generation_path:
            pipelines.append(
                load_dpo_ours_pipeline(
                    f"cuda:{i%num_device}", args.merge_weight, args.npo_lora_path
                )
            )
        elif "dpo+origin" in args.generation_path:
            pipelines.append(load_dpo_origin_pipeline(f"cuda:{i%num_device}"))
        elif "dpo/" in args.generation_path:
            pipelines.append(load_dpo_pipeline(f"cuda:{i%num_device}"))
        elif "dpo_lora/" in args.generation_path:
            pipelines.append(
                load_dpo_lora_pipeline(f"cuda:{i%num_device}", args.npo_lora_path)
            )
        elif "spo/" in args.generation_path:
            pipelines.append(load_spo_pipeline(f"cuda:{i%num_device}"))
        elif "spo+npo/" in args.generation_path:
            pipelines.append(
                load_spo_ours_pipeline(f"cuda:{i%num_device}", args.npo_lora_path)
            )
        elif "origin/" in args.generation_path:
            pipelines.append(load_origin_pipeline(f"cuda:{i%num_device}"))
        elif "origin+npo/" in args.generation_path:
            pipelines.append(
                load_origin_ours_pipeline(f"cuda:{i%num_device}", args.npo_lora_path)
            )
        elif "dreamshaper/" in args.generation_path:
            pipelines.append(load_dreamshaper_pipeline(f"cuda:{i%num_device}"))
        elif "dreamshaper+npo/" in args.generation_path:
            pipelines.append(
                load_dreamshaper_ours_pipeline(
                    f"cuda:{i%num_device}", args.npo_lora_path
                )
            )

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                generate_imgs,
                args.generation_path,
                prompts,
                args.resolution,
                pipelines[device_id],
                args.cfg,
                args.num_inference_steps,
                device_id,
                torch.float16,
                args.seed,
            )
            for device_id in range(num_processes)
        ]

        for future in as_completed(futures):
            print(f"Task completed: {future.result()}")
