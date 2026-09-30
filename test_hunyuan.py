"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- video-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- image-to-video: THUDM/CogVideoX-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --generate_type "t2v"
```

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""

import argparse
from typing import Literal

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline

from diffusers.utils import export_to_video, load_image, load_video
from receler.erasers.hunyuan_erasers import inject_eraser
from receler.erasers.utils import DisableEraser

import os
import json


def generate_video(
    prompt: str,
    model_path: str,
    eraser_path: str = None,
    lora_path: str = None,
    lora_weight: float = 1.0,
    eraser_rank: int = 128,
    output_path: str = "./output",
    image_or_video_path: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 6.0,
    num_frames: int = 49,
    height: int = 720,
    width: int = 1280,
    fps: int = 15,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    generate_clean: bool = True,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - eraser_path (str): The path of the legacy eraser weights to be used.
    - lora_path (str): The path of the PEFT/diffusers LoRA weights to be used.
    - eraser_rank (int): The rank of the eraser weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_frames (int): Number of generated frames.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = HunyuanVideoPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=torch.float16)
    pipe.vae.enable_tiling()

    

    if eraser_path and lora_path:
        raise ValueError("Use either --eraser_path or --lora_path, not both.")

    # Legacy T2VUnlearning eraser adapter path.
    if eraser_path:
        eraser_ckpt_path = os.path.join(eraser_path, f'eraser_weights.pt')
        eraser_config_path = os.path.join(eraser_path, f'eraser_config.json')
        with open(eraser_config_path) as f:
            eraser_config = json.load(f)
        # # inject erasers into pretrained cogvideo
        # if num_frames > 30:
        #     inject_eraser(pipe.transformer, eraser_ckpt=torch.load(eraser_ckpt_path, map_location='cpu'), eraser_rank=eraser_rank)#,eraser_weight=0.7)
        # else:
        inject_eraser(pipe.transformer, eraser_ckpt=torch.load(eraser_ckpt_path, map_location='cpu'), eraser_rank=eraser_rank)

    # New PEFT/diffusers LoRA path used by train_hunyuan_erase_lora.py.
    if lora_path:
        pipe.load_lora_weights(lora_path, adapter_name="erase_lora")
        pipe.set_adapters(["erase_lora"], adapter_weights=[lora_weight])
        print(f"Loaded PEFT/diffusers LoRA: {lora_path}")
        print(f"LoRA adapter weight: {lora_weight}")
        if hasattr(pipe, "get_active_adapters"):
            print(f"Active LoRA adapters: {pipe.get_active_adapters()}")
    

    pipe.to("cuda")

    # pipe.enable_sequential_cpu_offload()

    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.
    if generate_clean:
        if lora_path and hasattr(pipe, "disable_lora"):
            pipe.disable_lora()
        with DisableEraser(pipe.transformer):
                video_before = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
            ).frames[0]
        if lora_path and hasattr(pipe, "enable_lora"):
            pipe.enable_lora()
    video_erased = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
    ).frames[0]
    
    # 5. Export the generated frames to a video file.
    output_suffix = "_erase" if (eraser_path or lora_path) else "_base"
    if generate_clean:
        export_to_video(video_before, output_path + "_clean.mp4", fps=fps)
    export_to_video(video_erased, output_path + output_suffix + ".mp4", fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--eraser_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of PEFT/diffusers LoRA weights to be used")
    parser.add_argument("--lora_weight", type=float, default=1.0, help="PEFT/diffusers LoRA adapter weight")
    parser.add_argument("--eraser_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=30, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames to generate per prompt")
    parser.add_argument("--height", type=int, default=720, help="The height in pixels of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="The width in pixels of the generated video")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the exported video")
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--generate_clean", action="store_true", help="generate clean video for comparison")

    args = parser.parse_args()
    dtype = torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        eraser_path=args.eraser_path,
        lora_path=args.lora_path,
        lora_weight=args.lora_weight,
        eraser_rank=args.eraser_rank,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        generate_clean=args.generate_clean,
    )
