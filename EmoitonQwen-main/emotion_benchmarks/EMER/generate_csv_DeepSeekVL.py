import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import sys
from PIL import Image


from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

def build_content(image_path, text_subtitles, version="CN"):
    if version == "CN":
        unit =(
                    "你是一个视频情绪推理的专家。"
                    "请你仔细对视频进行推理，结合人物的面部表情、肢体动作、背景信息、对话内容（如果不为空）来解释推理当前视频表达的情绪。"
                    "\n当前视频的对话内容是:[{}],"
                    "推理要有依据，内容严谨。使用下面的开头：在视频中...."
                ).format(text_subtitles)
            
    elif version == "EN":
        unit =("You are an expert in video emotion reasoning."
                        "Please reason the video carefully, and explain the reason why the current video expresses this emotional label in combination with the facial expressions, body movements, background information and conversation content of the characters."
                        "\nThe conversation content of the current video is:[{}],"
                        "Reasoning should be based on rigorous content. Start with the following: In the video...."
                ).format(text_subtitles)
            
    return unit

def perform_inference(model, image_processor, tokenizer, text_subtitles, image_path, version):

    prompt = build_content(image_path, text_subtitles, version)
    message=[
        {
            "role": "User",
            "content": f"<image_placeholder>{prompt}",
            "images": [f"{image_path}"]
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]
    # load images and prepare for inputs
    pil_images = load_pil_images(message)
    prepare_inputs = image_processor(
        conversations=message,
        images=pil_images,
        force_batchify=True
    ).to(model.device)

    # run image encoder to get the image embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
        do_sample=False,
        use_cache=True
    )

    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    cleaned_output = response.replace('\n', ' ').replace('\r', '').strip()
    escaped_output = cleaned_output.replace('"', '""')

    quoted_output = f'"{escaped_output}"'
    return quoted_output

def evaluate_model(model, image_processor, tokenizer, csv_file, output_csv, video_folder, model_inference_func, version="CN", temp_dir=""):
    df = pd.read_csv(csv_file)

    if not os.path.exists(output_csv):
        with open(output_csv, 'w') as f:
            if version == "CN":
                f.write('names,chi_reasons\n')
            elif version == "EN":
                f.write('names,eng_reasons\n')

    with tqdm(total=len(df), desc="Processing EMER videos") as pbar:
        for index, row in df.iterrows():
            name = row['names']
            text_subtitles = row['subtitles']
            video_name = f"{name}.mp4"
            video_path = os.path.join(video_folder, video_name)

            if not os.path.exists(video_path):
                print(f"Warning: Video {video_name} does not exist.")
                continue
            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read()  
            if not success:
                print(f"Failed to read video {video_name}.")
                continue

            image_path = os.path.join(temp_dir, "temp_DeepSeekVL_EMER.jpg")
            cv2.imwrite(image_path, frame)

            response_text = model_inference_func(model, image_processor, tokenizer, text_subtitles, image_path, version)
            with open(output_csv, "a") as f:
                f.write(f"{name},{response_text}\n")
            
            pbar.update(1)

    print(f"Results saved to {output_csv}")



if __name__ == "__main__":
    # if your model support Chinese, you can set version = "CN", otherwise set version = "EN"
    version = "CN"
    # change the paths according to your environment
    csv_file = "/home/dell/EmotionQwen-main/eval_data/EMER/dataset-v1/gt-chi.csv"
    video_folder = "/home/dell/EmotionQwen-main/eval_data/EMER/videos"

    model_dir = "/home/dell/models/deepseek-vl-7b-base"
    output_csv = "/home/dell/EmotionQwen-main/emotion_benchmarks/EMER/results/DeepSeekVL.csv"
    
    vl_chat_processor: VLChatProcessor  = VLChatProcessor.from_pretrained(model_dir)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    evaluate_model(vl_gpt, vl_chat_processor, tokenizer, csv_file, output_csv, video_folder, perform_inference, version, temp_dir)