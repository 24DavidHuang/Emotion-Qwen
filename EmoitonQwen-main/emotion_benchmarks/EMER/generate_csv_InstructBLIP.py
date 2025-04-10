import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image

def build_content(image_path, text_subtitles, version="CN"):

    if version == "CN":
        unit = (
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

def perform_inference(model, processor, text_subtitles, image_path, version):
    image = Image.open(image_path).convert("RGB")
    prompt = build_content(image_path, text_subtitles, version)
    formatted_prompt = f"{prompt}\nAnswer:"
    inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=512,
        min_length=1,
    )
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    cleaned_output = output_text.replace('\n', ' ').replace('\r', '').strip()
    escaped_output = cleaned_output.replace('"', '""')

    quoted_output = f'"{escaped_output}"'
    return quoted_output

def evaluate_model(model, processor, csv_file, output_csv, video_folder, model_inference_func, version="CN", temp_dir=""):
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

            image_path = os.path.join(temp_dir, "temp_EMER.jpg")
            cv2.imwrite(image_path, frame)

            response_text = model_inference_func(model, processor, text_subtitles, image_path, version)
            with open(output_csv, "a") as f:
                f.write(f"{name},{response_text}\n")
            
            pbar.update(1)

    print(f"Results saved to {output_csv}")



if __name__ == "__main__":
    # if your model support Chinese, you can set version = "CN", otherwise set version = "EN"
    version = "EN"
    # change the paths according to your environment
    csv_file = "/home/dell/EmotionQwen-main/eval_data/EMER/dataset-v1/gt-chi.csv"
    video_folder = "/home/dell/EmotionQwen-main/eval_data/EMER/videos"

    model_dir = "/home/dell/models/instructblip-vicuna-13b"
    output_csv = "/home/dell/EmotionQwen-main/emotion_benchmarks/EMER/results/InstructBLIP.csv"
    
    model = InstructBlipForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    processor = InstructBlipProcessor.from_pretrained(model_dir)


    temp_dir = "/home/dell/EmotionQwen-main/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    evaluate_model(model, processor, csv_file, output_csv, video_folder, perform_inference, version, temp_dir)