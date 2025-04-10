import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

def build_content(image_path, emotion, text_subtitles, version="CN"):

    if version == "CN":
        unit ={   
                "text": (
                    "你是一个视频情绪推理的专家。"
                    "请你仔细对视频进行推理，结合人物的面部表情、肢体动作、背景信息、对话内容（如果不为空）来解释推理当前视频表达的情绪。"
                    "\n当前视频的对话内容是:[{}],"
                    "推理要有依据，内容严谨。使用下面的开头：在视频中...."
                ).format(text_subtitles)
            }
    elif version == "EN":
        unit ={
                "text": ("You are an expert in video emotion reasoning."
                        "Please reason the video carefully, and explain the reason why the current video expresses this emotional label in combination with the facial expressions, body movements, background information and conversation content of the characters."
                        "\nThe conversation content of the current video is:[{}],"
                        "Reasoning should be based on rigorous content. Start with the following: In the video...."
                ).format(text_subtitles)
            }

    return unit

def perform_inference(model, tokenizer, emotion, text_subtitles, image_path, version):
    messages = []
    messages.append({"image": image_path})
    messages.append(build_content(image_path, emotion, text_subtitles, version))

    # Preparation for inference
    messages = tokenizer.from_list_format(messages)
    output_text, history = model.chat(tokenizer, query=messages, history=None)

    cleaned_output = output_text.replace('\n', ' ').replace('\r', '').strip()
    escaped_output = cleaned_output.replace('"', '""')

    quoted_output = f'"{escaped_output}"'

    return quoted_output

def evaluate_model(model, tokenizer, csv_file, output_csv, video_folder, model_inference_func, version="CN", temp_dir=""):
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
            emotion = row['emotions']
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

            response_text = model_inference_func(model, tokenizer, emotion, text_subtitles, image_path, version)
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
    model_dir = "/home/dell/models/Qwen-VL-Chat"
    output_csv = "/home/dell/EmotionQwen-main/emotion_benchmarks/EMER/results/Qwen-VL.csv"
    

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:1", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)

    temp_dir = "/home/dell/EmotionQwen-main/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    evaluate_model(model, tokenizer, csv_file, output_csv, video_folder, perform_inference, version, temp_dir)