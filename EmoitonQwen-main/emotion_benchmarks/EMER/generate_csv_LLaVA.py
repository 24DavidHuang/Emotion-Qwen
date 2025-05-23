import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import sys
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from model_utils.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from model_utils.llava.conversation import conv_templates, SeparatorStyle

from model_utils.llava.model.builder import load_pretrained_model
from model_utils.llava.utils import disable_torch_init
from model_utils.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)

temperature = 0.2
top_p = None
num_beams = 1
max_new_tokens = 1024
conv_mode = "llava_v1"

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

    raw_image_data = Image.open(image_path)
    raw_image_data_list = [raw_image_data]

    if raw_image_data_list:
        # print("There is image here")
        raw_image_data_list = [img.convert("RGB") for img in raw_image_data_list]
        raw_image_sizes = [img.size for img in raw_image_data_list]
        images_tensor = process_images(
            raw_image_data_list,
            image_processor,
            model.config
        )[0]
        images_tensor = images_tensor.unsqueeze(0).half().cuda()
        prompt = build_content(image_path, text_subtitles, version)

        if getattr(model.config, 'mm_use_im_start_end', False):
            text_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            text_prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    else:
        # Handle the case where there are no images
        # print("There is no image here")
        raw_image_sizes = None
        images_tensor = None  

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], text_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=raw_image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs

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

            image_path = os.path.join(temp_dir, "temp_LLaVA_EMER.jpg")
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

    model_dir = "/home/dell/models/llava-v1.5-7b"
    output_csv = "/home/dell/EmotionQwen-main/emotion_benchmarks/EMER/results/LLaVA.csv"
    
    model_base = None
    init_model_name = get_model_name_from_path(model_dir)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_dir, model_base, init_model_name)

    temp_dir = "/home/dell/EmotionQwen-main/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    evaluate_model(model, image_processor, tokenizer, csv_file, output_csv, video_folder, perform_inference, version, temp_dir)