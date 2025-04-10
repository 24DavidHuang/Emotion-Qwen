import re
import sys
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.io import read_video

import os
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image

from sklearn.metrics import recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from qwen_vl_utils import process_vision_info

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

emotion_list = ['happy', 'sad', 'neutral', 'angry', 'surprise', 'disgust', 'fear']

temperature = 0.2
top_p = None
num_beams = 1
max_new_tokens = 1024

def perform_inference(model, image_processor, tokenizer, context_len, conv_mode, image_path):
    prompt = f"Analyze the following video, combined with subtitles and visual clues, determine the emotional categories of the characters. \
                Please choose one of the following words that best matches the emotions of the characters in the video as your answer: \
                [happy, sad, neutral, angry, surprise, disgust, fear]. Please reply with a single word indicating the emotion category."
    
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

def evaluate_model(model, image_processor, tokenizer, context_len, conv_mode, csv_file, video_folder, model_inference_func, emotion_list, temp_dir):
    df = pd.read_csv(csv_file)
    
    true_labels = []
    predicted_labels = []

    with tqdm(total=len(df), desc="Processing videos") as pbar:
        for index, row in df.iterrows():
            video_name = f"{row['video_name']}.mp4"
            label_id = row['label']
            if label_id == 0:
                raise ValueError("Label ID should be greater than 0.")
            label = emotion_list[ label_id -1 ]
            video_path = os.path.join(video_folder, video_name)

            if not os.path.exists(video_path):
                print(f"Warning: Video {video_name} does not exist.")
                continue

            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read() 
            if not success:
                print(f"Failed to read video {video_name}.")
                continue

            image_path = os.path.join(temp_dir, "temp_LLaVA.jpg")
            cv2.imwrite(image_path, frame)
            
            response_text = model_inference_func(model, image_processor, tokenizer, context_len, conv_mode, image_path)
            
            prediction = 1 if label.lower() in response_text.lower() else 0
            
            true_labels.append(1) 
            predicted_labels.append(prediction)

            current_accuracy = np.mean(predicted_labels)
            
            pbar.set_postfix_str(f"Current Accuracy: {current_accuracy:.4f}")
            pbar.update(1)

    print("\nEvaluation completed.")
    
    war = recall_score(true_labels, predicted_labels, average='binary') 
    uar = recall_score(true_labels, predicted_labels, average='macro')
    
    return war, uar


if __name__ == "__main__":
    csv_file = '/home/dell/EmotionQwen-main/eval_data/DFEW/test(single-labeled)/set_1.csv'  
    video_folder = '/home/dell/EmotionQwen-main/eval_data/DFEW/videos' 


    model_dir = "/home/dell/models/llava-v1.5-7b"
    model_base = None
    conv_mode = "llava_v1"


    init_model_name = get_model_name_from_path(model_dir)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_dir, model_base, init_model_name)

    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    war, uar = evaluate_model(model, image_processor, tokenizer, context_len, conv_mode, csv_file, video_folder, perform_inference, emotion_list, temp_dir)
    print(f"WAR: {war}, UAR: {uar}")

# conda activate llava