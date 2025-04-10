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


from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

emotion_list = ['happy', 'sad', 'neutral', 'angry', 'surprise', 'disgust', 'fear']
prompt = f"Analyze the following video, combined with subtitles and visual clues, determine the emotional categories of the characters. \
            Please choose one of the following words that best matches the emotions of the characters in the video as your answer: \
            [happy, sad, neutral, angry, surprise, disgust, fear]. Please reply with a single word indicating the emotion category."


def perform_inference(model, image_processor, tokenizer, image_path):
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
        max_new_tokens=64,
        do_sample=False,
        use_cache=True
    )

    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return response.strip(" ").strip("\n")



def evaluate_model(model, image_processor, tokenizer, csv_file, video_folder, model_inference_func, emotion_list, temp_dir):
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

            image_path = os.path.join(temp_dir, "temp_DeepSeekVL_DEFW.jpg")
            cv2.imwrite(image_path, frame)
            
            response_text = model_inference_func(model, image_processor, tokenizer, image_path)
            
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


    model_dir = "/home/dell/models/deepseek-vl-7b-base"

    vl_chat_processor: VLChatProcessor  = VLChatProcessor.from_pretrained(model_dir)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    war, uar = evaluate_model(vl_gpt, vl_chat_processor, tokenizer, csv_file, video_folder, perform_inference, emotion_list, temp_dir)
    print(f"WAR: {war}, UAR: {uar}")
