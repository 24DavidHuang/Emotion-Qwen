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

from sklearn.metrics import f1_score, accuracy_score


from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

emotion_list = ['happy', 'sad', 'neutral', 'angry', 'surprise', 'worried']
prompt = f"Analyze the following video, combined with subtitles and visual clues, determine the emotional categories of the characters. \
Please choose one of the following words that best matches the emotions of the characters in the video as your answer: \
['happy', 'sad', 'neutral', 'angry', 'surprise', 'worried']. Reply with a single word indicating the emotion category."


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
            video_name = f"{row['name']}.mp4"
            label = row['discrete']
            video_path = os.path.join(video_folder, video_name)

            if not os.path.exists(video_path):
                print(f"Warning: Video {video_name} does not exist.")
                continue

            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read() 
            if not success:
                print(f"Failed to read video {video_name}.")
                continue

            image_path = os.path.join(temp_dir, "temp_DeepSeekVL_MER.jpg")
            cv2.imwrite(image_path, frame)
            
            response_text = model_inference_func(model, image_processor, tokenizer, image_path)
            
            prediction = max((emotion.lower() in response_text.lower(), emotion) for emotion in emotion_list)[1]
            true_label = label.lower()
            print(f"Response_text: {response_text.lower()}, Predicted Label: {prediction}, True Label: {true_label}")

            true_labels.append(true_label)
            predicted_labels.append(prediction.lower())

            current_accuracy = accuracy_score(true_labels, predicted_labels)
            current_f1 = f1_score(true_labels, predicted_labels, average='weighted')

            pbar.set_postfix_str(f"Current Accuracy: {current_accuracy:.4f}, F1 Score: {current_f1:.4f}")
            pbar.update(1)

    print("\nEvaluation completed.")
    
    final_accuracy = accuracy_score(true_labels, predicted_labels)
    final_f1 = f1_score(true_labels, predicted_labels, average='weighted')  
    
    return final_f1, final_accuracy



if __name__ == "__main__":
    Track = "SEMI"

    if Track =="NOISE":
        csv_file = '/home/dell/EmotionQwen-main/eval_data/MER2024/reference-noise.csv' 
        video_folder = '/home/dell/EmotionQwen-main/eval_data/MER2024/MER2024_noise'  
    elif Track == "SEMI":
        csv_file = '/home/dell/EmotionQwen-main/eval_data/MER2024/reference-semi.csv'  
        video_folder = '/home/dell/EmotionQwen-main/eval_data/MER2024/MER2024_semi'  
    else :
        print("Track must be NOISE or SEMI!")


    model_dir = "/home/dell/models/deepseek-vl-7b-base"
    print("TRACK: ",Track,"\n", "model dir:",model_dir)

    vl_chat_processor: VLChatProcessor  = VLChatProcessor.from_pretrained(model_dir)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
    vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()

    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    F1_score, Accuracy = evaluate_model(vl_gpt, vl_chat_processor, tokenizer, csv_file, video_folder, perform_inference, emotion_list, temp_dir)
    print(f"F1 Score: {F1_score}, Accuracy: {Accuracy}")
