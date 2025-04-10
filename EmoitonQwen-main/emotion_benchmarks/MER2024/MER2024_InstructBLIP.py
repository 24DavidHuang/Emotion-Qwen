import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

emotion_list = ['happy', 'sad', 'neutral', 'angry', 'surprise', 'worried']

prompt = f"Analyze the following video, combined with subtitles and visual clues, determine the emotional categories of the characters. \
Please choose one of the following words that best matches the emotions of the characters in the video as your answer: \
['happy', 'sad', 'neutral', 'angry', 'surprise', 'worried']. Reply with a single word indicating the emotion category."


def perform_inference(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    formatted_prompt = f"{prompt}\nAnswer:"
    inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=64,
        min_length=1,
    )
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return output_text

def evaluate_model(model, processor, csv_file, video_folder, model_inference_func, emotion_list, temp_dir):
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

            image_path = os.path.join(temp_dir, "temp_InstructBLIP_MER.jpg")
            cv2.imwrite(image_path, frame)
            
            response_text = model_inference_func(model, processor, image_path)

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
    Track = "NOISE"

    
    if Track =="NOISE":
        csv_file = '/home/dell/EmotionQwen-main/eval_data/MER2024/reference-noise.csv' 
        video_folder = '/home/dell/EmotionQwen-main/eval_data/MER2024/MER2024_noise'  
    elif Track == "SEMI":
        csv_file = '/home/dell/EmotionQwen-main/eval_data/MER2024/reference-semi.csv'  
        video_folder = '/home/dell/EmotionQwen-main/eval_data/MER2024/MER2024_semi'  
    else :
        print("Track must be NOISE or SEMI!")


    model_dir = "/home/dell/models/instructblip-vicuna-13b"
    print("TRACK: ",Track,"\n", "model dir:",model_dir)

    model = InstructBlipForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model.to(device)

    processor = InstructBlipProcessor.from_pretrained(model_dir)

    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    F1_score, Accuracy= evaluate_model(model, processor, csv_file, video_folder, perform_inference, emotion_list, temp_dir)
    print(f"F1 Score: {F1_score}, Accuracy: {Accuracy}")