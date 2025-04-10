
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from qwen_vl_utils import process_vision_info

emotion_list = ['happy', 'sad', 'neutral', 'angry', 'surprise', 'worried']

prompt = f"Analyze the following video, combined with subtitles and visual clues, determine the emotional categories of the characters. \
Please choose one of the following words that best matches the emotions of the characters in the video as your answer: \
['happy', 'sad', 'neutral', 'angry', 'surprise', 'worried']. Reply with a single word indicating the emotion category."


def perform_inference(model, tokenizer, image_path):
    image_path = f"{image_path}"
    messages = []
    messages.append({"image": image_path})
    messages.append({"text": prompt})
    messages = tokenizer.from_list_format(messages)
    # Preparation for inference
    output_text, history = model.chat(tokenizer, query=messages, history=None)
    return output_text

def evaluate_model(model, tokenizer, csv_file, video_folder, model_inference_func, emotion_list, temp_dir):
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

            image_path = os.path.join(temp_dir, "temp.jpg")
            cv2.imwrite(image_path, frame)
            
            response_text = model_inference_func(model, tokenizer, image_path)
            
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
    model_dir = "/home/dell/models/Qwen-VL-Chat"
    Track = "NOISE"

    print("TRACK: ",Track,"\n", "model dir:",model_dir)
    if Track =="NOISE":
        csv_file = '/home/dell/EmotionQwen-main/eval_data/MER2024/reference-noise.csv' 
        video_folder = '/home/dell/EmotionQwen-main/eval_data/MER2024/MER2024_noise'  
    elif Track == "SEMI":
        csv_file = '/home/dell/EmotionQwen-main/eval_data/MER2024/reference-semi.csv'  
        video_folder = '/home/dell/EmotionQwen-main/eval_data/MER2024/MER2024_semi'  
    else :
        print("Track must be NOISE or SEMI!")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", bf16=True, trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)


    F1_score, Accuracy= evaluate_model(model, tokenizer, csv_file, video_folder, perform_inference, emotion_list, temp_dir)
    print(f"F1 Score: {F1_score}, Accuracy: {Accuracy}")