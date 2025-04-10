import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModel, AutoProcessor
from qwen_vl_utils import process_vision_info

emotion_list = ['happy', 'sad', 'neutral', 'angry', 'surprise', 'worried']

prompt = f"Analyze the following video, combined with subtitles and visual clues, determine the emotional categories of the characters. \
Please choose one of the following words that best matches the emotions of the characters in the video as your answer: \
['happy', 'sad', 'neutral', 'angry', 'surprise', 'worried']. Reply with a single word indicating the emotion category."


def perform_inference(model, processor, video_path):
    video_path = f"file://{video_path}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 602112,
                    "fps": 3.0,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip("\n").strip()

def evaluate_model(model, processor, csv_file, video_folder, model_inference_func, emotion_list):
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
            
            response_text = model_inference_func(model, processor, video_path)
            
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
    model_dir="/path/to/your/model"   
    Track = "SEMI"

    print("TRACK: ",Track,"\n", "model dir:",model_dir)
    if Track =="NOISE":
        csv_file = '/home/dell/EmotionQwen-main/eval_data/MER2024/reference-noise.csv' 
        video_folder = '/home/dell/EmotionQwen-main/eval_data/MER2024/MER2024_noise'  
    elif Track == "SEMI":
        csv_file = '/home/dell/EmotionQwen-main/eval_data/MER2024/reference-semi.csv'  
        video_folder = '/home/dell/EmotionQwen-main/eval_data/MER2024/MER2024_semi'  
    else :
        print("Track must be NOISE or SEMI!")

    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    F1_score, Accuracy= evaluate_model(model, processor, csv_file, video_folder, perform_inference, emotion_list)
    print(f"F1 Score: {F1_score}, Accuracy: {Accuracy}")