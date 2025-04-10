import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import recall_score
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from transformers import AutoTokenizer
from qwen_vl_utils import process_vision_info

emotion_list = ['happy', 'sad', 'neutral', 'angry', 'surprise', 'disgust', 'fear']

prompt = f"Analyze the following video, combined with subtitles and visual clues, determine the emotional categories of the characters. \
Please choose one of the following words that best matches the emotions of the characters in the video as your answer: \
[happy, sad, neutral, angry, surprise, disgust, fear]. Please reply with a single word indicating the emotion category."


def perform_inference(model, tokenizer, processor, video_path):
    video_path = f"file://{video_path}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 602112,
                    "fps": 2.0,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = tokenizer.apply_chat_template(
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

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip("\n").strip()

def evaluate_model(model, tokenizer, processor, csv_file, video_folder, model_inference_func, emotion_list):
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
            
            response_text = model_inference_func(model, tokenizer, processor, video_path)
            
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


    model_dir = "/home/dell/models/Qwen2-VL-7B"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    processor = Qwen2VLProcessor.from_pretrained(model_dir, trust_remote_code=True)

    war, uar = evaluate_model(model, tokenizer, processor, csv_file, video_folder, perform_inference, emotion_list)
    print(f"WAR: {war}, UAR: {uar}")