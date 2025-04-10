import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from transformers import AutoTokenizer
from qwen_vl_utils import process_vision_info


def build_content(video_path, text_subtitles, version="CN"):
    content=[]
    if video_path.lower().endswith('.mp4'):
        unit={
                "type": "video",
                "video": f"file://{video_path}"
            }
        content.append(unit)
    
    if version == "CN":
        content.append(
            {   
                "type": "text",
                "text": (
                    "你是一个视频情绪推理的专家。"
                    "请你仔细对视频进行推理，结合人物的面部表情、肢体动作、背景信息、对话内容（如果不为空）来解释推理当前视频表达的情绪。"
                    "\n当前视频的对话内容是:[{}],"
                    "推理要有依据，内容严谨。使用下面的开头：在视频中...."
                ).format(text_subtitles)
            }
        )
    elif version == "EN":
        content.append(
            {   
                "type": "text",
                "text": ("You are an expert in video emotion reasoning."
                        "Please reason the video carefully, and explain the reason why the current video expresses this emotional label in combination with the facial expressions, body movements, background information and conversation content of the characters."
                        "\nThe conversation content of the current video is:[{}],"
                        "Reasoning should be based on rigorous content. Start with the following: In the video...."
                ).format(text_subtitles)
            }
        )

    return content

def perform_inference(model, tokenizer, processor, text_subtitles, video_path):
    messages = [
        {
            "role": "user",
            "content": build_content(video_path, text_subtitles, version)
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
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    cleaned_output = output_text[0].replace('\n', ' ').replace('\r', '').strip()
    escaped_output = cleaned_output.replace('"', '""')

    quoted_output = f'"{escaped_output}"'

    return quoted_output

def evaluate_model(model, tokenizer, processor, csv_file, output_csv, video_folder, model_inference_func, version="CN"):
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

            response_text = model_inference_func(model, tokenizer, processor, text_subtitles, video_path)
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
    model_dir = "/home/dell/models/Qwen2-VL-7B"
    output_csv = "/home/dell/EmotionQwen-main/emotion_benchmarks/EMER/results/Qwen2-VL.csv"
    

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    processor = Qwen2VLProcessor.from_pretrained(model_dir, trust_remote_code=True)

    evaluate_model(model, tokenizer, processor, csv_file, output_csv, video_folder, perform_inference, version)