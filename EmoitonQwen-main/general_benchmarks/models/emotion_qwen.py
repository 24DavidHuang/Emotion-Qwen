import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from transformers import AutoModel, AutoTokenizer, AutoProcessor
# pip install --no-cache-dir git+https://github.com/huggingface/transformers@19e6e80e10118f855137b90740936c0b11ac397f -i https://pypi.tuna.tsinghua.edu.cn/simple

from .base_model import BaseModel
from .model_utils.qwen_vl_utils import process_vision_info

default_path = ""


class EmotionQwen(BaseModel):
    def __init__(self, 
                 model_name: str = "EmotionQwen", 
                 model_path: str = default_path):
        super().__init__(model_name, model_path)
        self.processor  = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, 
                                               torch_dtype=torch.bfloat16, 
                                               device_map="auto",
                                               trust_remote_code=True,
                                               attn_implementation="flash_attention_2"
                                            )

    def get_pil_image_return_image_base64(self, raw_image_data):
        if raw_image_data is None:
            print("raw_image_data is None")
            return None
        elif isinstance(raw_image_data, list):
            result = []
            for item in raw_image_data:
                processed_item = self.get_pil_image_return_image_base64(item)
                if isinstance(processed_item, list):
                    result.extend(processed_item)  
                else:
                    result.append(processed_item) 
            return result
        elif isinstance(raw_image_data, Image.Image):
            return raw_image_data
        elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
            return Image.open(io.BytesIO(raw_image_data["bytes"]))
        elif isinstance(raw_image_data, str):
            if raw_image_data.startswith(('http://', 'https://')):
                try:
                    response = requests.get(raw_image_data, timeout=10)
                    response.raise_for_status()
                    image_bytes = response.content
                    return Image.open(io.BytesIO(image_bytes))
                except (requests.RequestException, IOError) as e:
                    raise ValueError(f"Failed to load image from URL: {e}")
            else:
                image_bytes = base64.b64decode(raw_image_data)
                return Image.open(io.BytesIO(image_bytes))
        else:
            raise ValueError("Unsupported image data format")

    def generate(self, text_prompt: str, raw_image_data: str):
        image = self.get_pil_image_return_image_base64(raw_image_data)
        if image is not None:
            if isinstance(image, list):
                image=image[0]
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_base64 = f"data:image;base64,{img_str}"
        else:
            img_base64 = None

        message=[
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image", 
                        "image": img_base64
                    }, 
                    {
                        "type": "text", 
                        "text": text_prompt
                    },
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        if image is not None:
            image_inputs, video_inputs = process_vision_info(message)
        else:
            image_inputs, video_inputs = None, None
            
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt")
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return response[0].strip(" ").strip("\n")

    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for QwenVL
        pass
