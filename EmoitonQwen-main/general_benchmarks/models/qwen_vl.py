import os

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from .base_model import BaseModel

import io
import base64
import requests
from PIL import Image
from io import BytesIO

default_path = "Qwen/Qwen-VL-Chat"

def get_pil_image_return_image_base64(raw_image_data):
    if raw_image_data is None:
        return None
    elif isinstance(raw_image_data, list):
        result = []
        for item in raw_image_data:
            processed_item = get_pil_image_return_image_base64(item)
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


class QwenVL(BaseModel):
    def __init__(self, model_name: str = "qwen_vl", model_path: str = default_path):
        super().__init__(model_name, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        self.temp_dir = ".log/temp"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def generate(self, text_prompt: str, raw_image_data: str):
        image = get_pil_image_return_image_base64(raw_image_data)
        
        image_path = os.path.join(self.temp_dir, "temp.jpg")
        image.save(image_path)
        
        query = []
        query.append({"image": image_path})
        query.append({"text": text_prompt})
        query = self.tokenizer.from_list_format(query)
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for QwenVL
        pass
