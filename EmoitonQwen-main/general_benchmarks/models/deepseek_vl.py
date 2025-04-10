import os
import io
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


from .base_model import BaseModel
from .model_utils.qwen_vl_utils import process_vision_info

default_path = "deepseek-ai/deepseek-vl-7b-base"

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
        return None


class DeepSeekVL(BaseModel):
    def __init__(self, 
                 model_name: str = "deepseek_vl", 
                 model_path: str = default_path):
        super().__init__(model_name, model_path)
        self.vl_chat_processor: VLChatProcessor  = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
        self.temp_dir = ".log/temp"

    def generate(self, text_prompt: str, raw_image_data: str):
        image = get_pil_image_return_image_base64(raw_image_data)
        if image is not None:
            image_path = os.path.join(self.temp_dir, "temp_deepseek.jpg")
            image.save(image_path)

            message=[
                {
                    "role": "User",
                    "content": f"<image_placeholder>{text_prompt}",
                    "images": [f"{image_path}"]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
        else:
            message=[
                {
                    "role": "User",
                    "content": text_prompt,
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
        # load images and prepare for inputs
        pil_images = load_pil_images(message)
        prepare_inputs = self.vl_chat_processor(
            conversations=message,
            images=pil_images,
            force_batchify=True
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True
        )

        response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return response.strip(" ").strip("\n")

    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for QwenVL
        pass
