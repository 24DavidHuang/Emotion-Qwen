from .base_model import BaseModel

import torch
from transformers import AutoModel, AutoTokenizer

from PIL import Image
import io
import base64
import requests

import warnings
warnings.filterwarnings("ignore")

class MiniCPM_V(BaseModel):
    def __init__(
        self,
        model_path: str = "",
        model_base: str = None,
        model_name: str = "minicpm_v",
        max_new_tokens: int = 512
    ):
        super().__init__(model_name, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.max_new_tokens = max_new_tokens
        self.model = AutoModel.from_pretrained(model_path, 
                                  trust_remote_code=True,
                                #   use_safetensors = False,
                                  attn_implementation='sdpa', 
                                  torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        self.model = self.model.eval().cuda()


    def get_pil_image_return_list(self,raw_image_data) -> Image.Image:
        if raw_image_data==None:
            return None

        elif isinstance(raw_image_data, list):
            # If it's a list, assume it contains PIL.Image.Image objects or other supported types
            result = []
            for item in raw_image_data:
                try:
                    result.extend(self.get_pil_image_return_list(item))
                except ValueError as e:
                    print(f"Failed to process item in list: {e}")
            return result

        if isinstance(raw_image_data, Image.Image):
            return [raw_image_data]

        elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
            return [Image.open(io.BytesIO(raw_image_data["bytes"]))]
        
        elif isinstance(raw_image_data, str):
            # Check if the string is a URL
            if raw_image_data.startswith(('http://', 'https://')):
                try:
                    response = requests.get(raw_image_data, timeout=10)
                    response.raise_for_status()  # Raise an error for bad status codes
                    image_bytes = response.content
                    return [Image.open(io.BytesIO(image_bytes))]
                except (requests.RequestException, IOError) as e:
                    raise ValueError(f"Failed to load image from URL: {e}")
            else:  # Assuming this is a base64 encoded string
                image_bytes = base64.b64decode(raw_image_data)
                return [Image.open(io.BytesIO(image_bytes))]
        
        else:
            raise ValueError("Unsupported image data format")


    def generate(self, text_prompt: str, raw_image_data: str):
        image = self.get_pil_image_return_list(raw_image_data)
        msgs = [{'role': 'user', 'content': [image, text_prompt]}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            # **default_kwargs
        )

        return res.strip(" ").strip("\n")

    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for QwenVL
        pass