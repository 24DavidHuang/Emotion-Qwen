import re
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.io import read_video

from .base_model import BaseModel
from .model_utils.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from .model_utils.llava.conversation import conv_templates, SeparatorStyle

from .model_utils.llava.model.builder import load_pretrained_model
from .model_utils.llava.utils import disable_torch_init
from .model_utils.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)

from PIL import Image
import io
import base64
import requests

import warnings
warnings.filterwarnings("ignore")

def get_pil_image_return_list(raw_image_data) -> Image.Image:
    # print(f"Type of raw_image_data: {type(raw_image_data)}")
    # print(f"raw_image_data: {raw_image_data}")

    if raw_image_data==None:
        return None

    elif isinstance(raw_image_data, list):
        # If it's a list, assume it contains PIL.Image.Image objects or other supported types
        result = []
        for item in raw_image_data:
            try:
                result.extend(get_pil_image_return_list(item))
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

class LLaVA_Model(BaseModel):
    def __init__(
        self,
        model_path: str = "",
        model_base: str = None,
        model_name: str = "llava-v1.5-7b",
        conv_mode: str = "llava_v1",
        temperature: float = 0.2,
        top_p: float = None,
        num_beams: int = 1,
        max_new_tokens: int = 1024
    ):
        super().__init__(model_name, model_path)
        init_model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, init_model_name)
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

    def generate(self, text_prompt: str, raw_image_data: str):
        
        raw_image_data_list = get_pil_image_return_list(raw_image_data)

        if raw_image_data_list:
            # print("There is image here")
            raw_image_data_list = [img.convert("RGB") for img in raw_image_data_list]
            raw_image_sizes = [img.size for img in raw_image_data_list]
            images_tensor = process_images(
                raw_image_data_list,
                self.image_processor,
                self.model.config
            )[0]
            images_tensor = images_tensor.unsqueeze(0).half().cuda()

            if getattr(self.model.config, 'mm_use_im_start_end', False):
                text_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text_prompt
            else:
                text_prompt = DEFAULT_IMAGE_TOKEN + '\n' + text_prompt

        else:
            # Handle the case where there are no images
            # print("There is no image here")
            raw_image_sizes = None
            images_tensor = None  

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=raw_image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs

    def eval_forward(self, text_prompt: str, raw_image_data: str):
        pass
