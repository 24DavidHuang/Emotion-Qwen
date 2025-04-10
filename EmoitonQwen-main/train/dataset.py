import copy
import json
import logging
import math
import os
import re
import random
from dataclasses import dataclass, field
from typing import List, Union , Dict , Optional

import numpy as np
import torch
from PIL import Image
# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
import logging
import cv2

from vision_process import process_vision_info

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
ORIGIN_IMAGE_TOKEN = "<image>"
ORIGIN_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        transform,
        tokenizer,
        processor,
        batch_vision=False,
        face_detection=False,
        max_length=4096,
        image_folder = "",
        image_max_pixels=None,
        vidoe_max_pixels=None,
    ):
        super(SupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.processor = processor
        self.transform = transform
        self.batch_vision = batch_vision
        self.face_detection = face_detection
        self.max_length = max_length
        self.image_folder = image_folder
        self.image_max_pixels=image_max_pixels
        self.video_max_pixels=vidoe_max_pixels
        self.fps=2.0

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        cv2.setNumThreads(0)

        if_video = False
            
        if "image" in self.raw_data[i]:

            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            if isinstance(self.raw_data[i]["image"], str):
                image_path = os.path.join(self.image_folder, self.raw_data[i]["image"])
            elif isinstance(self.raw_data[i]["image"], Dict):
                image_path = {img_name : img_path for img_name, img_path in self.raw_data[i]["image"].items()}
            else:
                print(self.raw_data[i]["image"])
                raise ValueError(" 'image':imagepath type error")
            
            images = []
            if isinstance(image_path, str):
                if os.path.exists(image_path):
                    images.append(get_image_info(image_path, self.image_max_pixels))
                else:
                    print("No image in path:", image_path)
            elif isinstance(image_path, Dict):
                for img_name, img_path in image_path.items():
                    images.append(get_image_info(img_path, self.image_max_pixels))
        
        elif "video" in self.raw_data[i]:
            if_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            if isinstance(self.raw_data[i]["video"], str):
                video_path = os.path.join(self.image_folder, self.raw_data[i]["video"])
            elif isinstance(self.raw_data[i]["video"], Dict):
                video_path = {video_name : video_path for video_name, video_path in self.raw_data[i]["video"].items()}
            else:
                print(self.raw_data[i]["video"])
                raise ValueError(" 'video':video path type error")
            
            videos = []
            if isinstance(video_path, str):
                if os.path.exists(video_path):
                    video_input, video_kwargs = (get_video_info(video_path, self.video_max_pixels, self.fps)) # video_input.shape torch.Size([fps, channels=3, hight, width])
                    videos.append(video_input)
            elif isinstance(video_path, Dict):
                for video_name, video_path in video_path.items():
                    video_input, video_kwargs = get_video_info(video_path, self.video_max_pixels, self.fps)
                    videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
        
        sources = copy.deepcopy(replace_image_tokens(self.raw_data[i]["conversations"], if_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}\n{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = self.processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = self.processor(
                    text=[user_input],
                    images=images, 
                    videos=videos,
                    face_detection = self.face_detection,
                    padding=False, 
                    return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                inputs = self.processor(
                    text=[user_input], 
                    images=images, 
                    videos=videos, 
                    face_detection = self.face_detection,
                    padding=False, 
                    return_tensors='pt', 
                    **video_kwargs)
                all_second_gird.extend(inputs["second_per_grid_ts"])

                # else:
                #     inputs = self.processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')

                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = self.processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = self.processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)
        input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id=None)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict

        

def replace_image_tokens(conversations, if_video=False):
    if if_video:
        pattern = r'\n?' + re.escape(ORIGIN_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(ORIGIN_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    transformed_data = []
    for conversation in conversations:
        transformed_content = re.sub(pattern, replacement, conversation["content"])
        transformed_entry = {
            "role": conversation["role"],
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)
    
    return transformed_data
        

def get_image_info(image_path, max_pixel):
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "max_pixel": max_pixel
            }
            ]
        }
    ]
    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, max_pixels, fps):
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs

def truncate_sequence(input_ids, labels, max_length, eos_token_id=None):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

        
def data_collator(examples, pad_token_id=0): # multi image, set max_length = 8192
    batch_input_ids = []
    batch_label_ids = []
    batch_pixel_values = []
    batch_pixel_video_values = []
    batch_video_thw = []
    batch_image_thw = []
    batch_second_per_grid_ts = []
    
    for example in examples:
        keys = example.keys()
        if "pixel_values_videos" in keys:
            batch_pixel_video_values.append(example["pixel_values_videos"])
            batch_video_thw.append(example["video_grid_thw"])
        elif "pixel_values" in keys:
            batch_pixel_values.append(example["pixel_values"])
            batch_image_thw.append(example["image_grid_thw"])
        
        batch_input_ids.append(example["input_ids"])
        batch_label_ids.append(example["labels"])

        if "second_per_grid_ts" in keys:
            batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
    
    input_ids = pad_sequence(
        batch_input_ids, padding_side='right', padding_value=pad_token_id
    )

    attention_mask = input_ids != pad_token_id
    labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

    data_dict = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }

    if len(batch_pixel_values) > 0:
        pixel_values = torch.cat(batch_pixel_values, dim=0)
        image_thw = torch.cat(batch_image_thw, dim=0)
        data_dict["pixel_values"] = pixel_values
        data_dict["image_grid_thw"] = image_thw

    if len(batch_pixel_video_values) > 0:
        pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
        video_thw = torch.cat(batch_video_thw, dim=0)
        data_dict["pixel_values_videos"] = pixel_video_values
        data_dict["video_grid_thw"] = video_thw

    if len(batch_second_per_grid_ts) > 0:
        data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

    return data_dict
