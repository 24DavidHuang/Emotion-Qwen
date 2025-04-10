from transformers import AutoModel, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_dir="path/to/EmotionQwen"
video_path = "path/to/your/videos"
prompt = "Analyze the following video, combined with subtitles and visual clues, determine the emotional categories of the characters."

model = AutoModel.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
)

# default processer
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

# if CUDA OOM, set max_pixels:

# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(model_dir, max_pixels=max_pixels, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": f"file://{video_path}",
                "max_pixels": 448 * 448, # if CUDA OOM, lower this
                "fps": 1.0,
            },
            {"type": "text", "text": f"{prompt}"},
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

generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)