# Training Emotion-Qwen



## Table of Contents

- [Training Emotion-Qwen](#Training-Emotion-Qwen)
  - [Supported Method](#Supported-Method)
  - [Installation](#installation)
    - [Using `environment.yaml`](#using-environmentyaml)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
    - [Full Finetuning](#full-finetuning)
    - [Finetune with LoRA](#finetune-with-lora)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Method

- Deepspeed
- LoRA
- Full-finetuning
- Disable/enable Flash Attention 2
- Multi-image and video training

## Installation

Install the required packages using `environment.yaml`.

### Using `environment.yaml`

```bash
cd ./EmotionQwen-main
conda env create -f environment.yaml
conda activate emotion_qwen

pip install flash-attn --no-build-isolation
```

**Note:** You should install flash-attn after installing the other packages.

## Dataset Preparation

The dataset should be a JSON file where each entry contains information about conversations and images or videos. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, the image file names should have been in a list; the video tokens should all be `<video>`.**<br><br>
**Please see the example below and follow format your data.**

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
        "id": "000000033471",
        "image": "coco/train2017/000000033471.jpg",
        "conversations": [
            {
                "role": "user",
                "content": "<image>\nWhat are the colors of the bus in the image?"
            },
            {
                "role": "assistant",
                "content": "The bus in the image is white and red."
            },
            {
                "role": "user",
                "content": "What feature can be seen on the back of the bus?"
            },
            {
                "role": "assistant",
                "content": "The back of the bus features an advertisement."
            },
            {
                "role": "user",
                "content": "Is the bus driving down the street or pulled off to the side?"
            },
            {
                "role": "assistant",
                "content": "The bus is driving down the street, which is crowded with people and other vehicles."
            }
        ]
    },
  ...
]
```

</details>

<details>
<summary>Example for multi image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "role": "user",
        "content": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "role": "assistant",
        "content": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for video dataset</summary>

```json
[
    {
        "id": "00001",
        "video": "MER2024/sample_00002721.mp4",
        "conversations": [
            {
                "role": "user",
                "content": "<video>\nReply with a single word indicating the emotion category."
            },
            {
                "role": "assistant",
                "content": "neutral"
            }
        ]
    },
  ...
]
```

</details>
<br><br>


## Training

**Note:** Deepspeed zero2 is faster than zero3, however it consumes more memory. Also, most of the time zero2 is more stable than zero3.<br><br>

To run the training script, use the following command:

### Full Finetuning

```bash
cd EmotionQwen-main/train
bash finetune_ds.sh
```


### Finetune with LoRA

```bash
cd EmotionQwen-main/train
bash finetune_lora.sh
```


<details>
<summary>Training arguments</summary>

- `--data_path` (str): Path to the formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images or videos folder as referenced in the formatted training data. **(Required)**
- `--model_name_or_path` (str): Path to the Emotion-Qwen model. **(Required)**
- `--use_liger` (bool): Option for using liger kernel to save memory.
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps.
- `--tune_vision` (bool): Option to fine-tune vision_model.
- `--tune_llm` (bool): Option to fine-tune LLM.
- `--tune_general_compressor` (bool): Option to tune general_compressor.
- `--tune_emotion_compressor` (bool): Option to tune emotion_compressor.
- `--lora_enable` (bool): Option for using LoRA.
- `--lora_target_modules` (str): Target modules to add LoRA.
- `--learning_rate` (float): Learning rate for model.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--image_max_pixles` (int): Option for maximum maxmimum tokens for image.
- `--video_max_pixles` (int): Option for maximum maxmimum tokens for video.
- `--model_max_length` (int): Maximum sequence length.
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers.

</details>

#### Merge LoRA Weights

The `finetune_ds.sh` or `finetune_lora.sh` will automatically implement Lora merging, if it fails, manually execute it:

```
cd EmotionQwen-main/train
python merge_lora.py
```

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find Emotion-Qwen useful to you, please consider giving a :star: and citing:

```bibtex
@misc{Qwen2-VL-Finetuning,
  author = {Yuwon Lee},
  title = {Qwen2-VL-Finetune},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Qwen2-VL-Finetune}
}
```

## Acknowledgement

The whole training scipt is based on:  [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-o), [Qwen2-VL-Finetuning](https://github.com/2U1/Qwen2-VL-Finetune).
