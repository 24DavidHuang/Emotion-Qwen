import glob
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union, Literal, Tuple
from types import MethodType
from torchvision import transforms

import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers.integrations import deepspeed

from dataset import SupervisedDataset, data_collator
from trainer import EmotionQwenTrainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def copy_files_not_in_B(A_path, B_path):
    import shutil
    """
    Copies files from directory A to directory B if they exist in A but not in B.

    :param A_path: Path to the source directory (A).
    :param B_path: Path to the destination directory (B).
    """
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"The directory {A_path} does not exist.")
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    files_in_A = os.listdir(A_path)
    files_in_A = set([file for file in files_in_A if not (".bin" in file or "safetensors" in file)])
    # List all files in directory B
    files_in_B = set(os.listdir(B_path))
    files_to_copy = files_in_A - files_in_B

    for file in files_to_copy:
        src_file = os.path.join(A_path, file)
        dst_file = os.path.join(B_path, file)
        shutil.copy2(src_file, dst_file)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EmotionQwen")
    image_folder: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    image_max_pixels: int = field(default = 1280*28*28)
    video_max_pixels: int = field(default = 360*420)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    tune_general_compressor: Optional[bool] = field(default=True)
    tune_emotion_compressor: Optional[bool] = field(default=True)
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)
    external_file_dir: Optional[str] = field(default=None)
    extra_files: Optional[List[str]] = field(default=None)
    disable_flash_attn2: Optional[bool] = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def print_trainable_modules(model):
    print("Trainable lora modules:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def safe_save_model_for_hf_trainer(trainer: EmotionQwenTrainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed and trainer.args.local_rank == 0:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    processor: transformers.BaseImageProcessor,
    data_args,
    transform,
    data_collator=None,
    batch_vision=False,
    max_length=2048,
    image_folder = "",
    image_max_pixels=None,
    vidoe_max_pixels=None,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_json,
        transform,
        tokenizer,
        processor,
        batch_vision=batch_vision,
        max_length=max_length,
        image_folder=image_folder,
        image_max_pixels=image_max_pixels,
        vidoe_max_pixels=vidoe_max_pixels,
    )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json,
            transform,
            tokenizer,
            processor,
            batch_vision=batch_vision,
            max_length=max_length,
            image_folder=image_folder,
            image_max_pixels=image_max_pixels,
            vidoe_max_pixels=vidoe_max_pixels,
        )
    else:
        eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator= partial(data_collator),
    )


def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        
    return {'Total': all_param, 'Train': trainable_params, 'Trainable (%)': 100. * trainable_params / all_param}


local_rank = 0


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None) : 
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # new added
    if training_args.external_file_dir == None:
        training_args.external_file_dir = model_args.model_name_or_path

    if training_args.extra_files==None: 
        training_args.extra_files = [
            'configuration_emotionqwen_vl.py',
            'modeling_emotionqwen_vl.py',
            'processing_emotionqwen_vl.py',
            'preprocessor_config.json',
            'chat_template.json'
        ]

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )
    
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
        device_map=device_map
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
        padding_side="right",
    )
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side

    if not training_args.tune_vision:
        model.vpm.requires_grad_(False)
    if not training_args.tune_llm:
        model.llm.requires_grad_(False)
        model.lm_head.requires_grad_(False)

    if not training_args.tune_general_compressor:
        general_compressor_para = model.generalcompressor.parameters()
        for p in general_compressor_para:
            p.requires_grad = False

    if not training_args.tune_emotion_compressor:
        emotion_compressor_para = model.emotioncompressor.parameters()
        for p in emotion_compressor_para:
            p.requires_grad = False

        
    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")
            
        rank0_print("Currently using LoRA for fine-tuning the EmotionQwen model.")
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
        modules_to_save = ['embed_tokens','gatenet']
        if training_args.tune_general_compressor:
            modules_to_save.append('generalcompressor')
        if training_args.tune_emotion_compressor:
            modules_to_save.append('emotioncompressor')
        if training_args.tune_vision:
            modules_to_save.append('vpm')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        # print_trainable_modules(model)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    transform_func = build_transform()
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
        transform=transform_func,
        data_collator=data_collator,
        batch_vision=batch_vision,
        max_length=training_args.model_max_length,
        image_folder = model_args.image_folder,
        image_max_pixels=data_args.image_max_pixels,
        vidoe_max_pixels=data_args.video_max_pixels,
    )
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    training_args.gradient_checkpointing_kwargs={"use_reentrant":False}

    trainer = EmotionQwenTrainer(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        args=training_args,
        **data_module,
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir)

    
    if training_args.use_lora:
        from peft import PeftModel
        basemodel = trainer.model.base_model
        # basemodel = AutoModel.from_pretrained(
        #     model_args.model_name_or_path,
        #     trust_remote_code=True
        # )

        lora_model = PeftModel.from_pretrained(
            basemodel,
            training_args.output_dir,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        merge_model = lora_model.merge_and_unload()
        # merge_model._hf_peft_config_loaded = False

        merge_path = os.path.join(os.path.dirname(training_args.output_dir), 'merged_model')
        merge_model.save_pretrained(merge_path, safe_serialization=True ,max_shard_size = "5GB")

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        tokenizer.save_pretrained(merge_path)

        copy_files_not_in_B( model_args.model_name_or_path, merge_path)


if __name__ == "__main__":
    train()