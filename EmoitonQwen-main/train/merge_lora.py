from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import os
import shutil



model_type = "path/to/your/base_model"
path_to_adapter = "/path/to/your/lora_adapter"  # Path to the saved LoRA adapter
merge_path = "/path/to/the/output/merged_model"  # Path to save the merged model


def copy_files_not_in_B(A_path, B_path):
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

model = AutoModel.from_pretrained(
    model_type,
    trust_remote_code=True
)

lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()

merge_model = lora_model.merge_and_unload()
merge_model._hf_peft_config_loaded = False

merge_model.save_pretrained(merge_path, safe_serialization=True, max_shard_size = "5GB")

tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
tokenizer.save_pretrained(merge_path)

copy_files_not_in_B(model_type,merge_path)