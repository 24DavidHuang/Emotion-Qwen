import numpy as np
from tqdm import tqdm
from .base_eval_dataset import BaseEvalDataset
from datasets import load_dataset
import json
import os
import datetime
import re
import torch

# SEEDBench download path: https://www.modelscope.cn/datasets/lmms-lab/SEED-Bench/summary

class SEEDBenchDataset(BaseEvalDataset):
    def __init__(self, 
        data_path: str = "./SEEDBench", 
        split="", 
        default_output_path="./benchmarks/logs/SEEDBench", 
        cache_dir=None,
        sys_prompt="Your task is to select the most appropriate option based on the provided question and image. Respond with only the letter of the chosen option (e.g., A, B, C, D).",
        ):
        super().__init__("SEEDBenchDataset", data_path)
        print("Loading dataset from", data_path)
        self.data = load_dataset(data_path, split=split, cache_dir=cache_dir)
        self.default_output_path = default_output_path
        if not os.path.exists(default_output_path):
            os.makedirs(default_output_path)
        self.sys_prompt = sys_prompt

    
    def parse_pred_ans(self, pred_ans):
        pred_ans = pred_ans.lower().strip()

        if pred_ans in ["a", "b", "c", "d"]:
            return pred_ans.upper()

        match = re.match(r'^([abcd])[\.\:\)\s]', pred_ans)
        if match:
            return match.group(1).upper()

        prefix_pred_ans = pred_ans[:10]
        for letter in ["a", "b", "c", "d"]:
            if letter in prefix_pred_ans:
                return letter.upper()
            
        return "Unknown"

    def _evaluate(self, model):
        count = 0
        num_correct = 0
        cur_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        output_path = os.path.join(self.default_output_path, f"seedbench_{model.name}_test_submit_{cur_datetime}.json")
        output_f = open(output_path, "a")
        with tqdm(total=len(self.data), desc="Evaluating") as pbar:
            for data_dict in self.data:
                image = data_dict["image"]
                question = data_dict["question"] + " There are several options:\n"
                option_index = ["A", "B", "C", "D"]
                for cur_idx in range(4):
                    question += f" {option_index[cur_idx]}. {data_dict[f'choice_{option_index[cur_idx].lower()}']}\n"
                
                question += f"{self.sys_prompt}\n"
                print(f"question:{question}\n")

                answer = data_dict["answer"]
                prediction = model.generate(question, image)


                prediction_num = self.parse_pred_ans(prediction)
                print(f"answer: {answer}, prediction: {prediction}, prediction_num: {prediction_num}")

                if prediction_num == answer:
                    num_correct += 1
                count += 1

                answer_record = {"question_id": data_dict["question_id"], "answer": answer, "prediction": prediction_num}
                output_f.write(json.dumps(answer_record) + "\n")
                output_f.flush()

                # Update accuracy
                accuracy = num_correct / count * 100 if count > 0 else 0
                pbar.set_postfix(accuracy=f"{accuracy:.2f}%")
                pbar.update(1)

                torch.cuda.empty_cache()

        accuracy = num_correct / count * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy
