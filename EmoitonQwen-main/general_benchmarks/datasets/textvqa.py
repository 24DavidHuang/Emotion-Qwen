import os
import re
import pandas as pd
from tqdm import tqdm, trange
from datasets import load_dataset
from .base_eval_dataset import BaseEvalDataset
import pytz
import datetime
import json

utc_plus_8 = pytz.timezone("Asia/Shanghai")  # You can also use 'Asia/Singapore', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)


class TextVQADataset(BaseEvalDataset):
    def __init__(
        self, 
        data_path: str = "./TextVQA", 
        *, 
        split="validation", 
        cache_dir=None, 
        default_output_path="./benchmarks/logs/TextVQA", 
        batch=1, 
        debug=False, 
        prompt = "Answer the question directly with single word or phase."
    ):
        super().__init__("TextVQADataset", data_path, max_batch_size=batch)
        self.split = split
        self.data = load_dataset(data_path, split=self.split, cache_dir=cache_dir)
        self.default_output_path = default_output_path
        self.cur_datetime = utc_plus_8_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.debug = debug
        self.prompt = prompt


    def parse_pred_ans(self, pred_ans, options):
        cleaned_pred_ans = pred_ans.strip().lower()

        unique_options = {}
        for idx, option in enumerate(options):
            cleaned_option = option.strip().lower()
            if cleaned_option not in unique_options:
                # use option as the key to remove duplicates, index as the value
                unique_options[cleaned_option] = idx

        for option, idx in unique_options.items():
            if option in cleaned_pred_ans:
                return options[idx]  

        return "other"

    def _evaluate(self, model, *, batch=1):
        subdir = os.path.join(self.default_output_path, f"{model.name}_textvqa_eval_{self.cur_datetime}")
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        output_file = os.path.join(self.default_output_path, f"{model.name}_textvqa_eval_{self.cur_datetime}/{model.name}_textvqa_eval_result_{self.cur_datetime}.json")
        result_file = os.path.join(self.default_output_path, f"{model.name}_textvqa_eval_{self.cur_datetime}/{model.name}_textvqa_eval_score_{self.cur_datetime}.json")
        results = []

        total = 0
        total_correct = 0

        with tqdm(total=len(self.data), desc="Evaluating") as pbar:
            for data_dict in self.data:
                question = data_dict["question"]
                answers = data_dict["answers"]
                text = f"{self.prompt}\n{question}"

                output = model.generate(text, data_dict["image"])
                phrased_output = self.parse_pred_ans(output, answers)
                print(f"Question: {question}\n Answer: {answers}\n Output: {output}, Phrased Output: {phrased_output}")
                correct = phrased_output in answers

                if correct:
                    total_correct += 1
                    print("correct!!")
                else:
                    print("wrong!")
                
                total += 1
                results.append(
                    {
                        "question": question,
                        "answer": answers,
                        "output": output,
                        "prediction": phrased_output,
                        "correct": correct,
                    }
                )
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)
                    f.flush()

                accuracy = total_correct / total * 100 if total > 0 else 0
                pbar.set_postfix(accuracy=f"{accuracy:.2f}%")
                pbar.update(1)

            score = total_correct / total
            print(f"TextVQA Evaluator: Total: {total}")
            print(f"TextVQA Evaluator: Total correct: {total_correct}")
            print(f"TextVQA Evaluator: Score: {score}")
            with open(result_file, "w") as f:
                final_score = {
                    "score": score,
                    "total": total,
                    "correct": total_correct,
                }
                json.dump(final_score, f, indent=4)

            print(f"TextVQA Evaluator: Result saved to {os.path.abspath(output_file)}.")
            print(f"TextVQA Evaluator: Score saved to {os.path.abspath(result_file)}.")


if __name__ == "__main__":
    dataset = TextVQADataset(cache_dir="/data/hdw/cache")
    data = dataset.data
    print("=============================")
    import json

