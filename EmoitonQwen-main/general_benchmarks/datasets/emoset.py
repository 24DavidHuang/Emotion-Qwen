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


class EmoSetDataset(BaseEvalDataset):
    def __init__(
        self, 
        data_path: str = "./EmoSet", 
        *, 
        split="validation", 
        cache_dir=None, 
        default_output_path="./benchmarks/logs/EmoSet", 
        batch=1, 
        debug=False, 
        prompt = "Answer the question directly with single word or phase."
    ):
        super().__init__("EmoSetDataset", data_path, max_batch_size=batch)
        self.split = split
        # self.data = pd.read_parquet(data_path)
        self.data = load_dataset(data_path, split=self.split, cache_dir=cache_dir)
        self.default_output_path = default_output_path
        self.cur_datetime = utc_plus_8_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.debug = debug
        self.prompt = prompt


    def parse_pred_ans(self, pred_ans, answer):
        cleaned_pred_ans = pred_ans.strip().lower()

        if answer.lower() in cleaned_pred_ans:
            return answer
        else:
            return "other"

    def _evaluate(self, model, *, batch=1):
        subdir = os.path.join(self.default_output_path, f"{model.name}_emoset_eval_{self.cur_datetime}")
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        output_file = os.path.join(self.default_output_path, f"{model.name}_emoset_eval_{self.cur_datetime}/{model.name}_emoset_eval_result_{self.cur_datetime}.json")
        result_file = os.path.join(self.default_output_path, f"{model.name}_emoset_eval_{self.cur_datetime}/{model.name}_emoset_eval_score_{self.cur_datetime}.json")
        results = []

        total = 0
        total_correct = 0

        with tqdm(total=len(self.data), desc="Evaluating") as pbar:
            for data_dict in self.data:
                # print(data_dict)
                question = data_dict["question"]
                answer = data_dict["answer"]
                text = f"{self.prompt}\n{question}"
                output = model.generate(text, data_dict['image'])
                phrased_output = self.parse_pred_ans(output, answer)
                print(f"Question: {question}\n Answer: {answer}\n Output: {output}, Phrased Output: {phrased_output}")
                correct = phrased_output == answer

                if correct:
                    total_correct += 1
                    print("correct!!")
                else:
                    print("wrong!")
                
                total += 1
                results.append(
                    {
                        "question": question,
                        "answer": answer,
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
            print(f"EmoSet Evaluator: Total: {total}")
            print(f"EmoSet Evaluator: Total correct: {total_correct}")
            print(f"EmoSet Evaluator: Score: {score}")
            with open(result_file, "w") as f:
                final_score = {
                    "score": score,
                    "total": total,
                    "correct": total_correct,
                }
                json.dump(final_score, f, indent=4)

            print(f"EmoSet Evaluator: Result saved to {os.path.abspath(output_file)}.")
            print(f"EmoSet Evaluator: Score saved to {os.path.abspath(result_file)}.")


if __name__ == "__main__":
    dataset = EmoSetDataset(cache_dir="/data/hdw/cache")
    data = dataset.data
    print("=============================")
    import json

