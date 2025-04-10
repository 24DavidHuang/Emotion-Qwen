import os
import pandas as pd
from tqdm import tqdm, trange
from datasets import load_dataset
from .base_eval_dataset import BaseEvalDataset
import pytz
import datetime

utc_plus_8 = pytz.timezone("Asia/Shanghai")  # You can also use 'Asia/Shanghai', 'Asia/Taipei', etc.
utc_now = pytz.utc.localize(datetime.datetime.utcnow())
utc_plus_8_time = utc_now.astimezone(utc_plus_8)

class MMBenchDataset(BaseEvalDataset):
    def __init__(
        self,
        data_path: str = "mmbench-en",
        *,
        sys_prompt="Your task is to select the most appropriate option based on the provided hint, question, and image. Respond with only the letter of the chosen option (e.g., A, B, C, D).",
        version="default",
        split="",
        cache_dir=None,
        default_output_path="./benchmarks/logs/MMBench",
        debug=False,
    ):
        super().__init__("MMBenchDataset", data_path)
        self.version = str(version)
        self.name_converter = {"dev": "validation", "test": "test"}

        self.df = load_dataset(data_path, self.version, split=self.name_converter[split], cache_dir=cache_dir).to_pandas()


        self.default_output_path = default_output_path
        if os.path.exists(self.default_output_path) is False:
            os.makedirs(self.default_output_path)
        self.sys_prompt = sys_prompt
        self.cur_datetime = utc_plus_8_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.debug = debug

    def load_from_df(self, idx, key):
        if key in self.df.columns:
            value = self.df.loc[idx, key]
            return value if pd.notna(value) else None
        return None

    def create_options_prompt(self, idx, option_candidate):
        available_keys = set(self.df.columns) & set(option_candidate)
        options = {cand: self.load_from_df(idx, cand) for cand in available_keys if self.load_from_df(idx, cand)}
        sorted_options = dict(sorted(options.items()))
        options_prompt = f"Options:\n"
        for key, item in sorted_options.items():
            options_prompt += f"{key}. {item}\n"
        return options_prompt.rstrip("\n"), sorted_options

    def get_data(self, idx):
        row = self.df.loc[idx]
        option_candidate = ["A", "B", "C", "D"]
        options_prompt, options_dict = self.create_options_prompt(idx, option_candidate)

        data = {
            "img": row["image"],
            "question": row["question"],
            # "answer": row.get("answer"), #如果测试test_tsv则没有answer
            "options": options_prompt,
            "category": row["category"],
            "l2-category": row["l2-category"],
            "options_dict": options_dict,
            "index": row["index"],
            "hint": self.load_from_df(idx, "hint"),
            "source": row["source"],
            "split": row["split"],
        }

        if "answer" in row:
            data["answer"] = row["answer"]

        return data



    def _evaluate(self, model, *, batch=1):
        output_file = os.path.join(self.default_output_path, f"{model.name}_mmbench_eval_result_{self.cur_datetime}.xlsx")
        results = []
        correct_predictions = 0  

        progress_bar = tqdm(range(len(self.df)), desc="Evaluating", unit="item")
        
        for idx in progress_bar:
            cur_data = self.get_data(idx)
            
            cur_prompt = f"{cur_data['hint']}\n {cur_data['question']}\n {cur_data['options']}\n {self.sys_prompt}\n" \
                            if pd.notna(cur_data["hint"]) and cur_data["hint"] != "nan" \
                            else f"{cur_data['question']}\n {cur_data['options']}\n {self.sys_prompt}\n"
            
            pred_answer = model.generate(cur_prompt, cur_data["img"])

            if "answer" in cur_data:
                is_correct =  cur_data["answer"].strip().lower() in pred_answer.strip().lower()[:1] or pred_answer.strip().lower() == cur_data["answer"].strip().lower()
                if is_correct:
                    correct_predictions += 1
                
                tqdm.write(f"question: {cur_data['question']}\nprediction: {pred_answer}\nlabel: {cur_data['answer']}\nis_correct: {is_correct}")
            else:
                tqdm.write(f"question: {cur_data['question']}\nprediction: {pred_answer}")

            result = {
                "index": cur_data["index"],
                "question": cur_data["question"],
                **cur_data["options_dict"],
                "prediction": pred_answer,
            }
            if "answer" in cur_data:
                result["answer"] = cur_data["answer"]
            results.append(result)

            accuracy = correct_predictions / (idx + 1)
            progress_bar.set_description(f"Evaluating (Accuracy: {accuracy:.2%})")

        df = pd.DataFrame(results)
        columns_order = ["index", "question", "A", "B", "C", "D", "prediction"]
        if "answer" in df.columns:
            columns_order.append("answer")
        df = df[columns_order]

        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        print(f"MMBench Evaluator: Result saved to {output_file}.")
