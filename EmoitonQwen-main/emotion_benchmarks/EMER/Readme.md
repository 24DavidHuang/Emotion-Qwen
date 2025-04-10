# EMER Evaluation
1. Visit the following link to download the ground truth labels for the EMER dataset: [AffectGPT/EMER/dataset-v1](https://github.com/zeroQiaoba/AffectGPT/tree/master/EMER/dataset-v1)

    **`gt-chi.csv`** : Contains the ground truth annotations in Chinese.

    **`gt-eng.csv`** : Contains the ground truth annotations in English.
2. Move these label files to the following directory:  `EmotionQwen-main/emotion_benchmarks/EMER/results/`
3. Download Raw [MER 2023 Dataset](https://dl.acm.org/doi/abs/10.1145/3581783.3612836)

    To download the dataset, please fill out an [EULA](https://drive.google.com/file/d/1LOW2e6ZuyUjurVF0SNPisqSh4VzEl5lN) and send it to lianzheng2016@ia.ac.cn.

4. Move raw videos to the following directory: `EmotionQwen-main/eval_data/EMER/videos/`
5. Run the `generate_csv_XXX.py` file corresponding to the model and generate a reply csv file.
6. Run `evaluate_by_GPT.py` to generate EMER evaluation results.
