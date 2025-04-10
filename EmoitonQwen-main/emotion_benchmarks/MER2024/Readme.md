# MER2024 Evaluation
1. Visit the following link to download the labels and videos for the MER2024 Challenge: [MER2024 Challenge](https://zeroqiaoba.github.io/MER2024-website/).

    To download the dataset, please contact to merchallenge.contact@gmail.com.

2. Move `reference-noise.csv` and `reference-semi.csv` label files to the following directory:  `EmotionQwen-main/eval_data/MER2024/`

3. Move NOISE Track `MER2024_noise` videos to the following directory: `EmotionQwen-main/eval_data/MER2024/MER2024_noise/`
4. Move SEMI Track `MER2024_semi` videos to the following directory: `EmotionQwen-main/eval_data/MER2024/MER2024_semi/`
5. Visit the code `MER2024_XXX.py` file corresponding to the model, set `Track = "NOISE"` or `Track = "SEMI"` and generate MER2024 evaluation results.
