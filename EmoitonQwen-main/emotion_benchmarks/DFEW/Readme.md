# DFEW Evaluation
1. Visit the following link to download the labels and raw videos for the DFEW dataset: [Dynamic Facial Expression in-the-Wild](https://dfew-dataset.github.io/download.html).

    Permission to use but not reproduce or distribute the DFEW database is granted to all researchers given that the following steps are properly followed:
    Send an E-mail to Xingxun Jiang(jiangxingxun@seu.edu.cn) and Yuan Zong(xhzongyuan@seu.edu.cn) before downloading the database. You will need a password to access the files of DFEW database.

2. Move `test(single-labeled)` file from origin DFEW dataset to the following directory:  `EmotionQwen-main/eval_data/DFEW/`
    `test(single-labeled)` contains 5 sets for five fold cross validation.

        test(single-labeled)/:
            --set_1.csv
            --set_2.csv
            --set_3.csv
            --set_4.csv
            --set_5.csv


3. Move DFEW videos to the following directory: `/home/dell/EmotionQwen-main/eval_data/DFEW/videos`

4. Run the `dfew_emotion_XXX.py` file corresponding to the model and generate DFEW evaluation results.
