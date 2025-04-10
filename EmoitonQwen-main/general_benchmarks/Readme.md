# Gnenral Benchmarks Evaluation

1. Visit the following link to download general benchmarks:

    + [MME](https://www.modelscope.cn/datasets/lmms-lab/MME)

    + [MMBench](https://www.modelscope.cn/datasets/lmms-lab/MMBench)

    + [POPE](https://www.modelscope.cn/datasets/lmms-lab/POPE)

    + [ScienceQA](https://www.modelscope.cn/datasets/swift/ScienceQA)

    + [SeedBench](https://www.modelscope.cn/datasets/lmms-lab/SEED-Bench)

    + [TextVQA](https://www.modelscope.cn/datasets/lmms-lab/textvqa)

    + [VQAv2](https://www.modelscope.cn/datasets/lmms-lab/VQAv2)


2. Open `benchmark.yaml` and change the path to your downloaded benchmark and models.

```yaml
benchmark.yaml:

datasets:

  - name: mmbench
    data_path: path/to/your/mmbench-en
    split: dev

# If you don't want to evaluate this benchmark, then annotate it:

#   - name: mme
#     data_path: path/to/your/MME 
#     split: test
  
models:

  - name: emotion_qwen
    model_path: /path/to/EmotionQwen

# If you don't want to evaluate this model, then annotate it:

#   - name: XXX
#     model_path: /xxx/xxx

```


3. Perform the following operations in the terminal to start the evaluation:

    ```bash
    cd ./EmotionQwen-main
    CUDA_VISIBLE_DEVICES=0 python -m general_benchmarks.evaluate --config benchmark.yaml
    ```

4. The results will be stored in the `/EmotionQwen-main/logs` folder
