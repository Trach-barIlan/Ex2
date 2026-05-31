# Qwen 2.5 7B LoRA Fine-Tuning Guide

This guide explains how the fine-tuning script is structured, how it utilizes your data, and how to execute it on the cluster using the provided Slurm script.

## 1. The Dataset
Yes, the script strictly uses `full_hebrew_sft_data.jsonl`. 
The `datasets` library loads this file, and the script iterates through it. We specifically use the `apply_chat_template` method to format the `{"role": ..., "content": ...}` conversations exactly the way `Qwen/Qwen2.5-7B-Instruct` was originally trained to see them (with its specific special tokens). This ensures that the English prompts and the Hebrew responses map correctly to the required task.

## 2. What is "Overfitting a Tiny Dataset" (Sanity Check)?
When starting a fine-tuning pipeline for a new task or model, the best sanity check is to verify the model can perfectly memorize 1 or 2 examples. This is called **overfitting**.

**How the script handles it:**
I updated `finetune_qwen.py` to accept an `--overfit` argument. When you pass this argument, the script:
1. Slices the data to use only the **first 2 examples** of `full_hebrew_sft_data.jsonl`.
2. Sets the `num_train_epochs` to `20` (so it sees those same 2 examples repeatedly).
3. Logs the loss at every single step.

**How to verify success:** 
When the job runs, look at the Slurm `.out` file. You should see the Training Loss decrease rapidly towards `0.0`. If it does, you know your LoRA setup, chat templates, loss calculations, and optimizer are all working correctly.

## 3. How to Run the Code

Since all code is meant to run using the provided `run_generic.slurm` script, you use `sbatch` and pass the Python file (along with any arguments) directly to it.

### Step A: Run the Overfit Sanity Check First
Run this command in your terminal from the `Ex2` directory:
```bash
sbatch run_generic.slurm finetune_qwen.py --overfit
```
* **What it does:** Submits a job to run the overfitting check. 
* **Output:** It will save the test weights to `./qwen-hebrew-lora-overfit-final`. Monitor the generated `generic_<job_id>.out` file to ensure the loss drops toward 0.

### Step B: Run the Full Fine-Tuning
Once you confirm the overfit run was successful and the loss went down, you can run the full dataset:
```bash
sbatch run_generic.slurm finetune_qwen.py
```
* **What it does:** Submits a job that trains the LoRA adapters over the entire `full_hebrew_sft_data.jsonl` dataset for 1 epoch.
* **Output:** The final fine-tuned LoRA weights will be saved to the `./qwen-hebrew-lora-final` directory.
