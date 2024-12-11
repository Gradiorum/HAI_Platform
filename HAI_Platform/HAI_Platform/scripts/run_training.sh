#!/bin/bash
# Example script to run DPO training
python -m backend.training.dpo_training --dataset_path dpo_samples.json --base_model llama-2 --output_dir dpo_finetuned
