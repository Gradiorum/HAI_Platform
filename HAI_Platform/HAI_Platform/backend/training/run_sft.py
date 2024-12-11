import os
import json
import yaml
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from typing import Optional, Dict, Any, Union

# If using TRL’s SFTTrainer
try:
    from trl import SFTConfig, SFTTrainer
    HAS_TRL = True
except ImportError:
    HAS_TRL = False

# If using unsloth (example)
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

# If using TorchTune (example)
# Actual import paths depend on your torchtune installation
try:
    # pseudo import - replace with actual torchtune imports
    # from torchtune import run_recipe, TuneRecipeArgumentParser
    HAS_TORCHTUNE = True
except ImportError:
    HAS_TORCHTUNE = False

load_dotenv()  # Load environment variables from .env

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Merge environment variables if needed
    # For example, device from ENV if specified:
    env_device = os.environ.get("DEVICE")
    if env_device:
        config["device"] = env_device

    return config

def prepare_dataset(paths, text_field="text", formatting_func=None, packing=False, max_seq_length=1024):
    """
    Load and optionally concatenate multiple datasets. 
    formatting_func: a callable that processes examples into strings.
    If packing is True and using TRL’s SFTTrainer with ConstantLengthDataset,
    rely on SFTTrainer to handle it. Otherwise just return dataset(s).
    """
    if not isinstance(paths, list):
        paths = [paths]

    datasets = []
    for p in paths:
        # p could be a local file or a HuggingFace dataset name
        ds = load_dataset("json", data_files=p, split="train")
        # If a formatting_func is provided, apply it to produce a unified text field
        if formatting_func:
            ds = ds.map(lambda ex: {text_field: formatting_func(ex)}, batched=True)
        datasets.append(ds)

    if len(datasets) > 1:
        combined = concatenate_datasets(datasets)
    else:
        combined = datasets[0]

    # No tokenization here if we rely on SFTTrainer or other frameworks to do it
    # If you want pretokenization, do it here:
    return combined

def prepare_huggingface_trainer(model, tokenizer, train_data, val_data, sft_args):
    # Convert sft_args to TrainingArguments
    # Example: sft_args could hold batch_size, lr, epochs, etc.
    training_args = TrainingArguments(
        output_dir=sft_args.get("output_dir", "./outputs"),
        per_device_train_batch_size=sft_args.get("batch_size", 2),
        gradient_accumulation_steps=sft_args.get("gradient_accumulation_steps", 4),
        max_steps=sft_args.get("max_steps", 1000),
        logging_steps=sft_args.get("logging_steps", 10),
        evaluation_strategy=sft_args.get("evaluation_strategy", "no"),
        fp16=sft_args.get("fp16", False),
        bf16=sft_args.get("bf16", False),
        save_steps=sft_args.get("save_steps", 500),
        # Add any other HF arguments from sft_args
    )

    data_collator = None  # or create a data collator if needed

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer

def prepare_trl_sft_trainer(model_or_name, train_data, val_data, tokenizer, sft_args):
    if not HAS_TRL:
        raise ImportError("TRL not installed. Cannot run SFTTrainer.")

    # Convert sft_args to SFTConfig
    sft_config = SFTConfig(
        output_dir=sft_args.get("output_dir", "./outputs"),
        max_seq_length=sft_args.get("max_seq_length", 1024),
        packing=sft_args.get("packing", False),
        # map more sft_args as needed
    )

    # If formatting_func is needed, pass it here
    formatting_func = sft_args.get("formatting_func")

    peft_config = sft_args.get("peft_config", None)  # optional PEFT config if using LoRA

    trainer = SFTTrainer(
        model=model_or_name,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=sft_config,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        peft_config=peft_config,
    )
    return trainer

def prepare_unsloth_trainer(model_name, train_data, sft_args):
    if not HAS_UNSLOTH:
        raise ImportError("Unsloth not installed.")

    # Example: load unsloth model
    max_seq_length = sft_args.get("max_seq_length", 1024)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=sft_args.get("load_in_4bit", False),
    )

    # If LoRA or QLoRA is desired:
    if sft_args.get("use_lora", False):
        model = FastLanguageModel.get_peft_model(
            model,
            r=sft_args.get("lora_r",16),
            target_modules=sft_args.get("lora_target_modules",["q_proj", "k_proj"]),
            lora_alpha=sft_args.get("lora_alpha",32),
            lora_dropout=sft_args.get("lora_dropout",0.05),
            bias=sft_args.get("lora_bias","none"),
            use_gradient_checkpointing=sft_args.get("use_gradient_checkpointing",False),
            random_state=sft_args.get("seed", 42),
            max_seq_length=max_seq_length
        )

    # Then use TRL's SFTTrainer again with the unsloth model
    # or a custom training loop. For simplicity, here’s TRL’s SFTTrainer integration:
    return prepare_trl_sft_trainer(model, train_data, None, tokenizer, sft_args)

def prepare_torchtune_recipe(sft_args):
    if not HAS_TORCHTUNE:
        raise ImportError("TorchTune not installed.")

    # Example pseudo-code for TorchTune (adjust to your actual TorchTune usage):
    # This might call a CLI or a function from TorchTune that runs a recipe.
    # Typically torchtune might rely on a YAML config of its own.
    # For now, just a placeholder.
    recipe_config = sft_args.get("torchtune_config", "path/to/recipe.yaml")

    # You might do something like:
    # run_recipe(recipe_config, overrides=[f"model={sft_args['model_name']}"])
    # return nothing or a status code
    pass

def train_sft(model: Optional[str],
              sft_train: Union[str, list],
              sft_test: Optional[str],
              sft_val: Optional[str],
              sft_args: Dict[str, Any]):

    # 1. Load model if given as a string:
    # If using a HuggingFace model:
    backend = sft_args.get("backend", "huggingface")
    device = sft_args.get("device", "cuda")
    dtype = sft_args.get("dtype", "float16")

    # Load the datasets
    formatting_func = sft_args.get("formatting_func", None)
    train_data = prepare_dataset(sft_train, 
                                 text_field=sft_args.get("text_field","text"), 
                                 formatting_func=formatting_func, 
                                 packing=sft_args.get("packing", False),
                                 max_seq_length=sft_args.get("max_seq_length",1024))

    val_data = prepare_dataset(sft_val, text_field=sft_args.get("text_field","text"), 
                               formatting_func=formatting_func) if sft_val else None

    # 2. Based on backend, pick training approach:
    if backend == "huggingface":
        # load model and tokenizer
        if isinstance(model, str):
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=(torch.float16 if dtype=="float16" else torch.bfloat16))
        else:
            # model is already a loaded model instance
            tokenizer = sft_args["tokenizer"]

        trainer = prepare_huggingface_trainer(model, tokenizer, train_data, val_data, sft_args)
        trainer.train()
        trainer.save_model(sft_args.get("output_dir", "./outputs"))

    elif backend == "trl_sft":
        if not HAS_TRL:
            raise RuntimeError("TRL is not installed.")
        if isinstance(model, str):
            # SFTTrainer can load model by name
            model_or_name = model
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        else:
            model_or_name = model
            tokenizer = sft_args["tokenizer"]
        trainer = prepare_trl_sft_trainer(model_or_name, train_data, val_data, tokenizer, sft_args)
        trainer.train()
        trainer.save_model()

    elif backend == "unsloth":
        if not HAS_UNSLOTH:
            raise RuntimeError("Unsloth is not installed.")
        # unsloth approach typically:
        if not isinstance(model, str):
            raise ValueError("Unsloth requires a model name string.")
        trainer = prepare_unsloth_trainer(model, train_data, sft_args)
        trainer.train()
        trainer.save_model()

    elif backend == "torchtune":
        # With TorchTune, you might not even call model directly here.
        # You’d run a torchtune recipe that references the model and data.
        prepare_torchtune_recipe(sft_args)
        # This might not return a trainer object since TorchTune may run end-to-end.
        # Possibly you run a CLI command from here or a function that starts training.

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    print("Training complete.")
