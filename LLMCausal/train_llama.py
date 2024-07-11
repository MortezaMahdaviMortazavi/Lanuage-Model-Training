import torch
import pandas as pd
from datasets import Dataset , load_dataset
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM, 
	BitsAndBytesConfig,
	TrainingArguments    
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from dataclasses import dataclass
from trl import SFTTrainer , setup_chat_format
import os
import argparse
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Masked Language Model with LoRA")
    parser.add_argument('--output_dir', type=str, default="llama3-version3", help="Directory to save the trained model")
    parser.add_argument('--logger_file',type=str,default="logs/llama_train.log",help="logger file address for logging the training config")
    parser.add_argument('--num_train_epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help="Batch size per device during training")
    parser.add_argument('--gradient_accumulation', type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Use gradient checkpointing to save memory at the expense of slower backward pass")
    parser.add_argument('--device',type=str,default="cuda:0")
    parser.add_argument('--optim', type=str, default="adamw_8bit", help="Optimizer to use")
    parser.add_argument('--logging_steps', type=int, default=5, help="Log every X updates steps")
    parser.add_argument('--save_strategy', type=str, default="epoch", help="The checkpoint save strategy to use")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--fp16', action='store_true', help="Use mixed precision training")
    parser.add_argument('--max_grad_norm', type=float, default=0.3, help="Max gradient norm")
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help="Warmup ratio for learning rate scheduler")
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help="The scheduler type to use")
    parser.add_argument('--disable_tqdm', action='store_true', help="Disable tqdm progress bar")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="Dropout rate for LoRA")
    parser.add_argument('--lora_alpha', type=int, default=16, help="Alpha parameter for LoRA")
    parser.add_argument('--max_seq_length',type=int,default=256,help="Max sequence length for giving input to model")
    parser.add_argument('--r', type=int, default=16, help="Rank parameter for LoRA")
    parser.add_argument('--task_type', type=str, default="CAUSAL_LM", help="Task type for LoRA")
    parser.add_argument('--bias', type=str, default="none", help="Bias parameter for LoRA")
    parser.add_argument('--model_name', type=str, default="unsloth/llama-3-8b-bnb-4bit", help="Name of the pretrained model")
    parser.add_argument('--dataset_path', type=str, default='../datasets/6820.csv', help="Path to the dataset")
    parser.add_argument('--input_column',type=str,default="inputs",help="Input column of csv")
    parser.add_argument('--output_column',type=str,default="targets",help="Target column of csv")
    parser.add_argument('--attn_implementation', type=str, default="eager", help="Attention implementation to use")
    args = parser.parse_args()
    return args

def setup_logging(log_file_path):
    """Set up the logging configuration."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    if os.path.exists(log_file_path):
        open(log_file_path, 'w').close()

    # Set up logging
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(level=logging.INFO, handlers=[handler, console_handler])

def load_data(path: str,args):
    df = pd.read_csv(path)[[args.input_column, args.output_column]].head(1000)
    df = df.dropna()
    return Dataset.from_pandas(df)

def count_parameters(model):
    """
    Count the number of trainable (unfrozen) and non-trainable (frozen) parameters in the model.

    Args:
    model (torch.nn.Module): The model to count parameters for.

    Returns:
    None
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Frozen parameters: {frozen_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Frozen parameters: {frozen_params}")

def format_chat_template(row, tokenizer):
    row_json = [{"role": "user", "content": row["inputs"]},
                {"role": "assistant", "content": row["targets"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

def load_model(config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map={"": torch.device(config.device)},
        attn_implementation=config.attn_implementation
    )
    model, tokenizer = setup_chat_format(model, tokenizer)
    return tokenizer, model

def get_training_args(config):
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_accumulation_steps=config.gradient_accumulation,
        optim=config.optim,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        disable_tqdm=config.disable_tqdm,
        deepspeed="deepspeed.json"
    )

def get_peft_config(config):
    return LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type,
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

def peft_model(model, peft_config):
    return get_peft_model(
        prepare_model_for_kbit_training(model),
        peft_config,
    )

if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(args.logger_file)

    # Log hyperparameters and settings
    logging.info(f"Arguments: {args}")

    trainset = load_data(args.dataset_path,args)
    valset = load_data(args.dataset_path.replace('train', 'test'),args)



    tokenizer, model = load_model(args)
    model = peft_model(model, get_peft_config(args))

    trainset = trainset.map(lambda row: format_chat_template(row, tokenizer), num_proc=4)
    valset = valset.map(lambda row: format_chat_template(row, tokenizer), num_proc=4)

    count_parameters(model)

    training_args = get_training_args(args)
    trainer = SFTTrainer(
        max_seq_length=args.max_seq_length,
        model=model,
        train_dataset=trainset,
        eval_dataset=valset,
        peft_config=get_peft_config(args),
        tokenizer=tokenizer,
        packing=True,
        args=training_args,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model()
    print(torch.cuda.current_device())

    
