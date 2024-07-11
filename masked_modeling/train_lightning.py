import argparse
import torch
import pandas as pd
import math
import logging
import os
import pytorch_lightning as pl

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, get_scheduler, default_data_collator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm

"""
python train_lightning.py --data_path combined_datasets/cleaned_sentences.csv --logger_file logs/training.log --target_column text --model_name sbunlp/fabert --output_dir MLP_TrainedModels --mlm_probability 0.2 --batch_size 4 --chunk_size 256 --num_train_epochs 30 --learning_rate 5e-5 --random_seed 42 --print_per_batch_num 100 --train_size 0.9 --test_size 0.1 --accelerator gpu --train_strategy deepspeed_stage_2

"""

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Masked Language Model")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data CSV file")
    parser.add_argument('--target_column', type=str, required=False, help="Name of the target column in the data CSV file")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the pretrained model")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--logger_file", type=str, required=True, help="Log everything in the training")
    parser.add_argument('--mlm_probability', type=float, default=0.2, help="Probability for masking tokens in MLM")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument('--chunk_size', type=int, default=128, help="Chunk size for grouping texts")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for the optimizer")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--print_per_batch_num", type=int, default=100, help="Print loss and track training after this number of batch")
    parser.add_argument('--train_size', type=float, default=0.9, help="Train size rate")
    parser.add_argument('--test_size', type=float, default=0.1, help="Test size rate")
    parser.add_argument('--accelerator',type=str,default='cpu',help="Hardware type for training model : gpu , cpu , tpu")
    parser.add_argument('--train_strategy',type=str,default='auto',help="Hardware strategy for training model")
    return parser.parse_args()

def setup_logging(log_file_path):
    """Set up the logging configuration."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    if os.path.exists(log_file_path):
        open(log_file_path, 'w').close()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

def load_csv_data(path: str, target_column: str) -> Dataset:
    """Load and prepare dataset from a CSV file."""
    logging.info(f"Detected CSV file format. Targeting column '{target_column}'")
    df = pd.read_csv(path)
    original_length = len(df)
    df = filter_csv_data(df, target_column)
    logging.info(f"Filtered data from {original_length} to {len(df)} entries")
    return Dataset.from_pandas(df)

def filter_csv_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Filter the CSV data based on the target column."""
    return df[df[target_column].apply(lambda x: isinstance(x, str) and len(x) >= 5 and x.strip() != "")]

def load_text_data(path: str) -> Dataset:
    """Load and prepare dataset from a text file."""
    logging.info(f"Detected text file format.")
    lines = read_text_file(path)
    lines = filter_text_data(lines)
    logging.info(f"Loaded text dataset with {len(lines)} entries")
    return Dataset.from_dict({"text": lines})

def read_text_file(path: str) -> list:
    """Read lines from a text file."""
    with open(path, 'r', encoding='utf-8') as file:
        return file.readlines()

def filter_text_data(lines: list) -> list:
    """Filter the text data."""
    return [line.strip() for line in lines if len(line.strip()) >= 5]

def load_and_prepare_data(path: str, target_column: str = None) -> Dataset:
    """
    Load and prepare the dataset from a CSV or text file.
    
    Args:
    path (str): Path to the data file (CSV or text).
    target_column (str): Name of the target column in the data CSV file (ignored for text files).

    Returns:
    dataset (Dataset): HuggingFace Dataset object.
    """
    logging.info(f"Loading data from {path}")

    try:
        _, file_extension = os.path.splitext(path)
        
        if file_extension.lower() == '.csv':
            if not target_column:
                raise ValueError("Target column must be specified for CSV files.")
            return load_csv_data(path, target_column)
        
        elif file_extension.lower() == '.txt':
            return load_text_data(path)
        
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or text file.")
    
    except FileNotFoundError:
        logging.error(f"The file {path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"The file {path} is empty.")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

def load_tokenizer(model_name: str = "sbunlp/fabert"):
    """
    Load the tokenizer.
    
    Args:
    model_name (str): Name of the pretrained model.

    Returns:
    tokenizer: Pretrained tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)

def load_model_and_tokenizer(model_name: str):
    """
    Load the model and tokenizer.
    
    Args:
    model_name (str): Name of the pretrained model.

    Returns:
    tokenizer, model: Pretrained tokenizer and model.
    """
    logging.info(f"Loading tokenizer for model {model_name}")
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

def tokenize_function(data, tokenizer):
    """
    Tokenize the data.
    
    Args:
    data: Data to be tokenized.
    tokenizer: Pretrained tokenizer.

    Returns:
    Tokenized data.
    """
    return tokenizer(data["text"])

def group_texts(data, chunk_size: int):
    """
    Group texts into chunks of a specified size.
    
    Args:
    data: Data to be grouped.
    chunk_size (int): Size of each chunk.

    Returns:
    Grouped texts with labels.
    """
    concatenated_sequences = {k: sum(data[k], []) for k in data.keys()}
    total_concat_length = len(concatenated_sequences[list(data.keys())[0]])
    total_length = (total_concat_length // chunk_size) * chunk_size
    result = {k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)] for k, t in concatenated_sequences.items()}
    result["labels"] = result["input_ids"].copy()
    return result

def insert_random_mask(batch, data_collator):
    """
    Insert random masks into the data.
    
    Args:
    batch: Batch of data.
    data_collator: Data collator for language modeling.

    Returns:
    Data with random masks.
    """
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

def prepare_dataloader(dataset, tokenizer, batch_size: int, mlm_probability: float, train_size: int, test_size: int, chunk_size: int, seed: int):
    """
    Prepare the data loader.
    
    Args:
    dataset: Dataset object.
    tokenizer: Pretrained tokenizer.
    batch_size (int): Batch size.
    mlm_probability (float): Probability for masking tokens in MLM.
    train_size (int): Size of the training dataset.
    test_size (int): Size of the test dataset.
    chunk_size (int): Size of each chunk.
    seed (int): Random seed for reproducibility.

    Returns:
    train_dataloader, eval_dataloader: Data loaders for training and evaluation.
    """
    logging.info(f"Grouping texts into chunks of size {chunk_size}")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

    tokenize_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=list(dataset.column_names))
    processed_dataset = tokenize_dataset.map(lambda x: group_texts(x, chunk_size=chunk_size), batched=True)

    downsampled_dataset = processed_dataset.train_test_split(train_size=train_size, test_size=test_size, seed=seed)

    eval_dataset = downsampled_dataset["test"].map(
        lambda x: insert_random_mask(x, data_collator),
        batched=True,
        remove_columns=downsampled_dataset["test"].column_names,
    )
    eval_dataset = eval_dataset.rename_columns({
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
        "masked_token_type_ids": "token_type_ids"
    })
    print(len(downsampled_dataset["train"]), len(downsampled_dataset["test"]))

    train_dataloader = DataLoader(downsampled_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=default_data_collator)
    
    return train_dataloader, eval_dataloader

class LightingTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.step_count = 0

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.step_count += 1
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, logger=True)

        if batch_idx % 200 == 0:
            self.validation_perplexity()

        return loss

    def validation_perplexity(self):
        self.model.eval()
        losses = []
        val_dataloader = self.val_dataloader()
        for batch in val_dataloader:
            with torch.no_grad():
                outputs = self.forward(batch)
                loss = outputs.loss
                losses.append(loss)

        avg_loss = torch.mean(torch.tensor(losses))
        perplexity = torch.exp(avg_loss)

        result = f"Step {self.step_count}: Validation Perplexity: {perplexity.item()}"
        logging.info(result)
        print(result)
        self.model.train()
        return perplexity

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        num_update_steps_per_epoch = len(self.train_dataloader())
        num_training_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    @staticmethod
    def load_weights(model, checkpoint_path):
        weights = torch.load(checkpoint_path)
        new_weights = {}
        for key, value in weights.items():
            new_key = key.replace('model.', '')  # Remove 'model' prefix
            new_weights[new_key] = value
        del weights
        model.load_state_dict(new_weights)
        print("Loading weights was successful")

if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(args.logger_file)
    logging.info(f"Arguments: {args}")
    dataset = load_and_prepare_data(path=args.data_path, target_column=args.target_column)
    tokenizer, model = load_model_and_tokenizer(args.model_name)
    train_dataloader, eval_dataloader = prepare_dataloader(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_probability,
        train_size=args.train_size,
        test_size=args.test_size,
        chunk_size=args.chunk_size,
        seed=args.random_seed
    )
    lightning_module = LightingTrainer(model, args)
    lightning_module._train_dataloader = train_dataloader
    lightning_module._val_dataloader = eval_dataloader
    trainer = pl.Trainer(max_epochs=args.num_train_epochs,accelerator=args.accelerator,strategy=args.train_strategy)
    trainer.fit(lightning_module)
    lightning_module.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
