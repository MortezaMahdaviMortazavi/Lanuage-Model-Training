import argparse
import torch
import pandas as pd
import math
import logging
import os

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, get_scheduler, default_data_collator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm

"""
python train.py \
  --data_path combined_datasets/cleaned_sentences.csv \
  --target_column text \
  --model_name sbunlp/fabert \
  --output_dir MLP_TrainedModels \
  --logger_file logs/training.log \
  --mlm_probability 0.15 \
  --batch_size 64 \
  --chunk_size 512 \
  --num_train_epochs 30 \
  --learning_rate 5e-5 \
  --random_seed 42 \
  --print_per_batch_num 100 \
  --train_size 0.95 \
  --test_size 0.05


"""



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Masked Language Model")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data CSV file")
    parser.add_argument('--target_column', type=str,default="text",required=False, help="Name of the target column in the data CSV file")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the pretrained model")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--logger_file",type=str , required=True ,help="Log everything in the training")
    parser.add_argument('--mlm_probability', type=float, default=0.2, help="Probability for masking tokens in MLM")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument('--chunk_size', type=int, default=128, help="Chunk size for grouping texts")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for the optimizer")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--print_per_batch_num",type=int,default=100,help="Print loss and track training after this number of batch")
    parser.add_argument('--train_size', type=float, default=0.9, help="Train size rate")
    parser.add_argument('--test_size', type=float, default=0.1, help="Test size rate")
    return parser.parse_args()

def setup_logging(log_file_path):
    """Set up the logging configuration."""
    os.makedirs(os.path.dirname("./" + log_file_path), exist_ok=True)

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


def load_and_prepare_data(path: str, target_column: str = None):
    """
    Load and prepare the dataset from a CSV or text file.
    
    Args:
    path (str): Path to the data file (CSV or text).
    target_column (str): Name of the target column in the data CSV file (ignored for text files).

    Returns:
    dataset (Dataset): HuggingFace Dataset object.
    """
    logging.info(f"Loading data from {path}")

    # Determine the file extension
    _, file_extension = os.path.splitext(path)

    if file_extension.lower() == '.csv':
        logging.info(f"Detected CSV file format. Targeting column '{target_column}'")
        df = pd.read_csv(path)
        original_length = len(df)
        df = df[df[target_column].apply(lambda x: isinstance(x, str) and len(x) >= 5 and x.strip() != "")]
        logging.info(f"Filtered data from {original_length} to {len(df)} entries")
        dataset = Dataset.from_pandas(df)


    elif file_extension.lower() == '.txt':
        logging.info(f"Detected text file format.")
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines if len(line.strip()) >= 5]
        dataset = Dataset.from_dict({"text": lines})
        logging.info(f"Loaded text dataset with {len(lines)} entries")
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or text file.")
    
    return dataset

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
        "masked_token_type_ids":"token_type_ids"
    })
    print(len(downsampled_dataset["train"]), len(downsampled_dataset["test"]))

    train_dataloader = DataLoader(downsampled_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=default_data_collator)
    
    return train_dataloader, eval_dataloader

def train_model(
        model: AutoModelForMaskedLM, 
        train_dataloader: DataLoader, 
        eval_dataloader: DataLoader,
        batch_size: int, 
        num_train_epochs: int,
        print_per_batch_num: int, 
        learning_rate: float, 
        output_dir: str):
    """
    Train the masked language model.
    
    Args:
    model (AutoModelForMaskedLM): Pretrained model for masked language modeling.
    train_dataloader (DataLoader): Data loader for training.
    eval_dataloader (DataLoader): Data loader for evaluation.
    batch_size (int): Batch size.
    num_train_epochs (int): Number of training epochs.
    print_per_batch_num (int): Print loss every specified number of batches.
    learning_rate (float): Learning rate for the optimizer.
    output_dir (str): Directory to save the trained model.

    Returns:
    None
    """
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
    
    num_update_steps_per_epoch = len(train_dataloader)
    

    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    logging.info(f"num_training_steps: {num_training_steps}")
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    logging.info("Starting training process...")

    for epoch in range(num_train_epochs):
        model.train()
        train_loss = 0.0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            train_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if idx % print_per_batch_num == 0:
                print(f"Batch {idx}, Loss: {train_loss / (idx + 1)}")

        epoch_train_result = f"Epoch {epoch} Train Loss: {train_loss / len(train_dataloader)}"
        print(epoch_train_result)
        logging.info(epoch_train_result)

        model.eval()
        losses = []
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))

        losses = torch.cat(losses)[:len(eval_dataloader.dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        epoch_result = f">>> Epoch {epoch}: Perplexity: {perplexity}"
        print(epoch)
        logging.info(epoch_result)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

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
    train_model(
        model, 
        train_dataloader, 
        eval_dataloader, 
        batch_size=args.batch_size, 
        num_train_epochs=args.num_train_epochs, 
        print_per_batch_num=args.print_per_batch_num,  # You can change this as needed
        learning_rate=args.learning_rate, 
        output_dir=args.output_dir
    )