import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
import math
import argparse
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def parse_arguments():
    """Parse command line arguments for testing parameters."""
    parser = argparse.ArgumentParser(description="Test a Masked Language Model")
    parser.add_argument('--model_name', type=str, required=True, help="Path to the pretrained model or model identifier from Hugging Face's model hub")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the test dataset file")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for testing")
    return parser.parse_args()

def load_model_and_tokenizer(model_name_or_path):
    """Load the tokenizer and model from a path or model hub."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    return tokenizer, model

def prepare_data(data_file, tokenizer, block_size=512):
    """Prepare data for testing by tokenizing and formatting appropriately."""
    dataset = load_dataset('text', data_files={'test': data_file})
    def encode(examples):
        return tokenizer(examples['text'], add_special_tokens=True, truncation=True, max_length=block_size, padding='max_length')
    
    dataset = dataset.map(encode, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset['test']

def calculate_perplexity(model, dataloader):
    """Calculate the perplexity of the model on the test data."""
    model.eval()
    total_loss = 0
    total_length = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']}
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item() * batch['input_ids'].size(0)
            total_length += batch['input_ids'].size(0)

    average_loss = total_loss / total_length
    perplexity = math.exp(average_loss)
    return perplexity

def main():
    args = parse_arguments()
    tokenizer, model = load_model_and_tokenizer(args.model_name)
    test_dataset = prepare_data(args.data_file, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    perplexity = calculate_perplexity(model, test_dataloader)
    print(f"Perplexity of the model: {perplexity}")

if __name__ == "__main__":
    main()