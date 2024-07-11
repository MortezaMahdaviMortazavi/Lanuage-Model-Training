# Language-Model-Training

In this repository, I provide scripts for training different language models with various types such as Masked Modeling and Causal modeling using PyTorch and HuggingFace. 

## Repository Structure


### Folders and Files

- **LLMCausal/**: Contains scripts for fine-tuning language models using Causal modeling.
  - `train_llama.bash`: A bash script for training Llama models.
  - `train.py`: Python script for fine-tuning using a standard approach.
  - `train_lightning.py`: Python script for fine-tuning using PyTorch Lightning.

- **masked_modeling/**: Contains scripts for training models using masked language modeling.
  - `script1.py`: Description of what this script does.
  - `script2.py`: Description of what this script does.

- **README.md**: This file.
- **bash.sh**: A bash script for running tasks (you may describe its purpose).

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.6+
- PyTorch
- HuggingFace Transformers
- PyTorch Lightning (if using `train_lightning.py`)
- Other dependencies listed in `requirements.txt` (if provided)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Language-Model-Training.git
cd Language-Model-Training


pip install -r requirements.txt
