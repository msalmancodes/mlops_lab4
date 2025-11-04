# MLOps Lab 1 - Data Pipeline

## Overview
Improved data preprocessing pipeline for language model training, demonstrating MLOps best practices.

## Key Improvements
- ✅ **Different Tokenizer**: DistilBERT (vs original GPT-2)
- ✅ **Different Dataset**: AG News (vs original WikiText)
- ✅ **Configuration Management**: Centralized config dictionary
- ✅ **Train/Val Split**: Proper data splitting
- ✅ **Data Validation**: Filtering empty sequences
- ✅ **Logging & Monitoring**: Dataset statistics tracking
- ✅ **Reproducibility**: Seed setting & artifact saving
- ✅ **Dynamic Padding**: Efficient batch processing

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd mlops_lab1
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the data pipeline:
```bash
python lab1_improved.py
```

## Results

The pipeline processes:
- **120,000** training samples (AG News)
- **7,600** validation samples
- **~6.4M** training tokens
- **~400K** validation tokens

Output artifacts saved to `output/`:
- `training_config.json` - Configuration used
- `dataset_stats.json` - Dataset statistics
- `tokenizer/` - Saved tokenizer

## Configuration

Edit `config` dictionary in `lab1_improved.py`:
```python
config = {
    "model_name": "distilbert-base-uncased",
    "dataset_name": "ag_news",
    "block_size": 256,
    "batch_size": 16,
    "seed": 42,
    "max_samples": None  # Set to int for quick testing
}
```

## Dataset Statistics

From `output/dataset_stats.json`:
- Train samples: 24,853 blocks
- Val samples: 1,562 blocks
- Vocabulary size: 30,522 tokens
- Batches per epoch: 1,554

## Technologies Used
- **transformers** - Tokenizer and model utilities
- **datasets** - HuggingFace datasets library
- **torch** - PyTorch for data loading
- **numpy** - Numerical operations

## Author
[Your Name]

## License
MIT
