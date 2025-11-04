# MLOps Lab 4 - Data Pipeline

## Overview
Improved data preprocessing pipeline for language model training, demonstrating MLOps best practices.

## Repository
**GitHub**: [https://github.com/msalmancodes/mlops_lab4](https://github.com/msalmancodes/mlops_lab4)

## Key Improvements Over Original Lab
- ✅ **Different Tokenizer**: DistilBERT-base-uncased (vs original GPT-2)
- ✅ **Different Dataset**: AG News Classification (vs original WikiText-2)
- ✅ **Configuration Management**: Centralized config dictionary with timestamps
- ✅ **Train/Val Split**: Proper data splitting for model evaluation
- ✅ **Data Validation**: Filtering empty sequences and quality checks
- ✅ **Logging & Monitoring**: Comprehensive dataset statistics tracking
- ✅ **Reproducibility**: Seed setting & artifact saving
- ✅ **Dynamic Padding**: Efficient batch processing with variable lengths
- ✅ **Larger Parameters**: Increased block size (256) and batch size (16)
- ✅ **Artifact Management**: Saves config, tokenizer, and statistics

## Setup

### Prerequisites
- Python 3.8+
- pip
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/msalmancodes/mlops_lab4.git
cd mlops_lab4
```

2. Create virtual environment:
```bash
python -m venv venv

# Activate:
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run the complete data pipeline:
```bash
python lab1_improved.py
```

### Quick Test (Limited Dataset)
To test with a smaller dataset (10k samples):
```python
# Edit lab1_improved.py line 24:
"max_samples": 10000
```

## Pipeline Results

The pipeline processes the full AG News dataset:

### Dataset Size
- **Training samples**: 120,000 news articles
- **Validation samples**: 7,600 news articles
- **Categories**: 4 (World, Sports, Business, Sci/Tech)

### Processed Output
- **Train blocks**: 24,853 (256-token sequences)
- **Val blocks**: 1,562 (256-token sequences)
- **Total train tokens**: 6,362,368
- **Total val tokens**: 399,872
- **Vocabulary size**: 30,522 (DistilBERT vocab)
- **Batches per epoch**: 1,554

## Output Artifacts

All artifacts saved to `output/` directory:

### 1. `training_config.json`
```json
{
  "model_name": "distilbert-base-uncased",
  "dataset_name": "ag_news",
  "block_size": 256,
  "batch_size": 16,
  "seed": 42,
  "timestamp": "2025-11-03T22:29:15.996703"
}
```

### 2. `dataset_stats.json`
```json
{
  "train_samples": 24853,
  "val_samples": 1562,
  "train_tokens": 6362368,
  "val_tokens": 399872,
  "vocab_size": 30522,
  "batches_per_epoch": 1554
}
```

### 3. `tokenizer/`
Saved DistilBERT tokenizer for reproducibility and deployment.

## Configuration Options

Edit the `config` dictionary in `lab1_improved.py`:

```python
config = {
    "model_name": "distilbert-base-uncased",  # Tokenizer model
    "dataset_name": "ag_news",                 # HuggingFace dataset
    "block_size": 256,                         # Token sequence length
    "batch_size": 16,                          # Batch size
    "seed": 42,                                # Random seed
    "max_samples": None,                       # Limit dataset (None = full)
    "timestamp": datetime.now().isoformat()    # Auto-generated
}
```

## Technical Stack

- **transformers** (4.36.0) - HuggingFace transformers library
- **datasets** (2.16.0) - Dataset loading and processing
- **torch** (2.1.0) - PyTorch for data loading
- **accelerate** (0.25.0) - Training optimization
- **numpy** (1.24.3) - Numerical operations

## Project Structure

```
mlops_lab4/
├── venv/                      # Virtual environment (not in git)
├── requirements.txt           # Python dependencies
├── lab1_improved.py          # Main pipeline script
├── .gitignore                # Git ignore rules
├── README.md                 # This file
└── output/                    # Generated artifacts
    ├── training_config.json   # Run configuration
    ├── dataset_stats.json     # Dataset statistics
    └── tokenizer/             # Saved tokenizer
```

## MLOps Principles Demonstrated

1. **Version Control**: Git for code versioning
2. **Dependency Management**: requirements.txt for reproducibility
3. **Configuration as Code**: Centralized config dictionary
4. **Artifact Tracking**: Saving configs, stats, and models
5. **Data Validation**: Quality checks and filtering
6. **Logging**: Comprehensive statistics and progress tracking
7. **Reproducibility**: Seed setting and deterministic processing
8. **Modularity**: Functions for tokenization, grouping, collation
9. **Environment Isolation**: Virtual environment usage
10. **Documentation**: Clear README with setup instructions

## Comparison with Original Lab

| Feature | Original | Improved |
|---------|----------|----------|
| Tokenizer | GPT-2 | DistilBERT |
| Dataset | WikiText-2 (36k lines) | AG News (120k samples) |
| Block Size | 128 | 256 |
| Batch Size | 8 | 16 |
| Config Management | Hardcoded | Dictionary-based |
| Data Validation | None | Filtering + stats |
| Artifact Saving | None | Config + tokenizer + stats |
| Train/Val Split | Train only | Both splits |
| Documentation | Minimal | Comprehensive |

## Runtime Performance

- **First run**: ~5-8 minutes (includes downloads)
- **Subsequent runs**: ~3-5 minutes (cached data)
- **Quick test mode**: ~2-3 minutes (10k samples)

## Author
 
GitHub: [@msalmancodes](https://github.com/msalmancodes)

## License

MIT License - See LICENSE file for details

## Acknowledgments

- HuggingFace for transformers and datasets libraries
- Original lab materials from MLOps course
- AG News dataset creators (Zhang et al.)