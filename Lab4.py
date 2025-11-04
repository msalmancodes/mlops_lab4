"""
MLOps Lab 1 - Improved Data Pipeline
Changes: Different tokenizer (DistilBERT), different dataset (AG News), 
config management, train/val split, data validation, logging
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import json
from datetime import datetime
import numpy as np
import os


def main():
    # ============ 1. CONFIGURATION MANAGEMENT ============
    config = {
        "model_name": "distilbert-base-uncased",  # Changed from GPT-2
        "dataset_name": "ag_news",  # Changed from WikiText
        "block_size": 256,  # Increased block size
        "batch_size": 16,  # Increased batch size
        "seed": 42,
        "max_samples": None,  # Set to int to limit dataset size for faster testing
        "timestamp": datetime.now().isoformat()
    }

    print("="*60)
    print("MLOps Lab 1 - Data Pipeline")
    print("="*60)
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))

    # Set seed for reproducibility
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # ============ 2. LOAD TRAIN & VALIDATION SPLITS ============
    print("\n=== Loading Datasets ===")
    train_dataset = load_dataset(config["dataset_name"], split="train")
    val_dataset = load_dataset(config["dataset_name"], split="test")

    # Limit dataset size if specified (useful for quick testing)
    if config["max_samples"]:
        train_dataset = train_dataset.select(range(min(config["max_samples"], len(train_dataset))))
        val_dataset = val_dataset.select(range(min(config["max_samples"]//4, len(val_dataset))))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Sample data: {train_dataset[0]}")

    # ============ 3. TOKENIZATION WITH DATA VALIDATION ============
    print("\n=== Initializing Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # DistilBERT already has pad_token, but ensure it's set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True,
            max_length=config["block_size"],
            return_special_tokens_mask=False
        )

    print("\n=== Tokenizing ===")
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=val_dataset.column_names
    )

    # Filter out empty sequences (important data quality step!)
    tokenized_train = tokenized_train.filter(lambda x: len(x["input_ids"]) > 0)
    tokenized_val = tokenized_val.filter(lambda x: len(x["input_ids"]) > 0)

    print(f"Train sequences after filtering: {len(tokenized_train)}")
    print(f"Val sequences after filtering: {len(tokenized_val)}")

    # Log tokenization statistics
    train_lengths = [len(x["input_ids"]) for x in tokenized_train.select(range(min(1000, len(tokenized_train))))]
    print(f"Avg token length (sample): {np.mean(train_lengths):.1f}")
    print(f"Max token length (sample): {max(train_lengths)}")

    # ============ 4. GROUP INTO FIXED-LENGTH BLOCKS ============
    def group_texts(examples):
        concatenated_inputs = sum(examples["input_ids"], [])
        concatenated_masks = sum(examples["attention_mask"], [])
        
        total_len = (len(concatenated_inputs) // config["block_size"]) * config["block_size"]
        concatenated_inputs = concatenated_inputs[:total_len]
        concatenated_masks = concatenated_masks[:total_len]
        
        result_input_ids = [concatenated_inputs[i:i+config["block_size"]] 
                            for i in range(0, total_len, config["block_size"])]
        result_masks = [concatenated_masks[i:i+config["block_size"]] 
                        for i in range(0, total_len, config["block_size"])]
        
        return {"input_ids": result_input_ids, "attention_mask": result_masks}

    print("\n=== Grouping into blocks ===")
    lm_train = tokenized_train.map(group_texts, batched=True, batch_size=1000)
    lm_val = tokenized_val.map(group_texts, batched=True, batch_size=1000)

    print(f"Train blocks: {len(lm_train)}")
    print(f"Val blocks: {len(lm_val)}")

    # ============ 5. DATA STATISTICS (LOGGING) ============
    print("\n=== Data Statistics ===")
    train_tokens = len(lm_train) * config["block_size"]
    val_tokens = len(lm_val) * config["block_size"]
    print(f"Total train tokens: {train_tokens:,}")
    print(f"Total val tokens: {val_tokens:,}")
    print(f"Train batches per epoch: {len(lm_train) // config['batch_size']}")

    # ============ 6. CREATE DATALOADERS WITH DYNAMIC PADDING ============
    def collate_fn(batch):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
        attention_mask = [torch.tensor(example["attention_mask"], dtype=torch.long) for example in batch]
        
        # Pad sequences to max length in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

    train_loader = DataLoader(
        lm_train, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        lm_val, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )

    # ============ 7. VALIDATION CHECK ============
    print("\n=== DataLoader Validation ===")
    for batch in train_loader:
        print(f"Train batch shape: {batch['input_ids'].shape}")
        print(f"Train attention mask shape: {batch['attention_mask'].shape}")
        assert batch['input_ids'].shape[1] <= config['block_size'], "Block size exceeded!"
        assert batch['input_ids'].shape == batch['attention_mask'].shape, "Shape mismatch!"
        print(f"Sample token IDs: {batch['input_ids'][0][:10].tolist()}")
        break

    for batch in val_loader:
        print(f"\nVal batch shape: {batch['input_ids'].shape}")
        break

    # ============ 8. SAVE ARTIFACTS FOR REPRODUCIBILITY ============
    print("\n=== Saving Artifacts ===")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Save config
    with open('output/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Config saved to output/training_config.json")

    # Save tokenizer
    tokenizer.save_pretrained("output/tokenizer")
    print("✓ Tokenizer saved to output/tokenizer")

    # Save dataset statistics
    stats = {
        "train_samples": len(lm_train),
        "val_samples": len(lm_val),
        "train_tokens": len(lm_train) * config["block_size"],
        "val_tokens": len(lm_val) * config["block_size"],
        "vocab_size": tokenizer.vocab_size,
        "batches_per_epoch": len(train_loader)
    }

    with open('output/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("✓ Dataset stats saved to output/dataset_stats.json")

    print("\n" + "="*60)
    print("✓ Pipeline complete! Ready for training.")
    print("="*60)
    print(f"\nDataset: {config['dataset_name']}")
    print(f"Tokenizer: {config['model_name']}")
    print(f"Train samples: {len(lm_train):,}")
    print(f"Val samples: {len(lm_val):,}")
    print(f"Batches per epoch: {len(train_loader):,}")
    
    return train_loader, val_loader, config


if __name__ == "__main__":
    train_loader, val_loader, config = main()
    print("\n✓ All artifacts saved to 'output/' directory")