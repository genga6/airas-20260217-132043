"""Dataset loading and preprocessing utilities."""

from typing import List, Dict, Any
from datasets import load_dataset as hf_load_dataset
from omegaconf import DictConfig


def load_dataset(cfg: DictConfig) -> List[Dict[str, Any]]:
    """Load and preprocess the GSM8K dataset.
    
    Args:
        cfg: Configuration object with dataset settings
        
    Returns:
        List of samples with 'question' and 'answer' fields
    """
    dataset_name = cfg.run.dataset.name
    split = cfg.run.dataset.split
    max_samples = cfg.run.dataset.max_samples
    
    print(f"Loading dataset: {dataset_name}, split: {split}")
    
    # Load GSM8K dataset
    if dataset_name == "gsm8k":
        # GSM8K is available via HuggingFace datasets
        dataset = hf_load_dataset(
            "openai/gsm8k",
            "main",
            split=split,
            cache_dir=cfg.cache_dir,
        )
        
        # Convert to list of dicts
        samples = []
        for item in dataset:
            samples.append({
                "question": item["question"],
                "answer": item["answer"],
            })
        
        # Limit samples if specified
        if max_samples is not None and max_samples > 0:
            samples = samples[:max_samples]
        
        print(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
