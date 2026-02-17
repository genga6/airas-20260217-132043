"""Inference script for prompt-based reasoning experiments."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from tqdm import tqdm

from src.preprocess import load_dataset


def extract_answer(text: str) -> str:
    """Extract numerical answer from model output.
    
    Looks for patterns like:
    - #### 123
    - The answer is 123
    - Final answer: 123
    """
    # Try to find #### pattern (common in GSM8K)
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Try to find "answer is" pattern
    match = re.search(r'(?:answer is|answer:|final answer is|final answer:)\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    # Try to find last number in the text
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return ""


def run_inference(cfg: DictConfig) -> Dict[str, Any]:
    """Run inference on the dataset with the specified prompt method."""
    
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(f"WandB run initialized: {wandb_run.get_url()}")
    else:
        wandb_run = None
        print("WandB disabled")
    
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Model loading failed with 401 Unauthorized error for gated model google/gemma-3-4b-it
    # [CAUSE]: The model requires authentication but the code didn't pass the HF_TOKEN to from_pretrained()
    # [FIX]: Added token parameter to both model and tokenizer loading, fetching from HF_TOKEN environment variable
    #
    # [OLD CODE]:
    # model = AutoModelForCausalLM.from_pretrained(
    #     f"google/{cfg.run.model.name}",
    #     cache_dir=cfg.cache_dir,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     f"google/{cfg.run.model.name}",
    #     cache_dir=cfg.cache_dir,
    # )
    #
    # [NEW CODE]:
    # Load model and tokenizer
    print(f"Loading model: {cfg.run.model.name}")
    hf_token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        f"google/{cfg.run.model.name}",
        cache_dir=cfg.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        f"google/{cfg.run.model.name}",
        cache_dir=cfg.cache_dir,
        token=hf_token,
    )
    print("Model loaded successfully")
    
    # Load dataset
    print(f"Loading dataset: {cfg.run.dataset.name}")
    dataset = load_dataset(cfg)
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Prepare results storage
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = []
    correct = 0
    total = 0
    
    # Run inference
    print("\nRunning inference...")
    for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        question = sample["question"]
        ground_truth = sample["answer"]
        
        # Format prompt with the method's template
        prompt = cfg.run.method.prompt_template.format(problem=question)
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.run.model.max_new_tokens,
                temperature=cfg.run.model.temperature,
                top_p=cfg.run.model.top_p,
                do_sample=cfg.run.model.do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        response_only = response[len(prompt):].strip()
        
        # Extract answer
        predicted_answer = extract_answer(response_only)
        
        # Extract ground truth answer
        gt_answer = extract_answer(ground_truth)
        
        # Check correctness
        is_correct = predicted_answer == gt_answer
        if is_correct:
            correct += 1
        total += 1
        
        # Store prediction
        predictions.append({
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "ground_truth_answer": gt_answer,
            "response": response_only,
            "predicted_answer": predicted_answer,
            "correct": is_correct,
        })
        
        # Log to WandB
        if wandb_run and i % 10 == 0:
            wandb.log({
                "samples_processed": i + 1,
                "accuracy": correct / total,
            })
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "method": cfg.run.method.name,
        "model": cfg.run.model.name,
        "dataset": cfg.run.dataset.name,
    }
    
    print(f"\n{'=' * 80}")
    print(f"Inference complete!")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'=' * 80}")
    
    # Save results
    predictions_file = results_dir / "predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to: {predictions_file}")
    
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
    # Log final metrics to WandB
    if wandb_run:
        wandb.log(metrics)
        wandb.summary.update(metrics)
        wandb.finish()
    
    # Sanity validation for inference tasks
    if cfg.mode == "sanity_check":
        print("\n" + "=" * 80)
        print("SANITY CHECK VALIDATION")
        print("=" * 80)
        
        # Check if at least 5 samples processed
        samples_ok = total >= 5
        
        # Check if outputs are valid (not all identical)
        unique_predictions = len(set(p["predicted_answer"] for p in predictions))
        outputs_valid = unique_predictions > 1 or total < 2
        
        # Check if accuracy is reasonable (not 0 unless it's a very hard task)
        accuracy_ok = accuracy > 0 or total < 5
        
        validation_summary = {
            "samples": total,
            "outputs_valid": outputs_valid,
            "outputs_unique": unique_predictions,
            "accuracy": accuracy,
        }
        
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(validation_summary)}")
        
        if samples_ok and outputs_valid:
            print("SANITY_VALIDATION: PASS")
        else:
            reasons = []
            if not samples_ok:
                reasons.append(f"insufficient_samples({total}<5)")
            if not outputs_valid:
                reasons.append("all_outputs_identical")
            print(f"SANITY_VALIDATION: FAIL reason={','.join(reasons)}")
    
    return metrics


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for inference."""
    run_inference(cfg)


if __name__ == "__main__":
    main()
