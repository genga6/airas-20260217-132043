"""Evaluation script for comparing multiple runs and generating comparison visualizations."""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import wandb


def load_wandb_config():
    """Load WandB configuration from environment or config file."""
    import os
    from omegaconf import OmegaConf
    
    # Try to load from config file
    config_path = Path("config/config.yaml")
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        return cfg.wandb.entity, cfg.wandb.project
    
    # Fallback to environment variables
    entity = os.environ.get("WANDB_ENTITY", "airas")
    project = os.environ.get("WANDB_PROJECT", "2026-02-17")
    return entity, project


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """Fetch run data from WandB API.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run ID
        
    Returns:
        Dictionary with run config, summary, and history
    """
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get run summary (final metrics)
        summary = dict(run.summary)
        
        # Get run config
        config = dict(run.config)
        
        # Get run history (all logged metrics)
        history = run.history()
        
        return {
            "run_id": run_id,
            "config": config,
            "summary": summary,
            "history": history.to_dict('records') if not history.empty else [],
        }
    except Exception as e:
        print(f"Warning: Could not fetch run {run_id} from WandB: {e}")
        return None


def export_run_metrics(run_data: Dict[str, Any], results_dir: Path):
    """Export per-run metrics to JSON file.
    
    Args:
        run_data: Run data fetched from WandB
        results_dir: Directory to save metrics
    """
    if not run_data:
        return
    
    run_id = run_data["run_id"]
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Export metrics
    metrics_file = run_dir / "metrics.json"
    metrics = {
        "run_id": run_id,
        "summary": run_data["summary"],
        "config": run_data["config"],
    }
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics for {run_id} to {metrics_file}")


def create_per_run_figures(run_data: Dict[str, Any], results_dir: Path):
    """Create per-run figures.
    
    Args:
        run_data: Run data fetched from WandB
        results_dir: Directory to save figures
    """
    if not run_data or not run_data["history"]:
        return
    
    run_id = run_data["run_id"]
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    history = run_data["history"]
    
    # Plot accuracy over time if available
    if any("accuracy" in h for h in history):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = [h.get("samples_processed", i) for i, h in enumerate(history) if "accuracy" in h]
        accuracies = [h["accuracy"] for h in history if "accuracy" in h]
        
        ax.plot(steps, accuracies, marker='o', linestyle='-', linewidth=2)
        ax.set_xlabel("Samples Processed")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy Progress - {run_id}")
        ax.grid(True, alpha=0.3)
        
        fig_file = run_dir / f"{run_id}_accuracy_progress.pdf"
        plt.savefig(fig_file, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Generated figure: {fig_file}")


def create_comparison_figures(all_run_data: List[Dict[str, Any]], results_dir: Path):
    """Create comparison figures for all runs.
    
    Args:
        all_run_data: List of run data from all runs
        results_dir: Directory to save comparison figures
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out None values
    valid_runs = [r for r in all_run_data if r is not None]
    
    if not valid_runs:
        print("No valid runs to compare")
        return
    
    # Comparison bar chart for final accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    
    run_ids = [r["run_id"] for r in valid_runs]
    accuracies = [r["summary"].get("accuracy", 0) for r in valid_runs]
    
    # Use different colors for proposed vs comparative
    colors = []
    for run_id in run_ids:
        if "proposed" in run_id:
            colors.append("#2E86AB")  # Blue for proposed
        else:
            colors.append("#A23B72")  # Purple for comparative
    
    bars = ax.bar(range(len(run_ids)), accuracies, color=colors)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison Across Methods")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    fig_file = comparison_dir / "comparison_accuracy.pdf"
    plt.savefig(fig_file, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Generated comparison figure: {fig_file}")
    
    # If we have history data, create overlay plot
    runs_with_history = [r for r in valid_runs if r.get("history")]
    
    if runs_with_history:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for run_data in runs_with_history:
            run_id = run_data["run_id"]
            history = run_data["history"]
            
            if any("accuracy" in h for h in history):
                steps = [h.get("samples_processed", i) for i, h in enumerate(history) if "accuracy" in h]
                accuracies_over_time = [h["accuracy"] for h in history if "accuracy" in h]
                
                linestyle = '-' if "proposed" in run_id else '--'
                ax.plot(steps, accuracies_over_time, marker='o', linestyle=linestyle, 
                       linewidth=2, label=run_id, alpha=0.8)
        
        ax.set_xlabel("Samples Processed")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Progress Comparison")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        fig_file = comparison_dir / "comparison_accuracy_progress.pdf"
        plt.savefig(fig_file, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Generated comparison figure: {fig_file}")


def create_aggregated_metrics(all_run_data: List[Dict[str, Any]], results_dir: Path):
    """Create aggregated metrics JSON file.
    
    Args:
        all_run_data: List of run data from all runs
        results_dir: Directory to save aggregated metrics
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out None values
    valid_runs = [r for r in all_run_data if r is not None]
    
    if not valid_runs:
        print("No valid runs for aggregated metrics")
        return
    
    # Collect metrics by run_id
    metrics_by_run = {}
    for run_data in valid_runs:
        run_id = run_data["run_id"]
        metrics_by_run[run_id] = run_data["summary"]
    
    # Determine best proposed and best baseline
    proposed_runs = {k: v for k, v in metrics_by_run.items() if "proposed" in k}
    baseline_runs = {k: v for k, v in metrics_by_run.items() if "comparative" in k}
    
    best_proposed = None
    best_proposed_acc = 0
    if proposed_runs:
        best_proposed = max(proposed_runs.items(), key=lambda x: x[1].get("accuracy", 0))
        best_proposed_acc = best_proposed[1].get("accuracy", 0)
    
    best_baseline = None
    best_baseline_acc = 0
    if baseline_runs:
        best_baseline = max(baseline_runs.items(), key=lambda x: x[1].get("accuracy", 0))
        best_baseline_acc = best_baseline[1].get("accuracy", 0)
    
    gap = best_proposed_acc - best_baseline_acc if best_proposed and best_baseline else None
    
    aggregated = {
        "primary_metric": "accuracy",
        "metrics": metrics_by_run,
        "best_proposed": {
            "run_id": best_proposed[0] if best_proposed else None,
            "accuracy": best_proposed_acc,
        } if best_proposed else None,
        "best_baseline": {
            "run_id": best_baseline[0] if best_baseline else None,
            "accuracy": best_baseline_acc,
        } if best_baseline else None,
        "gap": gap,
    }
    
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Aggregated metrics saved to {agg_file}")
    
    if gap is not None:
        print(f"\nPerformance Gap: {gap:+.4f}")
        print(f"Best Proposed: {best_proposed[0]} - {best_proposed_acc:.4f}")
        print(f"Best Baseline: {best_baseline[0]} - {best_baseline_acc:.4f}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs")
    
    args = parser.parse_args()
    
    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    results_dir = Path(args.results_dir)
    
    print(f"Evaluating runs: {run_ids}")
    print(f"Results directory: {results_dir}")
    
    # Load WandB config
    entity, project = load_wandb_config()
    print(f"WandB: {entity}/{project}")
    
    # Fetch data for all runs
    all_run_data = []
    for run_id in run_ids:
        print(f"\nFetching data for run: {run_id}")
        run_data = fetch_run_data(entity, project, run_id)
        
        if run_data:
            all_run_data.append(run_data)
            
            # Export per-run metrics
            export_run_metrics(run_data, results_dir)
            
            # Create per-run figures
            create_per_run_figures(run_data, results_dir)
    
    # Create comparison figures and aggregated metrics
    if all_run_data:
        print("\nGenerating comparison visualizations...")
        create_comparison_figures(all_run_data, results_dir)
        create_aggregated_metrics(all_run_data, results_dir)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
