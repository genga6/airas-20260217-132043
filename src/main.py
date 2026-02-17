"""Main orchestrator for inference experiments."""

import os
import sys
import subprocess
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Orchestrate inference experiment execution."""
    
    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        print("Running in SANITY_CHECK mode - applying overrides")
        # Limit dataset samples
        cfg.run.dataset.max_samples = 10
        # Set wandb project to sanity namespace
        if cfg.wandb.project and not cfg.wandb.project.endswith("-sanity"):
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"
        print(f"Overridden dataset.max_samples to {cfg.run.dataset.max_samples}")
        print(f"Overridden wandb.project to {cfg.wandb.project}")
    
    # Create results directory for this run
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Determine task type and invoke appropriate script
    # This is an inference-only task
    print("\nInvoking inference script...")
    
    # Prepare environment and command
    env = os.environ.copy()
    
    # Run inference as a subprocess
    cmd = [
        sys.executable, "-u", "-m", "src.inference",
        f"--config-name=config",
        f"run={cfg.run.run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            cwd=Path.cwd(),
        )
        print(f"\nInference completed with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"\nInference failed with return code: {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
