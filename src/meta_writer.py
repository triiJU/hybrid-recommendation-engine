"""
Meta writer for experiment metadata.
"""
import json
import os
from datetime import datetime
import subprocess


def get_git_commit():
    """
    Get current git commit hash if available.
    
    Returns:
        Git commit hash string or None if not available
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def write_meta(
    output_path="experiments/meta.json",
    seed=None,
    weights=None,
    model_presence=None
):
    """
    Write experiment metadata to JSON file.
    
    Args:
        output_path: Path to write metadata
        seed: Random seed used
        weights: Dict of model weights (optional)
        model_presence: Dict with bool flags for content/neural models (optional)
    """
    meta = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit()
    }
    
    if seed is not None:
        meta["seed"] = seed
    
    if weights is not None:
        meta["weights"] = weights
    
    if model_presence is not None:
        meta["model_presence"] = model_presence
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    return meta


if __name__ == "__main__":
    # Example usage
    meta = write_meta(
        seed=42,
        weights={"cf": 0.6, "content": 0.4, "neural": 0.0},
        model_presence={"content": True, "neural": False}
    )
    print(f"Wrote metadata: {meta}")
