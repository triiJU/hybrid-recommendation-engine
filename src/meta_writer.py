"""
Meta writer for tracking experiment metadata.
"""
import json
import os
import subprocess
from datetime import datetime


def write_meta(output_path="experiments/meta.json", seed=None, weights=None, 
               neural_loaded=None, content_present=None):
    """
    Write experiment metadata to a JSON file.
    
    Args:
        output_path: Path to write the metadata file
        seed: Random seed used
        weights: Dictionary of blend weights
        neural_loaded: Whether neural model was loaded
        content_present: Whether content model is present
    """
    meta = {}
    
    # Timestamp
    meta["timestamp"] = datetime.now().isoformat()
    
    # Git commit (if available)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            meta["git_commit"] = result.stdout.strip()
    except Exception:
        pass
    
    # Add optional fields
    if seed is not None:
        meta["seed"] = seed
    
    if weights is not None:
        meta["weights"] = weights
    
    if neural_loaded is not None:
        meta["neural_loaded"] = neural_loaded
    
    if content_present is not None:
        meta["content_present"] = content_present
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to file
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    return meta


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Write experiment metadata")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--neural_loaded", action="store_true", help="Neural model loaded")
    parser.add_argument("--content_present", action="store_true", help="Content model present")
    parser.add_argument("--output", default="experiments/meta.json", help="Output path")
    args = parser.parse_args()
    
    # Try to read weights from hybrid_result.json if it exists
    weights = None
    hybrid_result_path = "experiments/hybrid_result.json"
    if os.path.exists(hybrid_result_path):
        try:
            with open(hybrid_result_path) as f:
                data = json.load(f)
                if "weights" in data:
                    weights = data["weights"]
        except Exception:
            pass
    
    meta = write_meta(
        output_path=args.output,
        seed=args.seed,
        weights=weights,
        neural_loaded=args.neural_loaded,
        content_present=args.content_present
    )
    
    print(json.dumps(meta, indent=2))
