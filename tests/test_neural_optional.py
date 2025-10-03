"""
Test that neural model methods handle missing models gracefully.
"""
import os
import torch
from src.neural_cf import NeuralCF


def test_neural_predict_single_unknown_user():
    """Test that predict_single returns 0.0 for unknown user."""
    model = NeuralCF(n_users=10, n_items=20)
    model.user_map = {i: i for i in range(10)}
    model.item_map = {i: i for i in range(20)}
    
    # Unknown user should return 0.0
    score = model.predict_single(user_id=999, item_id=5)
    assert score == 0.0


def test_neural_predict_single_unknown_item():
    """Test that predict_single returns 0.0 for unknown item."""
    model = NeuralCF(n_users=10, n_items=20)
    model.user_map = {i: i for i in range(10)}
    model.item_map = {i: i for i in range(20)}
    
    # Unknown item should return 0.0
    score = model.predict_single(user_id=5, item_id=999)
    assert score == 0.0


def test_neural_predict_single_known():
    """Test that predict_single returns a float for known user-item pairs."""
    model = NeuralCF(n_users=10, n_items=20)
    model.user_map = {i: i for i in range(10)}
    model.item_map = {i: i for i in range(20)}
    model.eval()
    
    # Known user and item should return a float
    score = model.predict_single(user_id=5, item_id=10)
    assert isinstance(score, float)


def test_neural_load_from_nonexistent_file():
    """Test that load_from_file handles missing file appropriately."""
    fake_path = "/tmp/nonexistent_model.pt"
    
    # Should raise an exception
    try:
        model = NeuralCF.load_from_file(fake_path)
        assert False, "Should have raised an exception"
    except (FileNotFoundError, RuntimeError, OSError):
        # Expected behavior
        pass


def test_neural_load_save_roundtrip():
    """Test that we can save and load a model."""
    # Create a simple model
    model = NeuralCF(n_users=5, n_items=10)
    model.user_map = {i: i for i in range(5)}
    model.item_map = {i: i for i in range(10)}
    
    # Save it
    temp_path = "/tmp/test_neural_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "user_map": model.user_map,
        "item_map": model.item_map
    }, temp_path)
    
    # Load it back
    loaded_model = NeuralCF.load_from_file(temp_path)
    
    # Check that maps are correct
    assert loaded_model.user_map == model.user_map
    assert loaded_model.item_map == model.item_map
    
    # Check that predictions work
    score = loaded_model.predict_single(user_id=2, item_id=5)
    assert isinstance(score, float)
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
