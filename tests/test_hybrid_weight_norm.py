"""
Test weight normalization in hybrid.py.
"""
from src.hybrid import normalize_weights


def test_weight_normalization_basic():
    """Test that weights are normalized to sum to 1.0."""
    w_cf, w_content, w_neural = normalize_weights(0.6, 0.3, 0.1, neural_available=True)
    
    # Check they sum to 1.0
    assert abs((w_cf + w_content + w_neural) - 1.0) < 1e-6
    
    # Check proportions are maintained
    assert abs(w_cf - 0.6) < 1e-6
    assert abs(w_content - 0.3) < 1e-6
    assert abs(w_neural - 0.1) < 1e-6


def test_weight_normalization_no_neural():
    """Test that neural weight is zeroed when neural model not available."""
    w_cf, w_content, w_neural = normalize_weights(0.6, 0.4, 0.2, neural_available=False)
    
    # Neural should be zero
    assert w_neural == 0.0
    
    # CF and content should be normalized
    assert abs((w_cf + w_content) - 1.0) < 1e-6
    assert abs(w_cf - 0.6) < 1e-6
    assert abs(w_content - 0.4) < 1e-6


def test_weight_normalization_all_zero():
    """Test fallback when all weights are zero."""
    w_cf, w_content, w_neural = normalize_weights(0.0, 0.0, 0.0, neural_available=True)
    
    # Should fallback to CF=1.0
    assert w_cf == 1.0
    assert w_content == 0.0
    assert w_neural == 0.0


def test_weight_normalization_negative():
    """Test fallback when weights are negative."""
    w_cf, w_content, w_neural = normalize_weights(-0.5, -0.3, -0.2, neural_available=True)
    
    # Should fallback to CF=1.0
    assert w_cf == 1.0
    assert w_content == 0.0
    assert w_neural == 0.0


def test_weight_normalization_unequal():
    """Test normalization with unequal weights."""
    w_cf, w_content, w_neural = normalize_weights(1.0, 2.0, 3.0, neural_available=True)
    
    # Should sum to 1.0
    assert abs((w_cf + w_content + w_neural) - 1.0) < 1e-6
    
    # Check proportions
    total = 1.0 + 2.0 + 3.0
    assert abs(w_cf - 1.0/total) < 1e-6
    assert abs(w_content - 2.0/total) < 1e-6
    assert abs(w_neural - 3.0/total) < 1e-6
