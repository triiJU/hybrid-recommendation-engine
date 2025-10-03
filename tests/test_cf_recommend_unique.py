"""
Test that CF recommendations have unique item IDs and expected length.
"""
import pandas as pd
from src.cf_baseline import CollaborativeFiltering


def test_cf_recommend_unique():
    """Test that CF recommendations return unique item IDs."""
    # Create minimal test data
    data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'item_id': [10, 20, 30, 10, 40, 50, 20, 30, 40],
        'rating': [5, 4, 3, 5, 5, 4, 4, 5, 3]
    }
    df = pd.DataFrame(data)
    
    # Fit CF model
    cf = CollaborativeFiltering(mode="user")
    cf.fit(df)
    
    # Get recommendations for user 1
    recs = cf.recommend(user_id=1, n=10)
    
    # Extract item IDs
    item_ids = [item_id for item_id, score in recs]
    
    # Check uniqueness
    assert len(item_ids) == len(set(item_ids)), "CF recommendations should have unique item IDs"
    
    # Check length is reasonable
    assert len(item_ids) <= 10, "Should not return more than requested"
    assert len(item_ids) > 0, "Should return at least one recommendation"


def test_cf_recommend_excludes_seen():
    """Test that CF recommendations exclude already rated items."""
    # Create minimal test data
    data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'item_id': [10, 20, 30, 10, 40, 50, 20, 30, 40],
        'rating': [5, 4, 3, 5, 5, 4, 4, 5, 3]
    }
    df = pd.DataFrame(data)
    
    # Fit CF model
    cf = CollaborativeFiltering(mode="user")
    cf.fit(df)
    
    # Get recommendations for user 1
    recs = cf.recommend(user_id=1, n=10)
    
    # Extract item IDs
    item_ids = [item_id for item_id, score in recs]
    
    # User 1 has rated items 10, 20, 30
    rated_items = {10, 20, 30}
    
    # Check that rated items are not in recommendations
    for item_id in item_ids:
        assert item_id not in rated_items, f"Item {item_id} was already rated by user"
