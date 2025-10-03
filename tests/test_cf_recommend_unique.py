"""
Test that CF recommendations contain unique item IDs.
"""
import pandas as pd
from src.cf_baseline import CollaborativeFiltering


def test_cf_recommend_unique():
    """Ensure CF recommendations contain unique item ids."""
    # Create a simple test dataset
    data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'item_id': [10, 20, 30, 10, 40, 50, 20, 30, 40],
        'rating': [5, 4, 3, 4, 5, 3, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    
    # Fit CF model
    cf = CollaborativeFiltering(mode="user")
    cf.fit(df)
    
    # Get recommendations for user 1
    recs = cf.recommend(user_id=1, n=5)
    
    # Extract item IDs
    item_ids = [item_id for item_id, score in recs]
    
    # Check uniqueness
    assert len(item_ids) == len(set(item_ids)), "CF recommendations should contain unique item IDs"
    
    # Check that recommended items are not already rated by the user
    user_rated = set([10, 20, 30])
    for item_id in item_ids:
        assert item_id not in user_rated, f"Item {item_id} was already rated by user"
