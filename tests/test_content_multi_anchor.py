"""
Test multi_anchor_recommend method for ContentModel.
"""
import pandas as pd
from src.content_based import ContentModel, ContentRecommender


def test_multi_anchor_recommend():
    """Test multi_anchor_recommend returns expected format."""
    # Create minimal test data
    items_data = {
        'item_id': [1, 2, 3, 4, 5],
        'title': [
            'The Matrix',
            'The Matrix Reloaded',
            'The Matrix Revolutions',
            'Inception',
            'Interstellar'
        ]
    }
    items_df = pd.DataFrame(items_data)
    
    # Fit content model
    cm = ContentModel()
    cm.fit(items_df, text_col='title')
    
    # Get recommendations using multiple anchors
    anchors = [1, 2]  # Matrix movies
    recs = cm.multi_anchor_recommend(anchors, exclude=None, top_n=3)
    
    # Check return format
    assert isinstance(recs, list), "Should return a list"
    assert len(recs) <= 3, "Should not return more than top_n"
    
    for item_id, score in recs:
        assert isinstance(item_id, (int, pd.Int64Dtype)), "Item ID should be integer"
        assert isinstance(score, float), "Score should be float"
        assert item_id not in anchors, "Should not recommend anchor items"


def test_multi_anchor_recommend_exclude():
    """Test multi_anchor_recommend respects exclude set."""
    # Create minimal test data
    items_data = {
        'item_id': [1, 2, 3, 4, 5],
        'title': [
            'The Matrix',
            'The Matrix Reloaded',
            'The Matrix Revolutions',
            'Inception',
            'Interstellar'
        ]
    }
    items_df = pd.DataFrame(items_data)
    
    # Fit content model
    cm = ContentModel()
    cm.fit(items_df, text_col='title')
    
    # Get recommendations with exclusions
    anchors = [1]
    exclude = {2, 3}
    recs = cm.multi_anchor_recommend(anchors, exclude=exclude, top_n=5)
    
    # Check that excluded items are not in recommendations
    rec_ids = [item_id for item_id, score in recs]
    for excluded_id in exclude:
        assert excluded_id not in rec_ids, f"Item {excluded_id} should be excluded"


def test_content_recommender_alias():
    """Test that ContentRecommender is an alias for ContentModel."""
    assert ContentRecommender is ContentModel, "ContentRecommender should be an alias for ContentModel"
    
    # Test that it can be instantiated
    cr = ContentRecommender()
    assert isinstance(cr, ContentModel), "ContentRecommender instance should be a ContentModel"
