import pickle
import os
from pathlib import Path
from hybrid_saved import EnhancedHybridRecommender

def load_hybrid_model(model_path):
    """Load the hybrid model from either object or dictionary format"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    with open(model_path, 'rb') as f:
        loaded = pickle.load(f)
    
    # If it's already a model instance
    if isinstance(loaded, EnhancedHybridRecommender):
        return loaded
        
    # If it's a saved state dictionary
    if isinstance(loaded, dict):
        model = EnhancedHybridRecommender()
        
        # Manually set all attributes from the dictionary
        for key, value in loaded.items():
            setattr(model, key, value)
            
        return model
    
    raise ValueError("Unknown model format in file")