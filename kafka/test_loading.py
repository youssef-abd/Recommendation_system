# test_model_loading.py
from model_loader import load_hybrid_model

try:
    model = load_hybrid_model('models/hybrid_model.pkl')
    print("SUCCESS! Model loaded with:")
    print(f"- Version: {model.model_version}")
    print(f"- Products: {len(model.popularity_scores)}")
    print(f"- Users: {len(model.user_activity_levels)}")
except Exception as e:
    print(f"FAILED: {str(e)}")