import pandas as pd

def test_hybrid_model(filepath, test_user='test-user-123', top_k=10):
    from hybrid_saved import EnhancedHybridRecommender  # Adjust if your class/file name differs
    
    print(f"Loading model from {filepath}...")
    model = EnhancedHybridRecommender.load_model(filepath)
    
    print("\n=== Model Internal State Overview ===")
    print(f"Model version: {model.model_version}")
    print(f"Is trained: {model.is_trained}")
    print(f"Training timestamp: {model.training_timestamp}")
    print(f"Popularity scores count: {len(model.popularity_scores)}")
    print(f"User activity levels count: {len(model.user_activity_levels)}")
    print(f"Content-based lookup entries count: {len(model.cb_lookup)}")
    print(f"User profiles count: {len(model.user_profiles)}")
    print(f"Product features count: {len(model.product_features)}")

    # Sample some popularity scores
    sample_popularity = list(model.popularity_scores.items())[:5]
    print("\nSample popularity scores (product_id, score):")
    for p in sample_popularity:
        print(p)
    
    # Sample CB lookup entries
    sample_cb = list(model.cb_lookup.items())[:5]
    print("\nSample CB lookup (product_id, score):")
    for p in sample_cb:
        print(p)

    # Test adaptive K
    try:
        adaptive_k = model._get_adaptive_k(test_user)
        print(f"\nAdaptive K for user '{test_user}': {adaptive_k}")
    except Exception as e:
        print(f"Error calling _get_adaptive_k: {e}")

    # Optionally test predictions for user
    print(f"\nGenerating predictions for user '{test_user}' with top_k={top_k}...")
    try:
        preds = model.predict_for_user(test_user, top_k=top_k)
        if preds.empty:
            print("No predictions generated.")
        else:
            print(preds.head(top_k))
    except Exception as e:
        print(f"Error during predict_for_user: {e}")

    # Run your generate_hybrid_recommendations method
    print(f"\nRunning generate_hybrid_recommendations with top_k={top_k}...")
    try:
        hybrid_recs = model.generate_hybrid_recommendations(top_k=top_k)
        if hybrid_recs.empty:
            print("No hybrid recommendations generated.")
        else:
            print(hybrid_recs.head(top_k))
    except Exception as e:
        print(f"Error during generate_hybrid_recommendations: {e}")

if __name__ == "__main__":
    # Adjust the path to your saved model file here
    model_path = "models/hybrid_model.pkl"
    test_hybrid_model(model_path)
