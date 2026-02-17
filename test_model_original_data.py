import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def test_model():
    print("🚀 Testing model with original saturation points data...")

    # Load artifacts
    try:
        model = joblib.load('optimal_dosage_predictor.pkl')
        le = joblib.load('label_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✅ Loaded model, encoder, and scaler.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return

    # Load data
    try:
        data = pd.read_csv('saturation_points.csv')
        print(f"✅ Loaded {len(data)} rows from saturation_points.csv")
    except FileNotFoundError:
        print("❌ Error: saturation_points.csv not found!")
        return

    # Prepare features
    X = data[['wc_ratio', 'sp_type', 'silica_fume']].copy()
    y_true = data['optimal_dosage']

    # Encode categorical variable
    X['sp_type'] = le.transform(X['sp_type'])

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    y_pred = model.predict(X_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n📊 Performance Metrics on Original Data:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")

    # Show detailed comparison
    print("\n📝 Detailed Comparison:")
    comparison_df = data[['sp_type', 'wc_ratio', 'silica_fume', 'optimal_dosage']].copy()
    comparison_df['predicted_dosage'] = y_pred
    comparison_df['difference'] = comparison_df['predicted_dosage'] - comparison_df['optimal_dosage']
    
    print(comparison_df.to_string(index=False))

    # Identify large errors
    threshold = 0.05
    large_errors = comparison_df[abs(comparison_df['difference']) > threshold]
    if not large_errors.empty:
        print(f"\n⚠️  Found {len(large_errors)} predictions with error > {threshold}:")
        print(large_errors.to_string(index=False))
    else:
        print(f"\n✅ All predictions are within {threshold} error margin.")

if __name__ == "__main__":
    test_model()
