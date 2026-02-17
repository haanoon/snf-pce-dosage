import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_model():
    print("🚀 Starting model training with Silica Fume integration...")

    # Load data
    try:
        data = pd.read_csv('05_oversampled_hybrid_RECOMMENDED (2).csv')
        print(f"✅ Loaded {len(data)} saturation points from 05_oversampled_hybrid_RECOMMENDED (2).csv")
    except FileNotFoundError:
        print("❌ Error: Dataset not found!")
        return

    # Prepare features and target
    # Features: W/C Ratio, SP Type, Silica Fume %
    X = data[['wc_ratio', 'sp_type', 'silica_fume']].copy()
    y = data['optimal_dosage']

    # Encode categorical variable
    le = LabelEncoder()
    X['sp_type'] = le.fit_transform(X['sp_type'])
    
    # Scale features (Critical for ANN - Commented out for GB)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    
    # Initialize model (Gradient Boosting)
    # from sklearn.neural_network import MLPRegressor
    # model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model = GradientBoostingRegressor(random_state=42)

    # Validate with LOOCV
    print("\n📊 Validating model with Leave-One-Out Cross-Validation...")
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    # Convert X to numpy to avoid indexing issues
    # X_np = X_scaled 
    X_np = X.values # Use original features for GB
    y_np = y.values

    for train_index, test_index in loo.split(X_np):
        X_train, X_test = X_np[train_index], X_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        
        y_true.append(y_test[0])
        y_pred.append(pred)
        
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"   MAE:  {mae:.4f}%")
    print(f"   RMSE: {rmse:.4f}%")

    if mae > 0.05:
        print("⚠️  Warning: MAE is higher than ideal targets (likely due to dataset noise)")
    else:
        print("✅ Performance meets expectations!")

    # Train final model on full dataset
    print("\n💾 Training final model on full dataset...")
    # model.fit(X_scaled, y)
    model.fit(X, y) # Fit on original X

    # Save artifacts
    joblib.dump(model, 'optimal_dosage_predictor.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    print("✅ Model saved to 'optimal_dosage_predictor.pkl'")
    print("✅ Encoder saved to 'label_encoder.pkl'")
    # joblib.dump(scaler, 'scaler.pkl')
    # print("✅ Scaler saved to 'scaler.pkl'")
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    train_model()
