import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compare_regression_models():
    print("🚀 Starting Regression Model Comparison (ANN vs. Gradient Boosting)...", flush=True)

    # 1. Load Data
    file_path = '05_oversampled_hybrid_RECOMMENDED (2).csv'
    try:
        data = pd.read_csv(file_path)
        print(f"✅ Loaded dataset: {file_path} with {len(data)} rows.", flush=True)
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found.", flush=True)
        return

    # 2. Prepare Data
    # Target: optimal_dosage
    # Features: wc_ratio, sp_type, silica_fume
    X = data[['wc_ratio', 'sp_type', 'silica_fume']].copy()
    y = data['optimal_dosage']

    # Encode Categorical Data
    le = LabelEncoder()
    X['sp_type'] = le.fit_transform(X['sp_type'])
    
    # Scale Data (Important for ANN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Initialize Models
    # Model A: Artificial Neural Network (MLPRegressor)
    ann_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
    
    # Model B: Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(random_state=42)

    # 4. Cross-Validation Evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    ann_scores = {'mae': [], 'rmse': [], 'r2': []}
    gb_scores = {'mae': [], 'rmse': [], 'r2': []}

    print("\n📊 Running 5-Fold Cross-Validation...", flush=True)
    
    X_np = X_scaled
    y_np = y.values

    for train_index, test_index in kf.split(X_np):
        X_train, X_test = X_np[train_index], X_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]
        
        # Train & Predict ANN
        ann_model.fit(X_train, y_train)
        y_pred_ann = ann_model.predict(X_test)
        
        # Train & Predict GB
        gb_model.fit(X_train, y_train)
        y_pred_gb = gb_model.predict(X_test)
        
        # Calculate Metrics
        ann_scores['mae'].append(mean_absolute_error(y_test, y_pred_ann))
        ann_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred_ann)))
        ann_scores['r2'].append(r2_score(y_test, y_pred_ann))
        
        gb_scores['mae'].append(mean_absolute_error(y_test, y_pred_gb))
        gb_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred_gb)))
        gb_scores['r2'].append(r2_score(y_test, y_pred_gb))

    # 5. Aggregate Results
    results = pd.DataFrame({
        'Model': ['ANN (MLPRegressor)', 'Gradient Boosting'],
        'MAE': [np.mean(ann_scores['mae']), np.mean(gb_scores['mae'])],
        'RMSE': [np.mean(ann_scores['rmse']), np.mean(gb_scores['rmse'])],
        'R2 Score': [np.mean(ann_scores['r2']), np.mean(gb_scores['r2'])]
    })

    print("\n🏆 --- Model Performance Summary ---", flush=True)
    print(results.to_string(index=False), flush=True)

    # 6. Generate Comparison Graphs
    print("\n🎨 Generating comparison graphs...", flush=True)
    
    # Graph 1: Performance Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    melted_results = results.melt(id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(x='Metric', y='Score', hue='Model', data=melted_results, palette='magma')
    plt.title('Model Comparison: ANN vs Gradient Boosting')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('gb_ann_comparison.png', dpi=300)
    print("✅ Saved 'gb_ann_comparison.png'", flush=True)

    # Graph 2: Predicted vs Actual Scatter Plot (using last fold data for visualization)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_ann, alpha=0.7, color='blue', label='Predictions')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'ANN: Actual vs Predicted (R2={r2_score(y_test, y_pred_ann):.2f})')
    plt.xlabel('Actual Optimal Dosage')
    plt.ylabel('Predicted Dosage')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_gb, alpha=0.7, color='purple', label='Predictions')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'Gradient Boosting: Actual vs Predicted (R2={r2_score(y_test, y_pred_gb):.2f})')
    plt.xlabel('Actual Optimal Dosage')
    plt.ylabel('Predicted Dosage')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gb_ann_scatter.png', dpi=300)
    print("✅ Saved 'gb_ann_scatter.png'", flush=True)

    print("\n🔎 Conclusion:", flush=True)
    if results.iloc[0]['MAE'] < results.iloc[1]['MAE']:
        print("   The ANN model outperformed Gradient Boosting.", flush=True)
    else:
        print("   Gradient Boosting performed better (or similar) to ANN.", flush=True)

if __name__ == "__main__":
    compare_regression_models()
