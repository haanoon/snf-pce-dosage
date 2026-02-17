import sys
print("🚀 Starting imports...", flush=True)
import pandas as pd
print("✅ Pandas imported", flush=True)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
print("✅ Imports complete. Starting main logic...", flush=True)

def main():
    print("🚀 Starting Model Comparison Analysis for 'sname' (sp_type)...", flush=True)

    # Load the requested oversampled dataset
    file_path = '05_oversampled_hybrid_RECOMMENDED (2).csv'
    try:
        data = pd.read_csv(file_path)
        print(f"✅ Loaded dataset: {file_path} with {len(data)} rows.", flush=True)
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found.", flush=True)
        return

    # Define Target and Features
    target_col = 'sp_type'
    if target_col not in data.columns:
        print(f"❌ Error: Target column '{target_col}' not found in dataset.", flush=True)
        print(f"   Available columns: {data.columns.tolist()}", flush=True)
        return

    feature_cols = [c for c in data.columns if c != target_col]
    
    print(f"🎯 Target Variable: {target_col}", flush=True)
    print(f"📋 Feature Variables: {feature_cols}", flush=True)

    X = data[feature_cols]
    y = data[target_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"🔤 Encoded target classes: {le.classes_}", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🌲 Training Model 1: Random Forest Classifier...", flush=True)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"✅ Random Forest Accuracy: {acc_rf:.4f}", flush=True)

    print("\n🧠 Training Model 2: Artificial Neural Network (MLP)...", flush=True)
    ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    ann_model.fit(X_train_scaled, y_train)
    y_pred_ann = ann_model.predict(X_test_scaled)
    acc_ann = accuracy_score(y_test, y_pred_ann)
    print(f"✅ ANN Accuracy: {acc_ann:.4f}", flush=True)

    print("\n📊 --- Comparative Analysis --- 📊", flush=True)
    results = pd.DataFrame({
        'Model': ['Random Forest', 'ANN (MLP)'],
        'Accuracy': [acc_rf, acc_ann]
    })
    print(results, flush=True)

    # Generate Comparison Chart
    print("\n🎨 Generating comparison chart...", flush=True)
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    # Create bar plot
    ax = sns.barplot(x='Model', y='Accuracy', data=results, palette=['#1f77b4', '#ff7f0e'])
    
    # Add labels and title
    plt.ylim(0, 1.1)
    plt.title('Model Comparison: Random Forest vs ANN (MLP)', fontsize=16)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.xlabel('Model Type', fontsize=12)
    
    # Add value annotations on top of bars
    for i, v in enumerate(results['Accuracy']):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Save the chart
    output_image = 'model_comparison_chart.png'
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved as '{output_image}'", flush=True)

    print("\n📝 Detailed Classification Reports:", flush=True)
    print("\n--- Random Forest Report ---", flush=True)
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    
    print("\n--- ANN Report ---", flush=True)
    print(classification_report(y_test, y_pred_ann, target_names=le.classes_))

    print("\n🔎 Conclusion:", flush=True)
    if acc_ann > acc_rf:
        print(f"   The ANN performed better ({acc_ann:.2%} vs {acc_rf:.2%}).", flush=True)
    elif acc_rf > acc_ann:
        print(f"   The Random Forest performed better ({acc_rf:.2%} vs {acc_ann:.2%}).", flush=True)
    else:
        print(f"   Both models performed equally well ({acc_ann:.2%}).", flush=True)
    
if __name__ == "__main__":
    main()
