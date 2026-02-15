# 🧪 Optimal Superplasticizer Dosage Predictor

A machine learning-based tool to predict the optimal dosage of superplasticizers (PCE or SNF) for cement paste based on Marsh Cone Test results.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## 🎯 Overview

This project uses machine learning to predict the **saturation dosage** of superplasticizers in cement paste. The saturation point is where adding more superplasticizer does not improve the fluidity of the paste, as measured by the Marsh Cone flow test.

### Key Features
✅ Predicts optimal dosage for PCE and SNF superplasticizers
✅ Works with different water-to-cement (W/C) ratios
✅ Interactive web interface (Streamlit)
✅ Command-line tool for quick predictions
✅ Comprehensive visualizations of flow curves

## 📊 Dataset

**Total Data Points:** 48 flow time measurements
- PCE: 21 measurements (7 dosages × 3 W/C ratios)
- SNF: 27 measurements (9 dosages × 3 W/C ratios)

**Saturation Points Identified:**
| SP Type | W/C 0.35 | W/C 0.40 | W/C 0.45 |
|---------|----------|----------|----------|
| **PCE** | 0.40% | 0.40% | 0.35% |
| **SNF** | 0.50% | 0.45% | 0.45% |

## 🎯 Model Performance

**Best Model:** Gradient Boosting Regressor
- MAE: 0.025%
- RMSE: 0.035%
- R² Score: 0.455

## 🚀 Installation & Usage

### Option 1: Command Line
```bash
python predict_dosage.py --wc 0.40 --sp PCE
```

### Option 2: Web Interface
```bash
streamlit run streamlit_app.py
```

## 📈 Key Findings

1. **PCE is more efficient:** Requires 20% less dosage than SNF
2. **W/C ratio matters:** Higher W/C generally needs lower dosage
3. **Saturation plateau:** Adding more SP beyond saturation wastes material

## 🔮 Future Enhancements
- Add Silica Fume variations (5%, 10%, 15%)
- Expand W/C ratio range
- Neural network models
- Mobile app version
