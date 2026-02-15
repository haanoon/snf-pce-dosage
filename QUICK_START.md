# 🚀 QUICK START GUIDE

## Get Started in 3 Minutes!

### Step 1: Install Dependencies (1 minute)
```bash
pip install -r requirements.txt
```

### Step 2: Make a Prediction (30 seconds)

**Option A: Command Line** (Fastest)
```bash
python predict_dosage.py --wc 0.40 --sp PCE
```

**Option B: Web Interface** (Most Interactive)
```bash
streamlit run streamlit_app.py
```

### Step 3: Interpret Results

**Example Output:**
```
Predicted Optimal Dosage: 0.400%
```

**What this means:**
- For 100 kg cement → Use 400 grams of PCE
- For 1000 kg cement → Use 4.0 kg of PCE
- This is the saturation point (don't add more!)

---

## Common Use Cases

### 1️⃣ Quick Dosage Check
```bash
# I'm using W/C 0.40 with PCE, what dosage?
python predict_dosage.py --wc 0.40 --sp PCE

# Output: 0.400%
```

### 2️⃣ Compare PCE vs SNF
```bash
# Check both for W/C 0.35
python predict_dosage.py --wc 0.35 --sp PCE  # → 0.400%
python predict_dosage.py --wc 0.35 --sp SNF  # → 0.500%

# SNF needs 25% more dosage!
```

### 3️⃣ Interpolate Unknown W/C Ratio
```bash
# What if my W/C is 0.37?
python predict_dosage.py --wc 0.37 --sp PCE --verbose

# Model interpolates between 0.35 and 0.40
```

### 4️⃣ Batch Calculations
```bash
# For 500 kg cement batch with W/C 0.42, SNF
python predict_dosage.py --wc 0.42 --sp SNF

# Output: 0.450%
# Calculation: 500 kg × 0.0045 = 2.25 kg SNF needed
```

---

## Files You Need

**Essential (for predictions):**
- ✅ `optimal_dosage_predictor.pkl` - The trained model
- ✅ `label_encoder.pkl` - Encodes PCE/SNF
- ✅ `predict_dosage.py` - Command-line tool

**For Web Interface:**
- ✅ `streamlit_app.py` - Web dashboard
- ✅ `processed_flow_data.csv` - Flow measurements
- ✅ `saturation_points.csv` - Training data

**Documentation:**
- 📄 `README.md` - Full documentation
- 📄 `PROJECT_SUMMARY.md` - Detailed project info
- 📄 This file - Quick start

---

## Troubleshooting

**Problem:** "Model files not found"
```bash
# Solution: Make sure you're in the right directory
ls *.pkl
# Should show: optimal_dosage_predictor.pkl, label_encoder.pkl
```

**Problem:** "Module not found"
```bash
# Solution: Install missing packages
pip install pandas scikit-learn joblib numpy
```

**Problem:** "Streamlit won't start"
```bash
# Solution: Install streamlit
pip install streamlit plotly
streamlit run streamlit_app.py
```

---

## Tips for Best Results

✅ **DO:**
- Use W/C ratios between 0.35-0.45 (best accuracy)
- Test predictions in the lab before large batches
- Use --verbose flag for detailed explanations
- Check the web interface for visualizations

❌ **DON'T:**
- Extrapolate far beyond W/C 0.30-0.50
- Assume model works for Silica Fume (not trained yet)
- Skip lab validation for critical applications
- Add more SP than predicted (wastes material)

---

## Command Reference

```bash
# Basic prediction
python predict_dosage.py --wc <ratio> --sp <type>

# With detailed output
python predict_dosage.py --wc <ratio> --sp <type> --verbose

# Web interface
streamlit run streamlit_app.py

# Install all dependencies
pip install -r requirements.txt
```

**Parameters:**
- `--wc` : Water-cement ratio (0.30 to 0.50)
- `--sp` : Superplasticizer type (PCE or SNF)
- `--verbose` : Show detailed interpretation

---

## Next Steps

1. **Try a few predictions** with different W/C ratios
2. **Validate in lab** - Test 2-3 predictions with Marsh Cone
3. **Use in mix design** - Apply to actual concrete batches
4. **Expand dataset** - Add Silica Fume data for better predictions

---

## Need Help?

- 📖 Read `README.md` for full documentation
- 📊 Use web interface for visual exploration
- 📧 Check `PROJECT_SUMMARY.md` for detailed insights
- 🔬 Validate predictions with lab tests

---

**You're ready to go! Start with:**
```bash
python predict_dosage.py --wc 0.40 --sp PCE
```

🎉 **Happy Predicting!**
