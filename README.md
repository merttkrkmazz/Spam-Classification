# CS452 Data Science Project - Spam Email Classification

## 📧 Project Overview
This project implements a machine learning-based spam email classification system using the Spambase dataset from UCI Machine Learning Repository.

## 📊 Dataset
- **Source:** [UCI Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/Spambase)
- **Size:** 4,601 emails
- **Features:** 57 features (word/character frequencies)
- **Target:** Binary classification (Spam vs Normal)
- **Distribution:** 1,813 spam (39.4%) / 2,788 normal (60.6%)

## 🔬 Methodology

### 1. Feature Selection
- **Method:** SelectKBest with ANOVA F-test
- **Result:** Selected top 20 most discriminative features from 57
- **Purpose:** Reduce dimensionality and improve model performance

### 2. Sampling Strategy
- Train/Test Split: 80% / 20%
- Stratified sampling to maintain class distribution
- StandardScaler for feature normalization
- 5-Fold Cross-Validation for robust evaluation

### 3. Classification Algorithms
Implemented and compared 4 different classifiers:
1. **Logistic Regression** - Linear baseline model
2. **K-Nearest Neighbors (KNN)** - Instance-based learning (k=5)
3. **Random Forest** - Ensemble of 100 decision trees
4. **Support Vector Machine (SVM)** - RBF kernel

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Cross-Validation Score (5-fold)

### 5. Statistical Testing
- Paired t-test to compare best performing algorithms
- Significance level: α = 0.05

## 📈 Results

| Algorithm | Accuracy | Precision | Recall | F1-Score | CV Mean |
|-----------|----------|-----------|--------|----------|---------|
| Logistic Regression | 88.60% | 92.28% | 79.74% | 85.56% | 88.97% |
| KNN | 89.79% | 93.02% | 82.05% | 87.19% | 89.86% |
| SVM | 92.07% | 93.19% | 87.69% | 90.36% | 91.44% |
| **Random Forest** | **94.03%** | **95.15%** | **90.51%** | **92.77%** | **93.83%** |

### 🏆 Best Performer: Random Forest
- **Accuracy:** 94.03%
- **F1-Score:** 92.77%
- **Statistical Significance:** p-value = 0.0007 (< 0.05) ✓

The Random Forest classifier significantly outperformed other algorithms with statistical significance (paired t-test, p < 0.05).

## 🛠️ Requirements
```bash
numpy
pandas
matplotlib
scikit-learn
scipy
```

## 🚀 Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
```

## 📁 Project Structure
```
CS452_Project/
├── main.py              # Main implementation
├── spambase.data        # Dataset
├── results.png          # Visualization results
├── README.md
└── requirements.txt
```

## 📊 Visualizations
The project generates comparison charts showing:
- Accuracy comparison across all algorithms
- All metrics (Accuracy, Precision, Recall, F1-Score) visualization

Results are saved as `results.png`.

## 🎓 Course Information
- **Course:** CS452 - Data Science
- **Project Type:** Classification
- **Date:** January 2026

## 📝 License
This project is for educational purposes.
