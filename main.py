# =============================================================================
# CS452 - Data Science Project
# Spam Email Classification
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =============================================================================
# 1. DATASET YUKLEME
# =============================================================================
print("=" * 60)
print("1. DATASET YUKLEME")
print("=" * 60)

# Spambase dataset'ini dosyadan oku
data_path = Path(__file__).resolve().parent / "spambase.data"
df = pd.read_csv(data_path, header=None)

# Son sutun target (spam=1, not spam=0)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Toplam ornek sayisi: {len(y)}")
print(f"Feature sayisi: {X.shape[1]}")
print(f"Spam sayisi: {sum(y == 1)}")
print(f"Normal email sayisi: {sum(y == 0)}")

# =============================================================================
# 2. SAMPLING STRATEGY (Train/Test Split)
# =============================================================================
print("\n" + "=" * 60)
print("2. SAMPLING STRATEGY")
print("=" * 60)

# %80 train, %20 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set boyutu: {len(y_train)}")
print(f"Test set boyutu: {len(y_test)}")

# =============================================================================
# 3. FEATURE SELECTION
# =============================================================================
print("\n" + "=" * 60)
print("3. FEATURE SELECTION")
print("=" * 60)

# En iyi 20 feature'i sec (sadece train uzerinden fit)
selector = SelectKBest(score_func=f_classif, k=20)
selector.fit(X_train, y_train)

print(f"Onceki feature sayisi: {X.shape[1]}")
print(f"Sonraki feature sayisi: {selector.get_support().sum()}")

# Secilen feature indexleri
selected_indices = selector.get_support(indices=True)
print(f"Secilen feature indexleri: {selected_indices}")

# =============================================================================
# 4. CLASSIFIERS TANIMLAMA
# =============================================================================
print("\n" + "=" * 60)
print("4. CLASSIFIERS")
print("=" * 60)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

print("Kullanilacak algoritmalar:")
for name in classifiers:
    print(f"  - {name}")

# =============================================================================
# 5. RUN ALGORITHM & 6. GET RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("5-6. ALGORITMALARI CALISTIR VE SONUCLAR")
print("=" * 60)

# Sonuclari sakla
results = {}
cv_scores = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in classifiers.items():
    print(f"\n--- {name} ---")
    
    # Pipeline: feature selection + scaling + model
    pipeline = Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=20)),
        ("scale", StandardScaler()),
        ("model", clf)
    ])
    
    # Modeli egit
    pipeline.fit(X_train, y_train)
    
    # Tahmin yap
    y_pred = pipeline.predict(X_test)
    
    # Metrikleri hesapla
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation (5-fold)
    cv_scores_fold = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring='accuracy'
    )
    
    # Kaydet
    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }
    cv_scores[name] = cv_scores_fold
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"CV Mean:   {cv_scores_fold.mean():.4f} (+/- {cv_scores_fold.std()*2:.4f})")

# =============================================================================
# 7. SONUCLARI KARSILASTIR (TABLO)
# =============================================================================
print("\n" + "=" * 60)
print("7. SONUC TABLOSU")
print("=" * 60)

results_df = pd.DataFrame(results).T
print(results_df.round(4))

# =============================================================================
# 8. STATISTICAL TEST
# =============================================================================
print("\n" + "=" * 60)
print("8. STATISTICAL TEST (Paired t-test)")
print("=" * 60)

# En iyi iki algoritmayi bul
cv_means = {name: scores.mean() for name, scores in cv_scores.items()}
sorted_algs = sorted(cv_means.items(), key=lambda x: x[1], reverse=True)

best_alg = sorted_algs[0][0]
second_alg = sorted_algs[1][0]

print(f"En iyi: {best_alg} (CV Mean: {cv_means[best_alg]:.4f})")
print(f"Ikinci: {second_alg} (CV Mean: {cv_means[second_alg]:.4f})")

# Paired t-test
t_stat, p_value = stats.ttest_rel(cv_scores[best_alg], cv_scores[second_alg])

print(f"\nPaired t-test ({best_alg} vs {second_alg}):")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Sonuc: Fark istatistiksel olarak ANLAMLI (p < 0.05)")
else:
    print("Sonuc: Fark istatistiksel olarak ANLAMLI DEGIL (p >= 0.05)")

# =============================================================================
# 9. GORSELLESTIRME
# =============================================================================
print("\n" + "=" * 60)
print("9. GRAFIKLER OLUSTURULUYOR...")
print("=" * 60)

# Accuracy karsilastirma grafigi
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Graf 1: Accuracy bar chart
alg_names = list(results.keys())
accuracies = [results[name]["Accuracy"] for name in alg_names]

axes[0].bar(alg_names, accuracies, color=['blue', 'green', 'red', 'purple'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Algoritma Karsilastirmasi - Accuracy')
axes[0].set_ylim([0.8, 1.0])

# Graf 2: Tum metrikler
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(alg_names))
width = 0.2

for i, metric in enumerate(metrics):
    values = [results[name][metric] for name in alg_names]
    axes[1].bar(x + i*width, values, width, label=metric)

axes[1].set_ylabel('Score')
axes[1].set_title('Tum Metrikler')
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(alg_names, rotation=15)
axes[1].legend()
axes[1].set_ylim([0.8, 1.0])

plt.tight_layout()
results_path = Path(__file__).resolve().parent / "results.png"
plt.savefig(results_path, dpi=150)
plt.show()

print("\nGrafik kaydedildi: results.png")

# =============================================================================
# FINAL RAPOR
# =============================================================================
print("\n" + "=" * 60)
print("FINAL RAPOR")
print("=" * 60)

print(f"""
Dataset: Spambase (UCI)
- Toplam ornek: 4601
- Feature (secim oncesi): 57
- Feature (secim sonrasi): 20

En Iyi Algoritma: {best_alg}
- Accuracy: {results[best_alg]['Accuracy']:.4f}
- F1-Score: {results[best_alg]['F1-Score']:.4f}

Statistical Test: p-value = {p_value:.4f}
""")

print("PROJE TAMAMLANDI!")
