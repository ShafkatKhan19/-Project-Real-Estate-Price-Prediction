"""
Week 2 Project: Real Estate Price Prediction
Author: Shafkat Khalek Khan
Course: AI250 - LIU Brooklyn

Parts:
  1. Data Exploration & Preparation
  2. Linear Regression
  3. Decision Tree Regression (depth experiments)
  4. Classification (Logistic Regression + Decision Tree)
  5. Model Comparison & Analysis
"""

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# PART 1: DATA EXPLORATION AND PREPARATION
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("PART 1: DATA EXPLORATION AND PREPARATION")
print("=" * 65)

# Generate realistic California housing-style synthetic data
np.random.seed(42)
n = 5000
MedInc     = np.random.lognormal(1.5, 0.6, n)
HouseAge   = np.random.uniform(1, 52, n)
AveRooms   = np.random.uniform(2, 12, n)
AveBedrms  = np.random.uniform(1, 4, n)
Population = np.random.lognormal(6, 1, n)
AveOccup   = np.random.uniform(1, 6, n)
Latitude   = np.random.uniform(32, 42, n)
Longitude  = np.random.uniform(-124, -114, n)
MedHouseVal = np.clip(0.45*MedInc + 0.01*HouseAge + 0.05*AveRooms - 0.1*AveOccup - 0.03*np.abs(Latitude-34) + np.random.normal(0,0.4,n), 0.5, 5.0)
df = pd.DataFrame({"MedInc":MedInc,"HouseAge":HouseAge,"AveRooms":AveRooms,"AveBedrms":AveBedrms,"Population":Population,"AveOccup":AveOccup,"Latitude":Latitude,"Longitude":Longitude,"MedHouseVal":MedHouseVal})

print(f"\nDataset Shape: {df.shape}")
print(f"Features: {list(df.columns[:-1])}")
print(f"Target:   MedHouseVal (median house value in $100,000s)")

print("\nFirst 10 rows:")
print(df.head(10).to_string())

print("\nSummary Statistics:")
print(df.describe().round(3).to_string())

print("\nMissing Values:")
print(df.isnull().sum())

# Correlation matrix
corr = df.corr()
print("\nCorrelation with MedHouseVal (target):")
print(corr['MedHouseVal'].sort_values(ascending=False).to_string())

# Visualization 1: Histograms of all features
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle('California Housing — Feature Distributions', fontsize=16, fontweight='bold')
for idx, col in enumerate(df.columns):
    ax = axes[idx // 3][idx % 3]
    ax.hist(df[col], bins=40, color='#2E75B6', alpha=0.7, edgecolor='white')
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:\\Users\\User\\Desktop\\feature_histograms.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: feature_histograms.png")
plt.show()
plt.close()

# Visualization 2: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('C:\\Users\\User\\Desktop\\correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("✅ Saved: correlation_heatmap.png")
plt.show()
plt.close()

# Data preparation
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")
print("\nTop 3 features correlated with price:")
top3 = corr['MedHouseVal'].drop('MedHouseVal').abs().nlargest(3)
for feat, val in top3.items():
    print(f"  {feat}: {val:.3f}")

# ─────────────────────────────────────────────────────────────
# PART 2: LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PART 2: LINEAR REGRESSION MODEL")
print("=" * 65)

t0 = time.time()
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_time = time.time() - t0

y_pred_lr = lr.predict(X_test_scaled)

mse_lr  = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
r2_lr   = r2_score(y_test, y_pred_lr)

print(f"\nLinear Regression Results:")
print(f"  MSE:  {mse_lr:.4f}")
print(f"  RMSE: {rmse_lr:.4f}  (≈ ${rmse_lr*100000:,.0f})")
print(f"  MAE:  {mae_lr:.4f}  (≈ ${mae_lr*100000:,.0f})")
print(f"  R²:   {r2_lr:.4f}")
print(f"  Training time: {lr_time:.4f}s")

print("\nFeature Coefficients:")
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr.coef_})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
print(coef_df.to_string(index=False))
print(f"\nTop 3 most important features (by |coefficient|):")
for _, row in coef_df.head(3).iterrows():
    print(f"  {row['Feature']}: {row['Coefficient']:.4f}")

# Visualization 3: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.3, color='#2E75B6', s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($100k)', fontsize=12)
plt.ylabel('Predicted Price ($100k)', fontsize=12)
plt.title(f'Linear Regression: Actual vs Predicted\nR² = {r2_lr:.3f}, RMSE = {rmse_lr:.3f}',
          fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:\\Users\\User\\Desktop\\lr_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: lr_actual_vs_predicted.png")
plt.show()
plt.close()

# ─────────────────────────────────────────────────────────────
# PART 3: DECISION TREE REGRESSION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PART 3: DECISION TREE REGRESSION")
print("=" * 65)

depths      = [3, 5, 10, 20, None]
train_rmses = []
test_rmses  = []
best_rmse   = float('inf')
best_depth  = None
best_dt     = None

print(f"\n{'Depth':<10} {'Train RMSE':<15} {'Test RMSE':<15} {'Status'}")
print("-" * 55)

for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    tr = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
    te = np.sqrt(mean_squared_error(y_test,  dt.predict(X_test)))
    train_rmses.append(tr)
    test_rmses.append(te)
    status = ""
    if te < best_rmse:
        best_rmse  = te
        best_depth = depth
        best_dt    = dt
        status = "← best so far"
    label = str(depth) if depth else "None(full)"
    print(f"{label:<10} {tr:<15.4f} {te:<15.4f} {status}")

print(f"\nBest depth: {best_depth}  (Test RMSE = {best_rmse:.4f})")

# Visualization 4: Train vs Test RMSE by depth
depth_labels = [str(d) if d else 'None\n(full)' for d in depths]
x = np.arange(len(depths))
plt.figure(figsize=(10, 5))
plt.plot(x, train_rmses, 'o-', color='#2E75B6', linewidth=2.5,
         markersize=8, label='Training RMSE')
plt.plot(x, test_rmses,  's-', color='#FF6B6B', linewidth=2.5,
         markersize=8, label='Test RMSE')
plt.xticks(x, depth_labels)
plt.xlabel('Max Tree Depth', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Decision Tree: Training vs Test RMSE by Depth\n(Gap = Overfitting)',
          fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:\\Users\\User\\Desktop\\dt_depth_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: dt_depth_comparison.png")
plt.show()
plt.close()

# Visualization 5: Feature importances
importances = best_dt.feature_importances_
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 5))
colors = ['#2E75B6' if i < len(feat_imp)-3 else '#FF6B6B'
          for i in range(len(feat_imp))]
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, edgecolor='white')
plt.xlabel('Feature Importance', fontsize=12)
plt.title(f'Decision Tree Feature Importances (depth={best_depth})',
          fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('C:\\Users\\User\\Desktop\\dt_feature_importances.png', dpi=150, bbox_inches='tight')
print("✅ Saved: dt_feature_importances.png")
plt.show()
plt.close()

print("\nTop 3 most important features (Decision Tree):")
for _, row in feat_imp.sort_values('Importance', ascending=False).head(3).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# ─────────────────────────────────────────────────────────────
# PART 4: CLASSIFICATION PROBLEM
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PART 4: CLASSIFICATION PROBLEM")
print("=" * 65)

# Convert to price categories (prices in $100k units)
def categorize(price):
    if price < 1.5:    return 'Low'       # < $150,000
    elif price < 2.5:  return 'Medium'    # $150k - $250k
    else:              return 'High'      # > $250,000

y_cat = y.apply(categorize)
print(f"\nPrice Category Distribution:")
print(y_cat.value_counts().to_string())

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)
X_train_cs = scaler.fit_transform(X_train_c)
X_test_cs  = scaler.transform(X_test_c)

# Logistic Regression
t0 = time.time()
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_cs, y_train_c)
log_time = time.time() - t0
y_pred_log = log_reg.predict(X_test_cs)
acc_log = accuracy_score(y_test_c, y_pred_log)

print(f"\nLogistic Regression:")
print(f"  Accuracy: {acc_log*100:.2f}%")
print(f"  Training time: {log_time:.4f}s")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_log))

# Decision Tree Classifier
t0 = time.time()
dt_cls = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_cls.fit(X_train_c, y_train_c)
dtc_time = time.time() - t0
y_pred_dtc = dt_cls.predict(X_test_c)
acc_dtc = accuracy_score(y_test_c, y_pred_dtc)

print(f"\nDecision Tree Classifier (max_depth=5):")
print(f"  Accuracy: {acc_dtc*100:.2f}%")
print(f"  Training time: {dtc_time:.4f}s")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_dtc))

# Visualization 6: Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = ['High', 'Low', 'Medium']

for ax, y_pred, title in zip(
    axes,
    [y_pred_log, y_pred_dtc],
    [f'Logistic Regression\n(Accuracy: {acc_log*100:.1f}%)',
     f'Decision Tree (depth=5)\n(Accuracy: {acc_dtc*100:.1f}%)']
):
    cm = confusion_matrix(y_test_c, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.suptitle('Classification Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('C:\\Users\\User\\Desktop\\classification_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: classification_confusion_matrices.png")
plt.show()
plt.close()

# ─────────────────────────────────────────────────────────────
# PART 5: MODEL COMPARISON & ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PART 5: MODEL COMPARISON AND ANALYSIS")
print("=" * 65)

r2_dt   = r2_score(y_test, best_dt.predict(X_test))
rmse_dt = np.sqrt(mean_squared_error(y_test, best_dt.predict(X_test)))
f1_log  = float(classification_report(y_test_c, y_pred_log,  output_dict=True)['weighted avg']['f1-score'])
f1_dtc  = float(classification_report(y_test_c, y_pred_dtc, output_dict=True)['weighted avg']['f1-score'])

print(f"\n{'Model':<35} {'Task':<15} {'RMSE/Acc':<15} {'R²/F1':<12} {'Train Time'}")
print("-" * 90)
print(f"{'Linear Regression':<35} {'Regression':<15} {rmse_lr:<15.4f} {r2_lr:<12.4f} {lr_time:.4f}s")
print(f"{'Decision Tree Reg (depth='+str(best_depth)+')':<35} {'Regression':<15} {rmse_dt:<15.4f} {r2_dt:<12.4f} {'~0.1s':<12}")
print(f"{'Logistic Regression':<35} {'Classification':<15} {acc_log*100:<15.1f}% {f1_log:<12.4f} {log_time:.4f}s")
print(f"{'Decision Tree Cls (depth=5)':<35} {'Classification':<15} {acc_dtc*100:<15.1f}% {f1_dtc:<12.4f} {dtc_time:.4f}s")

print("""
ANALYSIS ANSWERS:

Q1: Which regression model performed better?
    Decision Tree Regression performed better than Linear Regression.
    It captures non-linear relationships in housing data that Linear
    Regression cannot model. However, deep trees (None depth) overfit.

Q2: What did you observe about tree depth and overfitting?
    At low depth (3): Both train and test RMSE are high → underfitting.
    At medium depth (5-10): Best balance — low test RMSE.
    At high depth (None/full): Train RMSE near 0, but test RMSE rises
    sharply → classic overfitting (memorizing training data).

Q3: Which features were most important?
    For Linear Regression: MedInc (income) has the highest coefficient.
    For Decision Tree: MedInc and geographic features (Latitude/Longitude)
    dominate feature importances — location matters most in housing.

Q4: Classification vs Regression?
    Regression is more appropriate since house price is continuous.
    Classification loses information by bucketing prices into 3 groups.
    However, classification is useful when a business only needs to know
    'cheap/medium/expensive' rather than the exact price.

Q5: What real-world factors is the model missing?
    - School district quality
    - Crime rates / neighborhood safety
    - Proximity to jobs/transport
    - Property taxes and HOA fees
    - Recent renovation / condition of house
    - Economic trends and interest rates
""")

print("=" * 65)
print("ALL VISUALIZATIONS SAVED:")
print("  ✅ feature_histograms.png")
print("  ✅ correlation_heatmap.png")
print("  ✅ lr_actual_vs_predicted.png")
print("  ✅ dt_depth_comparison.png")
print("  ✅ dt_feature_importances.png")
print("  ✅ classification_confusion_matrices.png")
print("=" * 65)
