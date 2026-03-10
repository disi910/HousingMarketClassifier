"""
Train and evaluate housing market risk classifiers using sklearn.
Compares RandomForest and GradientBoosting, saves best model artifacts.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from enhanced_features import create_enhanced_features, create_labels, get_available_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# ── Load & prepare ──────────────────────────────────────────────
df = pd.read_csv(OUTPUT_DIR / 'processed_data.csv')
df_enhanced = create_enhanced_features(df)
df_labeled = create_labels(df_enhanced)
available_features = get_available_features(df_labeled)

X = df_labeled[available_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
y = df_labeled['risk_label'].astype(int).values

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: Hot={np.sum(y==0)}, Stable={np.sum(y==1)}, Cooling={np.sum(y==2)}")

# ── Time-based split ────────────────────────────────────────────
train_end = int(0.65 * len(X))
val_end = train_end + int(0.20 * len(X))

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ── Train models ────────────────────────────────────────────────
CLASS_NAMES = ['Hot', 'Stable', 'Cooling']
CLASS_LABELS = [0, 1, 2]

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=10, subsample=0.8, random_state=42
    ),
}

best_model_name, best_model, best_f1 = None, None, -1

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_val_s)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    acc = np.mean(y_pred == y_val)
    print(f"\n{name}: val_accuracy={acc:.1%}, val_macro_f1={f1:.3f}")
    if f1 > best_f1:
        best_f1, best_model_name, best_model = f1, name, model

print(f"\nBest model: {best_model_name} (macro-F1={best_f1:.3f})")

# ── Evaluate best model ────────────────────────────────────────
y_pred_val = best_model.predict(X_val_s)
y_pred_test = best_model.predict(X_test_s)

print(f"\n{'='*50}")
print(f"VALIDATION ({best_model_name})")
print(f"{'='*50}")
print(classification_report(y_val, y_pred_val,
                            labels=CLASS_LABELS, target_names=CLASS_NAMES,
                            zero_division=0))

print(f"{'='*50}")
print(f"TEST ({best_model_name})")
print(f"{'='*50}")
print(classification_report(y_test, y_pred_test,
                            labels=CLASS_LABELS, target_names=CLASS_NAMES,
                            zero_division=0))

val_acc = np.mean(y_pred_val == y_val)
test_acc = np.mean(y_pred_test == y_test)
test_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)

print(f"Final  |  Val accuracy: {val_acc:.1%}  |  Test accuracy: {test_acc:.1%}  |  Test macro-F1: {test_f1:.3f}")

# ── Confusion matrix ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (y_true, y_pred, title) in zip(axes, [
    (y_val, y_pred_val, 'Validation'),
    (y_test, y_pred_test, 'Test'),
]):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f'{title} Confusion Matrix')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

plt.suptitle(f'{best_model_name} — Housing Market Risk Classifier', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Feature importance ──────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    importances = np.zeros(len(available_features))

imp_df = pd.DataFrame({
    'feature': available_features,
    'importance': importances,
}).sort_values('importance', ascending=False)

print(f"\nTop 10 features:")
for _, row in imp_df.head(10).iterrows():
    print(f"  {row['feature']:35s} {row['importance']:.4f}")

plt.figure(figsize=(10, 7))
top = imp_df.head(15)
plt.barh(top['feature'][::-1], top['importance'][::-1])
plt.xlabel('Feature Importance (Gini)')
plt.title(f'Top 15 Features — {best_model_name}')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved: confusion_matrix.png, feature_importance.png")
