import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from random_forest_classifier import RandomForestClassifier
from enhanced_features import create_enhanced_features, create_labels, get_available_features

# Load and prepare data
df = pd.read_csv('processed/processed_data.csv')
print(f"Loaded {len(df)} rows of data")

# Create enhanced features
df_enhanced = create_enhanced_features(df)
print(f"Enhanced features created")

# Create labels
df_labeled = create_labels(df_enhanced)
print(f"Labels created: {len(df_labeled)} samples")

# Get available features
available_features = get_available_features(df_labeled)

# Prepare feature matrix and labels
X = df_labeled[available_features].fillna(0).values
y = df_labeled['risk_label'].astype(int).values

print(f"Features shape: {X.shape}")
print(f"Labels distribution:\n{pd.Series(y).value_counts()}")

# Time-based split (crucial for time series!)
train_size = int(0.65 * len(X))
val_size = int(0.20 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"Training: {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_trees=50, max_depth=4, max_features='sqrt')
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred_val = rf.predict(X_val_scaled)
y_pred_test = rf.predict(X_test_scaled)

# Results
print("\n=== Validation Results ===")
print(classification_report(y_val, y_pred_val, 
                          target_names=['Hot', 'Stable', 'Cooling']))

print("\n=== Test Results ===")
print(classification_report(y_test, y_pred_test, 
                          target_names=['Hot', 'Stable', 'Cooling']))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred_val)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Hot', 'Stable', 'Cooling'],
            yticklabels=['Hot', 'Stable', 'Cooling'])
plt.title('Confusion Matrix - Property Market Risk Classifier')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# Feature importance analysis
def calculate_feature_importance(model, X, y, feature_names):
    baseline_score = np.mean(model.predict(X) == y)
    importances = []
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])
        permuted_score = np.mean(model.predict(X_permuted) == y)
        importance = baseline_score - permuted_score
        importances.append(max(0, importance))
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df

importance_df = calculate_feature_importance(rf, X_val_scaled, y_val, available_features)
print("\n=== Feature Importance ===")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
top_features = importance_df.head(8)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance (Accuracy Drop)')
plt.title('Feature Importance for Market Risk Prediction')
plt.tight_layout()
plt.show()

print(f"Final Validation Accuracy: {np.mean(y_pred_val == y_val):.2%}")
print(f"Final Test Accuracy: {np.mean(y_pred_test == y_test):.2%}")