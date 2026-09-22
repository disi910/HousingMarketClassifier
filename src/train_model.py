"""
Train and evaluate housing market risk classifiers using sklearn.

The split is chronological: whole quarters are assigned to train/val/test, so
no quarter ever appears in two sets. An earlier version sliced the frame by row
position after sorting on ['region', 'quarter'], which split the panel by county
in alphabetical order instead of by time. Because several counties share an SSB
price region (and therefore share labels), and because national series such as
CPI and the policy rate act as a per-quarter fingerprint, that split leaked
heavily and inflated the reported scores.
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from enhanced_features import create_enhanced_features, create_labels, get_available_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

CLASS_NAMES = ['Hot', 'Stable', 'Cooling']
CLASS_LABELS = [0, 1, 2]

# Interest-rate and mortgage series only start in 2014. Training on the full
# panel means either dropping them or inventing values for a decade.
DEFAULT_MIN_YEAR = 2014


def chronological_split(df, train_frac=0.65, val_frac=0.20):
    """Assign whole quarters to train/val/test in time order.

    Every county's rows for a given quarter land in the same set, so the model
    is always evaluated on quarters it has never seen.
    """
    quarters = np.sort(df['quarter'].unique())  # "2005K1" strings sort correctly
    n_train = int(train_frac * len(quarters))
    n_val = int((train_frac + val_frac) * len(quarters))

    q_train, q_val = quarters[:n_train], quarters[n_train:n_val]
    train = df['quarter'].isin(q_train).values
    val = df['quarter'].isin(q_val).values
    return train, val, ~(train | val), (q_train, q_val, quarters[n_val:])


def evaluate(y_true, y_pred):
    return {
        'accuracy': float(np.mean(y_pred == y_true)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-year', type=int, default=DEFAULT_MIN_YEAR,
                        help='First year to include (default: %(default)s)')
    args = parser.parse_args()

    # ── Load & prepare ──────────────────────────────────────────────
    df = pd.read_csv(OUTPUT_DIR / 'processed_data.csv')
    df_labeled = create_labels(create_enhanced_features(df))
    df_labeled = df_labeled[df_labeled['year'] >= args.min_year].reset_index(drop=True)

    tr, va, te, (q_train, q_val, q_test) = chronological_split(df_labeled)

    features = get_available_features(df_labeled, train_mask=tr)
    X = df_labeled[features].replace([np.inf, -np.inf], np.nan).values
    y = df_labeled['risk_label'].astype(int).values

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{df_labeled['region'].nunique()} counties, years {args.min_year}+")
    print(f"Classes: Hot={np.sum(y==0)}, Stable={np.sum(y==1)}, Cooling={np.sum(y==2)}")
    print(f"Split by quarter (no quarter in two sets):")
    for name, qs, mask in [('train', q_train, tr), ('val', q_val, va), ('test', q_test, te)]:
        print(f"  {name:5s} {qs[0]}..{qs[-1]}  {len(qs):2d} quarters, {mask.sum():4d} rows")

    y_train, y_val, y_test = y[tr], y[va], y[te]

    # Impute and scale using training statistics only.
    imputer = SimpleImputer(strategy='median').fit(X[tr])
    scaler = StandardScaler().fit(imputer.transform(X[tr]))
    prep = lambda m: scaler.transform(imputer.transform(X[m]))
    X_train_s, X_val_s, X_test_s = prep(tr), prep(va), prep(te)

    # ── Baselines ───────────────────────────────────────────────────
    # A score means nothing without something to beat.
    majority = int(pd.Series(y_train).mode()[0])
    baselines = {
        'Majority class': {
            'val': evaluate(y_val, np.full_like(y_val, majority)),
            'test': evaluate(y_test, np.full_like(y_test, majority)),
        },
    }
    # Persistence: reuse the previous quarter's label, which is known at
    # prediction time. Labels are strongly autocorrelated, so this is the bar.
    prev = df_labeled['risk_label_prev'].fillna(majority).astype(int).values
    baselines['Persistence'] = {
        'val': evaluate(y_val, prev[va]),
        'test': evaluate(y_test, prev[te]),
    }
    # Seasonal lookup: the most common label for this quarter-of-year in the
    # training window, ignoring every macro series. Norwegian house prices have
    # a strong within-year cycle, so this is the bar that actually matters.
    seasonal_map = df_labeled[tr].groupby('quarter_num')['risk_label'].agg(
        lambda s: s.mode()[0]).astype(int)
    seasonal = df_labeled['quarter_num'].map(seasonal_map).fillna(majority).astype(int).values
    baselines['Seasonal lookup'] = {
        'val': evaluate(y_val, seasonal[va]),
        'test': evaluate(y_test, seasonal[te]),
    }

    print("\nBaselines:")
    for name, scores in baselines.items():
        print(f"  {name:16s} val_acc={scores['val']['accuracy']:.1%} "
              f"val_f1={scores['val']['macro_f1']:.3f} | "
              f"test_acc={scores['test']['accuracy']:.1%} "
              f"test_f1={scores['test']['macro_f1']:.3f}")

    # ── Train models ────────────────────────────────────────────────
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

    model_scores = {}
    best_model_name, best_model, best_f1 = None, None, -1
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        scores = evaluate(y_val, model.predict(X_val_s))
        model_scores[name] = scores
        print(f"\n{name}: val_accuracy={scores['accuracy']:.1%}, "
              f"val_macro_f1={scores['macro_f1']:.3f}")
        if scores['macro_f1'] > best_f1:
            best_f1, best_model_name, best_model = scores['macro_f1'], name, model

    print(f"\nBest model: {best_model_name} (val macro-F1={best_f1:.3f})")

    # ── Evaluate best model ────────────────────────────────────────
    y_pred_val = best_model.predict(X_val_s)
    y_pred_test = best_model.predict(X_test_s)

    for title, y_true, y_pred in [('VALIDATION', y_val, y_pred_val),
                                  ('TEST', y_test, y_pred_test)]:
        print(f"\n{'='*50}\n{title} ({best_model_name})\n{'='*50}")
        print(classification_report(y_true, y_pred, labels=CLASS_LABELS,
                                    target_names=CLASS_NAMES, zero_division=0))

    val_scores, test_scores = evaluate(y_val, y_pred_val), evaluate(y_test, y_pred_test)
    print(f"Final  |  Val accuracy: {val_scores['accuracy']:.1%}  "
          f"|  Test accuracy: {test_scores['accuracy']:.1%}  "
          f"|  Test macro-F1: {test_scores['macro_f1']:.3f}")
    best_baseline = max(baselines, key=lambda b: baselines[b]['test']['macro_f1'])
    gap = test_scores['macro_f1'] - baselines[best_baseline]['test']['macro_f1']
    print(f"\nStrongest baseline on test: {best_baseline} "
          f"(macro-F1={baselines[best_baseline]['test']['macro_f1']:.3f})")
    print(f"Model minus strongest baseline: {gap:+.3f} macro-F1")
    if gap <= 0:
        print("  The model does NOT beat the baseline. The macro features are not "
              "adding signal over the seasonal cycle.")

    # ── Confusion matrix ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (y_true, y_pred, title, qs) in zip(axes, [
        (y_val, y_pred_val, 'Validation', q_val),
        (y_test, y_pred_test, 'Test', q_test),
    ]):
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f'{title} Confusion Matrix ({qs[0]}–{qs[-1]})')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    plt.suptitle(f'{best_model_name} — Housing Market Risk Classifier '
                 f'(chronological split)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Feature importance ──────────────────────────────────────────
    importances = getattr(best_model, 'feature_importances_', np.zeros(len(features)))
    imp_df = pd.DataFrame({'feature': features, 'importance': importances}) \
        .sort_values('importance', ascending=False)

    print("\nTop 10 features:")
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

    # ── Persist results so the README can't drift from the code ────
    results = {
        'min_year': args.min_year,
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'split': {name: {'quarters': [qs[0], qs[-1]], 'n_quarters': len(qs),
                         'n_rows': int(mask.sum())}
                  for name, qs, mask in [('train', q_train, tr), ('val', q_val, va),
                                         ('test', q_test, te)]},
        'baselines': baselines,
        'models': model_scores,
        'best_model': best_model_name,
        'best': {'val': val_scores, 'test': test_scores},
        'best_baseline': best_baseline,
        'model_minus_baseline_test_f1': gap,
        'per_class_test': classification_report(
            y_test, y_pred_test, labels=CLASS_LABELS, target_names=CLASS_NAMES,
            zero_division=0, output_dict=True),
        'top_features': imp_df.head(10).to_dict('records'),
    }
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nSaved: confusion_matrix.png, feature_importance.png, results.json")


if __name__ == '__main__':
    main()
