import pandas as pd
import numpy as np


def _expanding_z(df, col):
    """Z-score a column against its own past only, per region.

    Standardising against statistics of the whole panel would let information
    from the test years bleed into the training features.
    """
    g = df.groupby('region')[col]
    mean = g.transform(lambda s: s.expanding(min_periods=2).mean())
    std = g.transform(lambda s: s.expanding(min_periods=2).std())
    return ((df[col] - mean) / (std + 1e-9)).fillna(0.0)


def create_enhanced_features(df):
    """Create features from existing and new data columns."""
    df = df.copy()
    df = df.sort_values(['region', 'quarter']).reset_index(drop=True)

    # ── Moving averages and momentum ────────────────────────────
    for col in ['price_index', 'sales_volume']:
        if col in df.columns:
            df[f'{col}_ma2'] = df.groupby('region')[col].transform(
                lambda x: x.rolling(2, min_periods=1).mean()
            )
            df[f'{col}_ma4'] = df.groupby('region')[col].transform(
                lambda x: x.rolling(4, min_periods=1).mean()
            )
            df[f'{col}_momentum'] = df[f'{col}_ma2'] / df[f'{col}_ma4']
            df[f'{col}_momentum'] = df[f'{col}_momentum'].fillna(1.0)

    # ── Price dynamics ──────────────────────────────────────────
    if 'price_index' in df.columns:
        df['price_change'] = df.groupby('region')['price_index'].pct_change()
        df['price_acceleration'] = df.groupby('region')['price_change'].diff()
        df['price_acceleration'] = df['price_acceleration'].fillna(0)
        # Year-over-year price change (4 quarters back)
        df['price_yoy'] = df.groupby('region')['price_index'].pct_change(4)

    # ── Volume/price relationship ───────────────────────────────
    if 'sales_volume' in df.columns and 'price_index' in df.columns:
        df['volume_price_ratio'] = df['sales_volume'] / (df['price_index'] + 1)

    # ── Interest rate dynamics ──────────────────────────────────
    if 'policy_rate' in df.columns:
        df['rate_change'] = df.groupby('region')['policy_rate'].diff().fillna(0)
        if 'price_index' in df.columns:
            df['rate_price_interaction'] = df['policy_rate'] * df['price_index'] / 100

    # ── Regional strength vs national ───────────────────────────
    if 'price_index' in df.columns:
        df['national_avg_price'] = df.groupby('quarter')['price_index'].transform('mean')
        df['regional_strength'] = df['price_index'] / df['national_avg_price']

    # ── Supply/demand imbalance score ───────────────────────────
    if 'population_change' in df.columns and 'sales_volume' in df.columns:
        df['supply_demand_score'] = (
            _expanding_z(df, 'population_change') - _expanding_z(df, 'sales_volume')
        )

    # ── Seasonal factors ────────────────────────────────────────
    if 'quarter_num' in df.columns:
        df['seasonal_factor'] = df['quarter_num'].map({
            1: 0.9, 2: 1.1, 3: 1.05, 4: 0.95
        })

    # ── NEW: Unemployment features ──────────────────────────────
    if 'unemployment_rate' in df.columns:
        df['unemployment_change'] = df.groupby('region')['unemployment_rate'].diff().fillna(0)
        if 'price_index' in df.columns:
            df['unemployment_price_interaction'] = (
                df['unemployment_rate'] * df['price_index'] / 100
            )

    # ── NEW: Building starts / construction pipeline ────────────
    if 'building_starts' in df.columns:
        df['building_starts_ma4'] = df.groupby('region')['building_starts'].transform(
            lambda x: x.rolling(4, min_periods=1).mean()
        )
        df['building_starts_yoy'] = df.groupby('region')['building_starts'].pct_change(4)
        if 'price_index' in df.columns:
            df['construction_price_ratio'] = df['building_starts'] / (df['price_index'] + 1)

    # ── NEW: Mortgage rate features ─────────────────────────────
    if 'mortgage_rate' in df.columns:
        df['mortgage_rate_change'] = df.groupby('region')['mortgage_rate'].diff().fillna(0)
        if 'policy_rate' in df.columns:
            df['mortgage_spread'] = df['mortgage_rate'] - df['policy_rate']

    # ── NEW: Real interest rate ─────────────────────────────────
    if 'mortgage_rate' in df.columns and 'cpi' in df.columns:
        df['cpi_yoy_change'] = df.groupby('region')['cpi'].pct_change(4) * 100
        df['real_interest_rate'] = df['mortgage_rate'] - df['cpi_yoy_change']

    # ── NEW: GDP features ───────────────────────────────────────
    if 'gdp_change' in df.columns:
        df['gdp_ma4'] = df.groupby('region')['gdp_change'].transform(
            lambda x: x.rolling(4, min_periods=1).mean()
        )
        if 'price_index' in df.columns:
            df['gdp_price_interaction'] = df['gdp_change'] * df['price_index'] / 100

    # ── NEW: Affordability ──────────────────────────────────────
    if 'price_index' in df.columns and 'household_income' in df.columns:
        df['affordability_ratio'] = df['price_index'] / (df['household_income'] / 1000 + 1)
        df['affordability_change'] = df.groupby('region')['affordability_ratio'].pct_change(4)

    # ── NEW: Composite demand indicator ─────────────────────────
    if 'population_change' in df.columns and 'unemployment_change' in df.columns:
        df['demand_indicator'] = (
            _expanding_z(df, 'population_change') - _expanding_z(df, 'unemployment_change')
        )

    return df


def create_labels(df, hot_threshold=2.0, cooling_threshold=-0.5):
    """Create risk labels based on next quarter price change.

    Returns the frame sorted by ['quarter', 'region'] so that a positional
    split is chronological. Sorting by region first silently turned the
    intended time split into an alphabetical split by county.
    """
    df = df.sort_values(['region', 'quarter'])
    df['price_next'] = df.groupby('region')['price_index'].shift(-1)
    df['price_change_next'] = (df['price_next'] / df['price_index'] - 1) * 100

    def categorize_risk(change):
        if pd.isna(change):
            return None
        elif change > hot_threshold:
            return 0  # Hot
        elif change < cooling_threshold:
            return 2  # Cooling
        else:
            return 1  # Stable

    df['risk_label'] = df['price_change_next'].apply(categorize_risk)

    # Previous quarter's label, known at prediction time: the persistence baseline.
    df['risk_label_prev'] = df.groupby('region')['risk_label'].shift(1)

    df = df.dropna(subset=['risk_label'])
    return df.sort_values(['quarter', 'region']).reset_index(drop=True)


def get_available_features(df, train_mask=None, min_coverage=0.7):
    """Features present in the frame and well populated over the training window.

    ``min_coverage`` is measured on the training rows only. Columns such as
    ``policy_rate`` and ``household_income`` are absent for large stretches of
    the panel; keeping them and filling the gaps with zeros would tell the model
    that Norway had a 0% policy rate before 2014.
    """
    possible_features = [
        # Original
        'price_index', 'price_index_momentum', 'price_acceleration', 'price_yoy',
        'sales_volume', 'sales_volume_momentum', 'volume_price_ratio',
        'policy_rate', 'rate_change', 'rate_price_interaction',
        'regional_strength', 'supply_demand_score',
        'seasonal_factor', 'quarter_num', 'cpi', 'population_change',
        # New
        'unemployment_rate', 'unemployment_change', 'unemployment_price_interaction',
        'building_starts', 'building_starts_ma4', 'building_starts_yoy',
        'construction_price_ratio',
        'mortgage_rate', 'mortgage_rate_change', 'mortgage_spread',
        'real_interest_rate', 'cpi_yoy_change',
        'gdp_change', 'gdp_ma4', 'gdp_price_interaction',
        'household_income', 'affordability_ratio', 'affordability_change',
        'demand_indicator',
    ]

    present = [f for f in possible_features if f in df.columns]
    if train_mask is None:
        train_mask = pd.Series(True, index=df.index)

    train_df = df.loc[train_mask]
    available, dropped = [], []
    for f in present:
        coverage = train_df[f].notna().mean()
        (available if coverage >= min_coverage else dropped).append((f, coverage))

    print(f"Features: {len(available)} kept, {len(dropped)} dropped "
          f"(<{min_coverage:.0%} coverage in training window)")
    if dropped:
        print("  dropped: " + ", ".join(f"{f} ({c:.0%})" for f, c in dropped))
    return [f for f, _ in available]
