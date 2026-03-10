import pandas as pd
import numpy as np


def create_enhanced_features(df):
    """Create features from existing and new data columns."""
    df = df.copy()
    df = df.sort_values(['region', 'quarter'])

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
        df['rate_change'] = df['policy_rate'].diff().fillna(0)
        if 'price_index' in df.columns:
            df['rate_price_interaction'] = df['policy_rate'] * df['price_index'] / 100

    # ── Regional strength vs national ───────────────────────────
    if 'price_index' in df.columns:
        df['national_avg_price'] = df.groupby('quarter')['price_index'].transform('mean')
        df['regional_strength'] = df['price_index'] / df['national_avg_price']

    # ── Supply/demand imbalance score ───────────────────────────
    if 'population_change' in df.columns and 'sales_volume' in df.columns:
        pop_std = df['population_change'].std()
        vol_std = df['sales_volume'].std()
        df['supply_demand_score'] = (
            df['population_change'] / (pop_std if pop_std > 0 else 1) -
            df['sales_volume'] / (vol_std if vol_std > 0 else 1)
        )

    # ── Seasonal factors ────────────────────────────────────────
    if 'quarter_num' in df.columns:
        df['seasonal_factor'] = df['quarter_num'].map({
            1: 0.9, 2: 1.1, 3: 1.05, 4: 0.95
        })

    # ── NEW: Unemployment features ──────────────────────────────
    if 'unemployment_rate' in df.columns:
        df['unemployment_change'] = df['unemployment_rate'].diff().fillna(0)
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
        df['mortgage_rate_change'] = df['mortgage_rate'].diff().fillna(0)
        if 'policy_rate' in df.columns:
            df['mortgage_spread'] = df['mortgage_rate'] - df['policy_rate']

    # ── NEW: Real interest rate ─────────────────────────────────
    if 'mortgage_rate' in df.columns and 'cpi' in df.columns:
        df['cpi_yoy_change'] = df.groupby('region')['cpi'].pct_change(4) * 100
        df['real_interest_rate'] = df['mortgage_rate'] - df['cpi_yoy_change']

    # ── NEW: GDP features ───────────────────────────────────────
    if 'gdp_change' in df.columns:
        df['gdp_ma4'] = df['gdp_change'].rolling(4, min_periods=1).mean()
        if 'price_index' in df.columns:
            df['gdp_price_interaction'] = df['gdp_change'] * df['price_index'] / 100

    # ── NEW: Affordability ──────────────────────────────────────
    if 'price_index' in df.columns and 'household_income' in df.columns:
        df['affordability_ratio'] = df['price_index'] / (df['household_income'] / 1000 + 1)
        df['affordability_change'] = df.groupby('region')['affordability_ratio'].pct_change(4)

    # ── NEW: Composite demand indicator ─────────────────────────
    if 'population_change' in df.columns and 'unemployment_rate' in df.columns:
        pop_z = (df['population_change'] - df['population_change'].mean()) / (df['population_change'].std() + 1e-9)
        unemp_z = (df['unemployment_change'] - df['unemployment_change'].mean()) / (df['unemployment_change'].std() + 1e-9) if 'unemployment_change' in df.columns else 0
        df['demand_indicator'] = pop_z - unemp_z

    return df


def create_labels(df, hot_threshold=2.0, cooling_threshold=-0.5):
    """Create risk labels based on next quarter price change."""
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
    return df.dropna(subset=['risk_label'])


def get_available_features(df):
    """Get list of features that exist in the dataframe."""
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

    available = [f for f in possible_features if f in df.columns and df[f].notna().sum() > 0]
    print(f"Features: {len(available)} available")
    return available
