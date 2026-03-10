import pandas as pd
import numpy as np

def create_enhanced_features(df):
    """Create powerful features from existing data"""
    df = df.copy()
    df = df.sort_values(['region', 'quarter'])
    
    # Moving averages and momentum
    for col in ['price_index', 'sales_volume']:
        if col in df.columns:
            df[f'{col}_ma2'] = df.groupby('region')[col].transform(
                lambda x: x.rolling(2, min_periods=1).mean()
            )
            df[f'{col}_ma4'] = df.groupby('region')[col].transform(
                lambda x: x.rolling(4, min_periods=1).mean()
            )
            # Momentum (short/long term ratio)
            df[f'{col}_momentum'] = df[f'{col}_ma2'] / df[f'{col}_ma4']
            df[f'{col}_momentum'] = df[f'{col}_momentum'].fillna(1.0)
    
    # Price acceleration (2nd derivative)
    if 'price_index' in df.columns:
        df['price_change'] = df.groupby('region')['price_index'].pct_change()
        df['price_acceleration'] = df.groupby('region')['price_change'].diff()
        df['price_acceleration'] = df['price_acceleration'].fillna(0)
    
    # Volume/price relationship
    if 'sales_volume' in df.columns and 'price_index' in df.columns:
        df['volume_price_ratio'] = df['sales_volume'] / (df['price_index'] + 1)
    
    # Interest rate dynamics
    if 'policy_rate' in df.columns:
        df['rate_change'] = df['policy_rate'].diff().fillna(0)
        if 'price_index' in df.columns:
            df['rate_price_interaction'] = df['policy_rate'] * df['price_index'] / 100
    
    # Regional strength vs national
    if 'price_index' in df.columns:
        df['national_avg_price'] = df.groupby('quarter')['price_index'].transform('mean')
        df['regional_strength'] = df['price_index'] / df['national_avg_price']
    
    # Supply/demand imbalance score
    if 'population_change' in df.columns and 'sales_volume' in df.columns:
        pop_std = df['population_change'].std()
        vol_std = df['sales_volume'].std()
        df['supply_demand_score'] = (
            df['population_change'] / (pop_std if pop_std > 0 else 1) - 
            df['sales_volume'] / (vol_std if vol_std > 0 else 1)
        )
    
    # Seasonal factors
    if 'quarter_num' in df.columns:
        df['seasonal_factor'] = df['quarter_num'].map({
            1: 0.9, 2: 1.1, 3: 1.05, 4: 0.95
        })
    
    return df

def create_labels(df):
    """Create risk labels based on next quarter price change"""
    df = df.sort_values(['region', 'quarter'])
    df['price_next'] = df.groupby('region')['price_index'].shift(-1)
    df['price_change_next'] = (df['price_next'] / df['price_index'] - 1) * 100
    
    def categorize_risk(change):
        if pd.isna(change):
            return None
        elif change > 2:
            return 0  # Hot
        elif change < -0.5:
            return 2  # Cooling
        else:
            return 1  # Stable
    
    df['risk_label'] = df['price_change_next'].apply(categorize_risk)
    return df.dropna(subset=['risk_label'])

def get_available_features(df):
    """Get list of features that exist in the dataframe"""
    possible_features = [
        'price_index', 'price_index_momentum', 'price_acceleration',
        'sales_volume', 'sales_volume_momentum', 'volume_price_ratio',
        'policy_rate', 'rate_change', 'rate_price_interaction',
        'regional_strength', 'supply_demand_score',
        'seasonal_factor', 'quarter_num', 'cpi', 'population_change'
    ]
    
    available = [f for f in possible_features if f in df.columns]
    print(f"Available features: {available}")
    return available