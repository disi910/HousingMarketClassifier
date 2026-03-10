import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_trading_strategy(df_labeled, predictions, test_indices):
    """Simulate trading strategy based on predictions"""
    
    results = []
    initial_capital = 1000000  # 1M NOK
    
    test_df = df_labeled.iloc[test_indices].copy()
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        if i >= len(predictions):
            break
            
        region = row['region']
        quarter = row['quarter']
        current_price = row['price_index']
        
        # Get actual next quarter price change
        actual_change = row['price_change_next']
        
        # Trading decision based on prediction
        pred = predictions[i]
        if pred == 0:  # Hot prediction
            position = 'LONG'
            profit = actual_change * 10000  # Leverage
        elif pred == 2:  # Cooling prediction
            position = 'SHORT'
            profit = -actual_change * 5000  # Lower leverage for shorts
        else:  # Stable prediction
            position = 'HOLD'
            profit = 0
        
        results.append({
            'region': region,
            'quarter': quarter,
            'prediction': ['Hot', 'Stable', 'Cooling'][pred],
            'position': position,
            'actual_change': actual_change,
            'profit': profit
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate performance metrics
    total_profit = results_df['profit'].sum()
    profitable_trades = (results_df['profit'] > 0).sum()
    total_trades = len(results_df[results_df['position'] != 'HOLD'])
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    avg_profit = results_df['profit'].mean()
    profit_std = results_df['profit'].std()
    sharpe_ratio = avg_profit / profit_std if profit_std > 0 else 0
    
    print("\n=== Backtesting Results ===")
    print(f"Total Profit: {total_profit:,.0f} NOK")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Best Trade: {results_df['profit'].max():,.0f} NOK")
    print(f"Worst Trade: {results_df['profit'].min():,.0f} NOK")
    print(f"Average Profit per Trade: {avg_profit:,.0f} NOK")
    
    # Plot cumulative returns
    results_df['cumulative_profit'] = results_df['profit'].cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['cumulative_profit'])
    plt.title('Cumulative Trading Profit Based on ML Predictions')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Profit (NOK)')
    plt.grid(True)
    plt.show()
    
    return results_df

# Usage example (add to train_model.py)
if __name__ == "__main__":
    # After training your model, run backtest
    test_start = train_size + val_size
    test_indices = list(range(test_start, len(df_labeled)))
    
    backtest_results = backtest_trading_strategy(df_labeled, y_pred_test, test_indices)