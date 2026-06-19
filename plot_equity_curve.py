import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

INITIAL_CAPITAL = 10000.0


def simulate_strategy(predictions, prices, initial_capital=INITIAL_CAPITAL):
    capital = initial_capital
    shares = 0.0
    in_position = False
    capital_history = []
    trade_profits = []

    for i in range(len(predictions)):
        pred = predictions[i]
        price = prices[i]

        if pred == 1 and not in_position:
            shares = capital / price
            entry_price = price
            in_position = True
        elif pred == 2 and in_position:
            capital = shares * price
            profit_pct = (price - entry_price) / entry_price * 100
            trade_profits.append(profit_pct)
            shares = 0.0
            in_position = False

        if in_position:
            capital_history.append(shares * price)
        else:
            capital_history.append(capital)

    if in_position:
        capital = shares * prices[-1]
        trade_profits.append((prices[-1] - entry_price) / entry_price * 100)

    return np.array(capital_history), capital, trade_profits


def simulate_bah(prices, initial_capital=INITIAL_CAPITAL):
    shares = initial_capital / prices[0]
    capital_history = shares * prices
    final_capital = shares * prices[-1]
    return capital_history, final_capital


def calc_max_drawdown(capital_history):
    peak = np.maximum.accumulate(capital_history)
    drawdown = (capital_history - peak) / peak
    return drawdown.min() * 100


def calc_sharpe_ratio(capital_history, annual_factor=252):
    daily_returns = np.diff(capital_history) / capital_history[:-1]
    if len(daily_returns) == 0 or daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(annual_factor)


def calc_idle_ratio(predictions):
    in_position = False
    idle_days = 0
    for pred in predictions:
        if pred == 1 and not in_position:
            in_position = True
        elif pred == 2 and in_position:
            in_position = False
        if not in_position:
            idle_days += 1
    return idle_days / len(predictions) * 100


def print_stats(name, capital_history, final_capital, trade_profits, predictions, prices):
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    max_dd = calc_max_drawdown(capital_history)
    sharpe = calc_sharpe_ratio(capital_history)
    idle = calc_idle_ratio(predictions)
    n_trades = len(trade_profits)
    win_trades = [p for p in trade_profits if p > 0]
    loss_trades = [p for p in trade_profits if p <= 0]
    win_rate = len(win_trades) / n_trades * 100 if n_trades > 0 else 0
    avg_profit = np.mean(trade_profits) if trade_profits else 0
    max_profit = max(trade_profits) if trade_profits else 0
    max_loss = min(trade_profits) if trade_profits else 0
    avg_len = len(predictions) / n_trades if n_trades > 0 else 0

    print(f"\n{'='*55}")
    print(f"  {name} 回测统计")
    print(f"{'='*55}")
    print(f"  初始资金:       ${INITIAL_CAPITAL:.2f}")
    print(f"  最终资金:       ${final_capital:.2f}")
    print(f"  总收益率:       {total_return:.2f}%")
    print(f"  最大回撤:       {max_dd:.2f}%")
    print(f"  Sharpe比率:     {sharpe:.4f}")
    print(f"  交易次数:       {n_trades}")
    print(f"  胜率:           {win_rate:.2f}%")
    print(f"  平均每笔收益:   {avg_profit:.2f}%")
    print(f"  最大单笔盈利:   {max_profit:.2f}%")
    print(f"  最大单笔亏损:   {max_loss:.2f}%")
    print(f"  空仓比例:       {idle:.2f}%")
    print(f"  平均持仓天数:   {avg_len:.1f}")
    print(f"{'='*55}")

    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'idle': idle
    }


def load_csv(path):
    if os.path.basename(path) == 'cnn_result.csv' and os.path.dirname(path) == '':
        pass
    df = pd.read_csv(path, sep=';')
    if 'prediction' not in df.columns:
        df = pd.read_csv(path, sep=';', header=None,
                         names=['prediction', 'test_label', 'test_price'])
    return df


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else ['cnn_result.csv']

    all_data = []
    for f in files:
        label = os.path.basename(os.path.dirname(f)) if os.path.dirname(f) else os.path.splitext(os.path.basename(f))[0]
        if label == 'cnn_result':
            label = os.path.basename(os.path.dirname(f)) if os.path.dirname(f) else 'My Result'
        df = load_csv(f)
        all_data.append((label, df))

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#795548', '#607D8B']

    bah_plotted = False
    stats_all = []

    for idx, (label, df) in enumerate(all_data):
        predictions = df['prediction'].values
        prices = df['test_price'].values

        strategy_history, strategy_final, trade_profits = simulate_strategy(predictions, prices)
        bah_history, bah_final = simulate_bah(prices)

        color = colors[idx % len(colors)]
        axes[0].plot(strategy_history, label=f'CNN策略 - {label}', linewidth=1.5, color=color)

        if not bah_plotted:
            axes[0].plot(bah_history, label=f'买入持有 (最终: ${bah_final:.2f})',
                         linewidth=2, color='gray', linestyle='--', alpha=0.7)
            bah_plotted = True

        drawdown = np.zeros_like(strategy_history)
        peak = np.maximum.accumulate(strategy_history)
        drawdown = (strategy_history - peak) / peak * 100
        axes[1].plot(drawdown, linewidth=1, color=color, alpha=0.8, label=f'回撤 - {label}')

        s = print_stats(f"CNN策略 [{label}]", strategy_history, strategy_final, trade_profits, predictions, prices)
        s['label'] = label
        stats_all.append(s)

    bah_history, bah_final = simulate_bah(all_data[0][1]['test_price'].values)
    bah_return = (bah_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    bah_dd = calc_max_drawdown(bah_history)
    bah_sharpe = calc_sharpe_ratio(bah_history)
    print(f"\n{'='*55}")
    print(f"  买入持有 统计")
    print(f"{'='*55}")
    print(f"  最终资金:   ${bah_final:.2f}")
    print(f"  总收益率:   {bah_return:.2f}%")
    print(f"  最大回撤:   {bah_dd:.2f}%")
    print(f"  Sharpe比率: {bah_sharpe:.4f}")
    print(f"{'='*55}")

    if len(stats_all) > 1:
        print(f"\n{'='*70}")
        print(f"  多结果横向对比")
        print(f"{'='*70}")
        print(f"  {'来源':<15} {'最终资金':>10} {'收益率':>10} {'最大回撤':>10} {'Sharpe':>8} {'交易次数':>8} {'胜率':>8}")
        for s in stats_all:
            print(f"  {s['label']:<15} ${s['final_capital']:>9.2f} {s['total_return']:>9.2f}% {s['max_drawdown']:>9.2f}% {s['sharpe']:>8.4f} {s['n_trades']:>8} {s['win_rate']:>7.1f}%")
        print(f"  {'买入持有':<15} ${bah_final:>9.2f} {bah_return:>9.2f}% {bah_dd:>9.2f}% {bah_sharpe:>8.4f}")
        print(f"{'='*70}")

    axes[0].set_title('CNN预测策略 vs 买入持有 — 资金曲线对比', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('资金 ($)', fontsize=12)
    axes[0].legend(fontsize=10, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=INITIAL_CAPITAL, color='red', linestyle=':', alpha=0.5, linewidth=1)

    axes[1].set_title('策略回撤 (%)', fontsize=12)
    axes[1].set_xlabel('交易日', fontsize=12)
    axes[1].set_ylabel('回撤 (%)', fontsize=12)
    axes[1].legend(fontsize=9, loc='lower left')
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(range(len(drawdown)), drawdown, alpha=0.15, color='red')
    axes[1].axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()

    output_path = 'equity_curve_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: {os.path.abspath(output_path)}")
    plt.show()


if __name__ == '__main__':
    main()