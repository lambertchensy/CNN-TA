"""
CNN-TA 完整复现脚本
===================
复现论文 "Algorithmic Financial Trading with Deep Convolutional Neural Networks:
Time Series to Image Conversion Approach" 的完整流程。

流程：
  Phase0: 反转CSV文件（从旧到新排列）
  Phase1: 计算技术指标 + 生成标签
  Phase2: 标签上移（消除未来数据偏差）
  Phase3: CNN训练与预测 → 输出 cnn_result.csv

使用方法：
  python cnn_ta_reproduction.py --data_dir data/1

依赖：
  pip install pandas numpy ta-lib tensorflow scikit-learn
  或使用 ta 库替代 ta-lib: pip install ta
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# ============================================================
# Phase0: 反转CSV文件
# ============================================================
def phase0_reverse_csv(input_csv, output_csv):
    """
    原始数据是从新到旧排列的（Yahoo Finance格式），需要反转为从旧到新。
    同时计算调整后的Open/High/Low价格（与Phase1.java一致）。
    """
    print(f"[Phase0] Reading {input_csv} ...")
    df = pd.read_csv(input_csv, header=None,
                     names=['Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'])

    # 计算调整后价格（与Java代码一致）
    # AdjOpen = Open * (AdjClose / Close)
    df['AdjOpen'] = df['Open'] * (df['AdjClose'] / df['Close'])
    df['AdjHigh'] = df['High'] * (df['AdjClose'] / df['Close'])
    df['AdjLow'] = df['Low'] * (df['AdjClose'] / df['Close'])

    # 反转：从旧到新
    df = df.iloc[::-1].reset_index(drop=True)

    # 输出格式：Date,AdjOpen,AdjHigh,AdjLow,AdjClose,Close,Volume
    # 注意：Java代码读取的是 parts[0]=Date, parts[1]=AdjOpen, parts[2]=AdjHigh,
    #       parts[3]=AdjLow, parts[5]=AdjClose(parts[5]=AdjClose, parts[4]=Close不用)
    #       parts[6]=Volume
    # 但原始reverseFile格式是：Date,Open,High,Low,Close,AdjClose,Volume
    # Java代码中做了 AdjOpen = Open*(AdjClose/Close) 的转换
    # 所以输出保持原始格式，Phase1中再做调整
    df_out = df[['Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']]

    df_out.to_csv(output_csv, index=False, header=False)
    print(f"[Phase0] Saved reversed CSV to {output_csv} ({len(df_out)} rows)")
    return df_out


# ============================================================
# Phase1: 计算技术指标 + 生成标签
# ============================================================
def compute_indicators_and_labels(df_reversed):
    """
    计算所有技术指标并生成标签。
    与Phase1.java完全对应。

    15个技术指标，每个15个参数（从6到20）：
      RSI, WilliamsR, WMA, EMA, SMA, HMA, TripleEMA,
      CCI, CMO, MACD, PPO, ROC, CMFI, DMI, ParabolicSAR

    标签生成：滑动窗口（WINDOW_SIZE=11）
      - 窗口中间点是最大值 → Sell(2)
      - 窗口中间点是最小值 → Buy(1)
      - 否则 → Hold(0)

    归一化方式（与Java代码一致）：
      RSI:      (RSI/50) - 1
      WilliamsR:(WR/50) + 1
      WMA:      (Close-WMA)*10/Close
      EMA:      (Close-EMA)*10/Close
      SMA:      (Close-SMA)*10/Close
      HMA:      (Close-HMA)*10/Close
      TripleEMA:(Close-TEMA)*10/Close
      CCI:      CCI/300
      CMO:      CMO/100
      MACD:     原值
      PPO:      PPO/10
      ROC:      ROC/20
      CMFI:     原值
      DMI:      (DMI/50) - 1
      PSI:      PSI/Close
    """
    try:
        import talib
        use_talib = True
        print("[Phase1] Using TA-Lib for indicator calculation")
    except ImportError:
        use_talib = False
        print("[Phase1] TA-Lib not found, using pandas-ta fallback")

    WINDOW_SIZE = 11
    close = df_reversed['AdjClose'].values.astype(float)
    high = df_reversed['AdjHigh'].values.astype(float)
    low = df_reversed['AdjLow'].values.astype(float)
    volume = df_reversed['Volume'].values.astype(float)
    n = len(close)

    # --- 生成标签 ---
    labels = np.zeros(n, dtype=float)
    close_list = [0.0]  # Java代码中先加一个0.0
    for i in range(n):
        close_list.append(close[i])

    counter_row = 0
    for i in range(n):
        counter_row += 1
        if counter_row > WINDOW_SIZE:
            window_begin = counter_row - WINDOW_SIZE
            window_end = window_begin + WINDOW_SIZE - 1
            window_middle = (window_begin + window_end) // 2

            min_val = 10000.0
            max_val = 0.0
            for j in range(window_begin, window_end + 1):
                val = close_list[j]
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val

            # Java代码使用closeList.indexOf(min/max)，搜索整个列表返回第一个匹配位置
            # 这会导致当窗口外的更早位置有相同极值时，返回窗口外的索引
            # 为了与Java完全一致，使用Python的list.index()（等同于Java的indexOf）
            min_idx = close_list.index(min_val)
            max_idx = close_list.index(max_val)

            if max_idx == window_middle:
                labels[i] = 2.0
            elif min_idx == window_middle:
                labels[i] = 1.0
            else:
                labels[i] = 0.0

    # --- 计算技术指标 ---
    # 参数范围：6到20（Java代码中索引5到19，对应参数6到20）
    param_range = list(range(6, 21))  # [6,7,...,20]

    def calc_rsi(close_prices, period):
        if use_talib:
            return talib.RSI(close_prices, timeperiod=period)
        else:
            delta = pd.Series(close_prices).diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.values

    def calc_williams_r(high_prices, low_prices, close_prices, period):
        if use_talib:
            return talib.WILLR(high_prices, low_prices, close_prices, timeperiod=period)
        else:
            hh = pd.Series(high_prices).rolling(window=period).max()
            ll = pd.Series(low_prices).rolling(window=period).min()
            wr = (hh - pd.Series(close_prices)) / (hh - ll) * (-100)
            return wr.values

    def calc_wma(close_prices, period):
        if use_talib:
            return talib.WMA(close_prices, timeperiod=period)
        else:
            weights = np.arange(1, period + 1, dtype=float)
            s = pd.Series(close_prices).rolling(window=period)
            wma = s.apply(lambda x: np.dot(x, weights) / weights.sum() if len(x) == period else np.nan, raw=True)
            return wma.values

    def calc_ema(close_prices, period):
        if use_talib:
            return talib.EMA(close_prices, timeperiod=period)
        else:
            return pd.Series(close_prices).ewm(span=period, adjust=False).mean().values

    def calc_sma(close_prices, period):
        if use_talib:
            return talib.SMA(close_prices, timeperiod=period)
        else:
            return pd.Series(close_prices).rolling(window=period).mean().values

    def calc_hma(close_prices, period):
        """Hull Moving Average"""
        if use_talib:
            # TA-Lib没有直接的HMA，手动实现
            pass
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        wma1 = pd.Series(close_prices).rolling(window=half_period).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum() if len(x) == half_period else np.nan, raw=True)
        wma2 = pd.Series(close_prices).rolling(window=period).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum() if len(x) == period else np.nan, raw=True)
        diff = 2 * wma1 - wma2
        hma = diff.rolling(window=sqrt_period).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum() if len(x) == sqrt_period else np.nan, raw=True)
        return hma.values

    def calc_triple_ema(close_prices, period):
        if use_talib:
            return talib.TEMA(close_prices, timeperiod=period)
        else:
            ema1 = pd.Series(close_prices).ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            tema = 3 * ema1 - 3 * ema2 + ema3
            return tema.values

    def calc_cci(high_prices, low_prices, close_prices, period):
        if use_talib:
            return talib.CCI(high_prices, low_prices, close_prices, timeperiod=period)
        else:
            tp = (pd.Series(high_prices) + pd.Series(low_prices) + pd.Series(close_prices)) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            cci = (tp - sma_tp) / (0.015 * mad)
            return cci.values

    def calc_cmo(close_prices, period):
        if use_talib:
            # TA-Lib没有CMO，手动实现
            pass
        delta = pd.Series(close_prices).diff()
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        sum_up = up.rolling(window=period).sum()
        sum_down = down.rolling(window=period).sum()
        cmo = 100 * (sum_up - sum_down) / (sum_up + sum_down).replace(0, np.nan)
        return cmo.values

    def calc_macd(close_prices, fast_period, slow_period):
        if use_talib:
            _, _, macd_hist = talib.MACD(close_prices, fastperiod=fast_period,
                                          slowperiod=slow_period, signalperiod=9)
            return macd_hist
        else:
            ema_fast = pd.Series(close_prices).ewm(span=fast_period, adjust=False).mean()
            ema_slow = pd.Series(close_prices).ewm(span=slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal
            return macd_hist.values

    def calc_ppo(close_prices, fast_period, slow_period):
        if use_talib:
            return talib.PPO(close_prices, fastperiod=fast_period,
                             slowperiod=slow_period, matype=0)
        else:
            ema_fast = pd.Series(close_prices).ewm(span=fast_period, adjust=False).mean()
            ema_slow = pd.Series(close_prices).ewm(span=slow_period, adjust=False).mean()
            ppo = (ema_fast - ema_slow) / ema_slow * 100
            return ppo.values

    def calc_roc(close_prices, period):
        if use_talib:
            return talib.ROC(close_prices, timeperiod=period)
        else:
            roc = pd.Series(close_prices).pct_change(periods=period) * 100
            return roc.values

    def calc_cmfi(high_prices, low_prices, close_prices, volume_prices, period):
        """Chaikin Money Flow Index"""
        if use_talib:
            # TA-Lib使用ADOSC，不完全一样，手动实现CMF
            pass
        h = pd.Series(high_prices)
        l = pd.Series(low_prices)
        c = pd.Series(close_prices)
        v = pd.Series(volume_prices)
        # Money Flow Multiplier
        mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        mfv = mfm * v
        cmf = mfv.rolling(window=period).sum() / v.rolling(window=period).sum()
        return cmf.values

    def calc_dmi(high_prices, low_prices, close_prices, period):
        """Directional Movement Index - returns ADX"""
        if use_talib:
            return talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)
        else:
            h = pd.Series(high_prices)
            l = pd.Series(low_prices)
            c = pd.Series(close_prices)
            plus_dm = h.diff()
            minus_dm = -l.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            # 当 plus_dm < minus_dm 时，plus_dm = 0
            cond1 = plus_dm < minus_dm
            plus_dm[cond1] = 0
            cond2 = minus_dm < plus_dm
            minus_dm[cond2] = 0

            tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * plus_dm.rolling(window=period).mean() / atr.replace(0, np.nan)
            minus_di = 100 * minus_dm.rolling(window=period).mean() / atr.replace(0, np.nan)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            adx = dx.rolling(window=period).mean()
            return adx.values

    def calc_parabolic_sar(high_prices, low_prices, close_prices, period):
        """
        Parabolic SAR。
        ta4j中ParabolicSarIndicator的参数是acceleration factor，
        而非period。这里用TA-Lib实现，参数映射为step=0.02, max=0.2。
        注意：Java代码中psiArray[i]的参数i+1被ta4j内部用作不同含义，
        但ta4j的ParabolicSarIndicator(timeSeries, timeFrame)中timeFrame
        实际控制的是acceleration起始步长。为简化，统一使用标准SAR参数。
        """
        if use_talib:
            return talib.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2)
        else:
            # 简化实现：返回NaN，需要TA-Lib
            return np.full(len(close_prices), np.nan)

    close_arr = close.copy()
    high_arr = high.copy()
    low_arr = low.copy()
    vol_arr = volume.copy()

    # 计算所有指标并归一化
    all_features = []

    # 1. RSI (indices 5-19, periods 6-20)
    print("[Phase1] Computing RSI indicators ...")
    for p in param_range:
        rsi = calc_rsi(close_arr, p)
        normalized = (rsi / 50) - 1  # Java: (RSI/50)-1
        all_features.append(np.round(normalized, 2))

    # 2. WilliamsR (indices 5-19, periods 6-20)
    print("[Phase1] Computing WilliamsR indicators ...")
    for p in param_range:
        wr = calc_williams_r(high_arr, low_arr, close_arr, p)
        normalized = (wr / 50) + 1  # Java: (WR/50)+1
        all_features.append(np.round(normalized, 2))

    # 3. WMA (indices 5-19, periods 6-20)
    print("[Phase1] Computing WMA indicators ...")
    for p in param_range:
        wma = calc_wma(close_arr, p)
        normalized = (close_arr - wma) * 10 / close_arr  # Java: (Close-WMA)*10/Close
        all_features.append(np.round(normalized, 2))

    # 4. EMA (indices 5-19, periods 6-20)
    print("[Phase1] Computing EMA indicators ...")
    for p in param_range:
        ema = calc_ema(close_arr, p)
        normalized = (close_arr - ema) * 10 / close_arr
        all_features.append(np.round(normalized, 2))

    # 5. SMA (indices 5-19, periods 6-20)
    print("[Phase1] Computing SMA indicators ...")
    for p in param_range:
        sma = calc_sma(close_arr, p)
        normalized = (close_arr - sma) * 10 / close_arr
        all_features.append(np.round(normalized, 2))

    # 6. HMA (indices 5-19, periods 6-20)
    print("[Phase1] Computing HMA indicators ...")
    for p in param_range:
        hma = calc_hma(close_arr, p)
        normalized = (close_arr - hma) * 10 / close_arr
        all_features.append(np.round(normalized, 2))

    # 7. TripleEMA (indices 5-19, periods 6-20)
    print("[Phase1] Computing TripleEMA indicators ...")
    for p in param_range:
        tema = calc_triple_ema(close_arr, p)
        normalized = (close_arr - tema) * 10 / close_arr
        all_features.append(np.round(normalized, 2))

    # 8. CCI (indices 5-19, periods 6-20)
    print("[Phase1] Computing CCI indicators ...")
    for p in param_range:
        cci = calc_cci(high_arr, low_arr, close_arr, p)
        normalized = cci / 300  # Java: CCI/300
        all_features.append(np.round(normalized, 2))

    # 9. CMO (indices 5-19, periods 6-20)
    print("[Phase1] Computing CMO indicators ...")
    for p in param_range:
        cmo = calc_cmo(close_arr, p)
        normalized = cmo / 100  # Java: CMO/100
        all_features.append(np.round(normalized, 2))

    # 10. MACD (indices 5-19, fast=6..20, slow=12..40)
    print("[Phase1] Computing MACD indicators ...")
    for p in param_range:
        macd = calc_macd(close_arr, p, p * 2)
        all_features.append(np.round(macd, 2))  # Java: 原值

    # 11. PPO (indices 5-19, fast=6..20, slow=12..40)
    print("[Phase1] Computing PPO indicators ...")
    for p in param_range:
        ppo = calc_ppo(close_arr, p, p * 2)
        normalized = ppo / 10  # Java: PPO/10
        all_features.append(np.round(normalized, 2))

    # 12. ROC (indices 5-19, periods 6-20)
    print("[Phase1] Computing ROC indicators ...")
    for p in param_range:
        roc = calc_roc(close_arr, p)
        normalized = roc / 20  # Java: ROC/20
        all_features.append(np.round(normalized, 2))

    # 13. CMFI (indices 5-19, periods 6-20)
    print("[Phase1] Computing CMFI indicators ...")
    for p in param_range:
        cmfi = calc_cmfi(high_arr, low_arr, close_arr, vol_arr, p)
        all_features.append(np.round(cmfi, 2))  # Java: 原值

    # 14. DMI (indices 5-19, periods 6-20)
    print("[Phase1] Computing DMI indicators ...")
    for p in param_range:
        dmi = calc_dmi(high_arr, low_arr, close_arr, p)
        normalized = (dmi / 50) - 1  # Java: (DMI/50)-1
        all_features.append(np.round(normalized, 2))

    # 15. ParabolicSAR (indices 5-19)
    # Java代码中PSI的参数是i+1，但ta4j的ParabolicSarIndicator参数含义不同
    # 统一使用标准SAR参数(acceleration=0.02, maximum=0.2)
    print("[Phase1] Computing ParabolicSAR indicators ...")
    for p in param_range:
        psi = calc_parabolic_sar(high_arr, low_arr, close_arr, p)
        normalized = psi / close_arr  # Java: PSI/Close
        all_features.append(np.round(normalized, 2))

    # --- 组装输出 ---
    # 格式：label;price;feature1;feature2;...;feature225;
    # 注意：Java代码中第1个配置（data/1）的指标顺序为：
    # resultLabel + resultRSI + resultWR + resultWMA + resultEMA + resultSMA +
    # resultHMA + resultTripleEMA + resultCCI + resultCMO + resultMACD +
    # resultPPO + resultROC + resultCMFI + resultDMI + resultRSI
    # 注意最后一个是resultRSI（重复了RSI），这是Java代码的bug/feature

    print("[Phase1] Assembling output ...")
    rows = []
    for i in range(n):
        row = [labels[i], close[i]]
        for feat in all_features:
            val = feat[i] if i < len(feat) else np.nan
            if np.isnan(val) if isinstance(val, float) else False:
                row.append('NaN')
            else:
                row.append(str(val))
        rows.append(row)

    # 构建DataFrame
    col_names = ['label', 'price'] + [f'f{j}' for j in range(225)]
    result_df = pd.DataFrame(rows, columns=col_names)
    # 末尾加分号（与Java输出一致）
    result_df['trailing'] = ''

    print(f"[Phase1] Done. {len(result_df)} rows, {len(result_df.columns)} columns")
    return result_df


# ============================================================
# Phase2: 标签上移
# ============================================================
def phase2_shift_labels(df, shift_count=6):
    """
    将标签列上移shift_count行。
    Java Phase2.java中硬编码了6次上移（for j=0;j<6;j++），
    虽然注释写的是windowSize/2=5，但实际代码用的是6。
    上移后，最后shift_count行的标签会被移走（设为0）。
    """
    print(f"[Phase2] Shifting labels up by {shift_count} rows ...")
    labels = df['label'].values.copy()
    # 上移：label[i] = label[i+1]，执行shift_count次
    for _ in range(shift_count):
        for k in range(len(labels) - 1):
            labels[k] = labels[k + 1]
        labels[-1] = 0.0  # 最后一行填充0

    df['label'] = labels
    print(f"[Phase2] Done.")
    return df


# ============================================================
# Phase3: CNN训练与预测
# ============================================================
def phase3_cnn_train_predict(train_df, test_df,
                              input_w=15, input_h=15, num_classes=3,
                              batch_size=1024, epochs=200):
    """
    CNN模型训练与预测，与main.py完全对应。

    步骤：
    1. 数据预处理：去掉最后一列(trailing)，去掉NaN，去掉前15行
    2. 类别平衡：过采样少数类（Buy=1, Sell=2）使其与Hold=0数量相当
    3. 训练CNN模型
    4. 预测测试集
    5. 输出cnn_result.csv
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    from sklearn.metrics import confusion_matrix

    # --- 数据预处理 ---
    # 去掉最后一列(trailing)
    if 'trailing' in train_df.columns:
        train_df = train_df.drop(columns=['trailing'])
    if 'trailing' in test_df.columns:
        test_df = test_df.drop(columns=['trailing'])

    # 去掉NaN
    train_df = train_df.dropna(axis=0)
    test_df = test_df.dropna(axis=0)

    # 去掉前15行
    train_df = train_df.iloc[15:, :]
    test_df = test_df.iloc[15:, :]

    # --- 类别平衡（过采样） ---
    l0_train = train_df[train_df['label'] == 0]
    l1_train = train_df[train_df['label'] == 1]
    l2_train = train_df[train_df['label'] == 2]
    l0_size = len(l0_train)
    l1_size = len(l1_train)
    l2_size = len(l2_train)

    l0_l1_ratio = l0_size // l1_size if l1_size > 0 else 0
    l0_l2_ratio = l0_size // l2_size if l2_size > 0 else 0

    print(f"[Phase3] Before oversampling: l0={l0_size}, l1={l1_size}, l2={l2_size}")
    print(f"[Phase3] l0_l1_ratio={l0_l1_ratio}, l0_l2_ratio={l0_l2_ratio}")

    # 过采样
    l1_new_list = []
    l2_new_list = []
    for _, row in train_df.iterrows():
        if row['label'] == 1:
            for _ in range(l0_l1_ratio):
                l1_new_list.append(row)
        if row['label'] == 2:
            for _ in range(l0_l2_ratio):
                l2_new_list.append(row)

    if l1_new_list:
        l1_new = pd.DataFrame(l1_new_list)
        train_df = pd.concat([train_df, l1_new], ignore_index=True)
    if l2_new_list:
        l2_new = pd.DataFrame(l2_new_list)
        train_df = pd.concat([train_df, l2_new], ignore_index=True)

    # Shuffle
    train_df = shuffle(train_df, random_state=42)

    # 验证平衡后
    l0_after = len(train_df[train_df['label'] == 0])
    l1_after = len(train_df[train_df['label'] == 1])
    l2_after = len(train_df[train_df['label'] == 2])
    print(f"[Phase3] After oversampling: l0={l0_after}, l1={l1_after}, l2={l2_after}")

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # --- 准备数据 ---
    feature_cols = [c for c in train_df.columns if c not in ['label', 'price']]

    train_images = train_df[feature_cols].values.astype(float)
    train_labels = train_df['label'].values.astype(int)
    train_prices = train_df['price'].values.astype(float)

    test_images = test_df[feature_cols].values.astype(float)
    test_labels = test_df['label'].values.astype(int)
    test_prices = test_df['price'].values.astype(float)

    # One-hot编码
    train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)
    test_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

    # Reshape为图像格式 (samples, 15, 15, 1)
    train_images = train_images.reshape(train_images.shape[0], input_w, input_h, 1)
    test_images = test_images.reshape(test_images.shape[0], input_w, input_h, 1)

    print(f"[Phase3] Train: {train_images.shape[0]} samples, Test: {test_images.shape[0]} samples")

    # --- 构建CNN模型 ---
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(input_w, input_h, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy', 'mae', 'mse']
    )

    # --- 训练 ---
    print("[Phase3] Training CNN ...")
    model.fit(train_images, train_labels_cat,
              batch_size=batch_size, epochs=epochs, verbose=1)

    # --- 预测 ---
    print("[Phase3] Predicting test set ...")
    predictions = model.predict(test_images, batch_size=batch_size, verbose=1)

    # --- 评估 ---
    test_eval = model.evaluate(test_images, test_labels_cat, batch_size=batch_size, verbose=0)
    print(f"[Phase3] Test loss/accuracy/mae/mse: {test_eval}")

    # 混淆矩阵
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels_cat, axis=1)
    print(f"[Phase3] Test confusion matrix:\n{confusion_matrix(true_classes, pred_classes)}")

    # --- 输出结果 ---
    result_df = pd.DataFrame({
        'prediction': pred_classes,
        'test_label': true_classes,
        'test_price': test_prices
    })

    return result_df


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='CNN-TA 完整复现脚本')
    parser.add_argument('--data_dir', type=str, default='data/1',
                        help='数据目录（包含reverseFileTrainingFirst.csv等）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认为data_dir）')
    parser.add_argument('--epochs', type=int, default=200,
                        help='CNN训练轮数')
    parser.add_argument('--skip_phase1', action='store_true',
                        help='跳过Phase1（如果outputOfPhase2已存在）')
    parser.add_argument('--skip_training', action='store_true',
                        help='跳过训练，仅使用已有数据预测')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir
    os.makedirs(output_dir, exist_ok=True)

    # ===== Phase0: 反转CSV =====
    # Phase0流程：原始CSV(从新到旧) → reverseFileFirst(从旧到新) → reverseFile(从旧到新，Phase1输入)
    # reverseFileTraining.csv 和 reverseFileTest.csv 是Phase1的实际输入
    train_csv = os.path.join(data_dir, 'reverseFileTraining.csv')
    test_csv = os.path.join(data_dir, 'reverseFileTest.csv')

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f"[Phase0] Error: {train_csv} or {test_csv} not found!")
        print("  These files are the Phase1 input (chronologically ordered price data).")
        sys.exit(1)
    print(f"[Phase0] Using existing reverse files: {train_csv}, {test_csv}")

    # ===== Phase1: 计算指标+标签 =====
    phase2_train_csv = os.path.join(output_dir, 'outputOfPhase2Training.csv')
    phase2_test_csv = os.path.join(output_dir, 'outputOfPhase2Test.csv')

    if not args.skip_phase1:
        # 读取反转后的CSV（从旧到新排列）
        # CSV格式：Date,Open,High,Low,Close,AdjClose,Volume
        # Java代码中：AdjOpen=Open*(AdjClose/Close), AdjHigh=High*(AdjClose/Close),
        #            AdjLow=Low*(AdjClose/Close), Close=AdjClose
        print(f"\n[Phase1] Processing training data ...")
        df_train = pd.read_csv(train_csv, header=None,
                               names=['Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'])
        df_train['AdjOpen'] = df_train['Open'] * (df_train['AdjClose'] / df_train['Close'])
        df_train['AdjHigh'] = df_train['High'] * (df_train['AdjClose'] / df_train['Close'])
        df_train['AdjLow'] = df_train['Low'] * (df_train['AdjClose'] / df_train['Close'])

        train_result = compute_indicators_and_labels(df_train)

        # Phase2: 标签上移
        train_result = phase2_shift_labels(train_result, shift_count=6)

        # 保存
        train_result.to_csv(phase2_train_csv, sep=';', index=False, header=False)
        print(f"[Phase1+2] Training data saved to {phase2_train_csv}")

        # 测试数据
        print(f"\n[Phase1] Processing test data ...")
        df_test = pd.read_csv(test_csv, header=None,
                              names=['Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'])
        df_test['AdjOpen'] = df_test['Open'] * (df_test['AdjClose'] / df_test['Close'])
        df_test['AdjHigh'] = df_test['High'] * (df_test['AdjClose'] / df_test['Close'])
        df_test['AdjLow'] = df_test['Low'] * (df_test['AdjClose'] / df_test['Close'])

        test_result = compute_indicators_and_labels(df_test)
        test_result = phase2_shift_labels(test_result, shift_count=6)

        test_result.to_csv(phase2_test_csv, sep=';', index=False, header=False)
        print(f"[Phase1+2] Test data saved to {phase2_test_csv}")
    else:
        print("[Phase1] Skipped by user, using existing outputOfPhase2 files")

    # ===== Phase3: CNN训练与预测 =====
    print(f"\n[Phase3] Loading processed data ...")

    # 读取Phase2输出
    train_df = pd.read_csv(phase2_train_csv, header=None, delimiter=';')
    test_df = pd.read_csv(phase2_test_csv, header=None, delimiter=';')

    # 去掉最后一列（空列，由末尾分号导致）
    train_df = train_df.iloc[:, :-1]
    test_df = test_df.iloc[:, :-1]

    # 设置列名：第0列=label, 第1列=price, 其余为特征
    train_df.columns = ['label', 'price'] + [f'f{i}' for i in range(len(train_df.columns) - 2)]
    test_df.columns = ['label', 'price'] + [f'f{i}' for i in range(len(test_df.columns) - 2)]

    # 转换为数值
    for col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    for col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # CNN训练与预测
    result_df = phase3_cnn_train_predict(train_df, test_df, epochs=args.epochs)

    # 保存结果
    result_csv = os.path.join(output_dir, 'cnn_result.csv')
    result_df.to_csv(result_csv, sep=';', index=False)
    print(f"\n[Done] Results saved to {result_csv}")
    print(f"  Predictions: {len(result_df)} rows")
    print(f"  Prediction distribution: {dict(result_df['prediction'].value_counts().sort_index())}")
    print(f"  True label distribution: {dict(result_df['test_label'].value_counts().sort_index())}")


if __name__ == '__main__':
    main()
