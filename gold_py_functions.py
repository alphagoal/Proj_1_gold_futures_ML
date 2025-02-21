from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score

# 1-month risk-free return accumulated from overnight LIBOR rate
def calculate_monthly_log_rf_ret(date_now,col):
    start_date = date_now - pd.DateOffset(months=1)
    mask = (col.index > start_date) & (col.index <= date_now)
    return col.loc[mask].sum()

# This is to do the preprocessing of all the x variables
def build_preprocessor(x1_standarize, x1x2_standardize,X1,X2,X):
    if x1_standarize == True:
        sign_features = X1.columns
        continuous_features = X2.columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), continuous_features),
                ('sign', 'passthrough', sign_features)
            ]
        )
        return preprocessor

    elif x1x2_standardize == True:
        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), X.columns)])

        return preprocessor

    else:
        return None

# This is to plot the net value changes for all four trading strategies (Random forest, LASSO) (Long-short, Long-only)
def trading_plot(gc_data,GC_type,fwd_ret_period,monthly_risk_free_rate,excess_ret_benchmark,learning_result, transaction_cost=[0, 0.002, 0.004, 0.01], benchmark_bp=0.000):
    long_short_result = pd.DataFrame(learning_result)
    # if "Date" is not index, set it to index
    if "Date" in long_short_result.columns:
        long_short_result.set_index("Date", inplace=True)

    # Get the matching return
    matching_ret = gc_data.loc[long_short_result.index, f"{GC_type}_{fwd_ret_period}M_Fwd_Ret"]

    def calculate_strategy_return(row, ret_pred_column, tc, strategy="long_short"):
        """
           Calculate the strategy return based on predicted returns.

           Parameters:
           row (pd.Series): A row of a DataFrame containing the predicted return.
           ret_pred_column (str): The column name containing predicted returns.
           tc (float): Transaction cost.
           strategy (str): The type of strategy to use, default is "long_short".

           Returns:
           pd.Series: A series containing 'strategy_return' and 'strategy_action'.
           """
        # Extract the predicted return for the given row
        ret_pred = row[ret_pred_column]
        # Get the risk-free rate for the corresponding date
        rf_rate = monthly_risk_free_rate.loc[pd.to_datetime(row.name), f"US000{fwd_ret_period}M Index"]
        # Initialize strategy action and return
        strategy_action = "none"
        strategy_return = 0  # just in case
        # Long-short strategy: Take long positions on high expected excess returns and short positions on low excess expected returns
        # and buy the risk-free rate when the expected excess return is near 0
        if strategy == "long_short":
            if ret_pred > excess_ret_benchmark:  # If the predicted excess return is higher than the threshold, go long
                strategy_action = "long"
                strategy_return = matching_ret[row.name] / fwd_ret_period * 1 - tc
            # If the predicted excess return is around 0
            # excess return is ret_real-rf, we would only short if the excess return is not only less than the -rf, but the real return also need to less than the -rf
            # hence abs(excess_return）< 2rf
            elif (0 < ret_pred < excess_ret_benchmark) or (abs(ret_pred) < 2 * rf_rate and ret_pred < 0):
                strategy_action = "risk_free"
                strategy_return = rf_rate - tc
            # we short when the real return less than 0 and absolute value greater than rf
            elif ret_pred < 0 and abs(ret_pred) > 2 * rf_rate:
                strategy_action = "short"
                strategy_return = matching_ret[row.name] / fwd_ret_period * (-1) - tc
        else:
            # we don't do short here, only long and buy risk-free rate
            if ret_pred > excess_ret_benchmark:
                strategy_action = "long"
                strategy_return = matching_ret[row.name] / fwd_ret_period * 1 - tc
            elif (0 < ret_pred < excess_ret_benchmark) or (abs(ret_pred) < 2 * rf_rate and ret_pred < 0):
                strategy_action = "risk_free"
                strategy_return = rf_rate - tc
            elif ret_pred < 0 and abs(ret_pred) > 2 * rf_rate:
                strategy_action = "risk_free"
                strategy_return = rf_rate - tc

        return pd.Series({
            "strategy_return": strategy_return,
            "strategy_action": strategy_action,
        })

    # calculate strategy return
    for tc in transaction_cost:
        long_short_result[[f"strategy_return_{int(tc * 10000)}bp",
                           "strategy_action"]] = long_short_result.apply(
            lambda row: calculate_strategy_return(row, "ret_pred", tc, "long_short"), axis=1)

        # calculate cumulative return for our trading strategy
        long_short_result[f"cumulative_strategy_{int(tc * 10000)}bp"] = np.exp(
            long_short_result[f"strategy_return_{int(tc * 10000)}bp"].cumsum())

    # calculate benchmark return
    long_short_result[["benchmark_return", "benchmark_action"]] = long_short_result.apply(calculate_strategy_return,
                                                                                          axis=1, args=(
        "historical_mean", benchmark_bp, "long_short"))

    # calculate buy & hold return
    long_short_result["buy_and_hold_return"] = gc_data[f"{GC_type}_Monthly_raw_Return"].loc[long_short_result.index]

    # calculate cumulative return for all strategies
    long_short_result["cumulative_benchmark"] = np.exp(long_short_result["benchmark_return"].cumsum())
    long_short_result["cumulative_buy_and_hold"] = np.exp(long_short_result["buy_and_hold_return"].cumsum())
    long_short_result['cumulative_risk_free'] = np.exp(monthly_risk_free_rate[f"US000{fwd_ret_period}M Index"].cumsum())
    long_short_result = long_short_result.dropna()

    long_result = pd.DataFrame(learning_result)
    if "Date" in long_result.columns:
        long_result.set_index("Date", inplace=True)
    for tc in transaction_cost:
        long_result[[f"strategy_return_{int(tc * 10000)}bp", "strategy_action"]] = long_result.apply(
            calculate_strategy_return, axis=1, args=("ret_pred", tc, "long"))
        long_result[f"cumulative_strategy_{int(tc * 10000)}bp"] = np.exp(
            long_result[f"strategy_return_{int(tc * 10000)}bp"].cumsum())

    # calculate benchmark return
    long_result[["benchmark_return", "benchmark_action"]] = long_result.apply(calculate_strategy_return, axis=1, args=(
    "historical_mean", benchmark_bp, "long"))

    # calculate buy & hold return
    long_result["buy_and_hold_return"] = gc_data[f"{GC_type}_Monthly_raw_Return"].loc[long_result.index]

    # calculate cumulative return
    long_result["cumulative_benchmark"] = np.exp(long_result["benchmark_return"].cumsum())
    long_result["cumulative_buy_and_hold"] = np.exp(long_result["buy_and_hold_return"].cumsum())
    long_result['cumulative_risk_free'] = np.exp(monthly_risk_free_rate[f"US000{fwd_ret_period}M Index"].cumsum())
    long_result = long_result.dropna()

    # create subplots
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # plot long-short strategy
    for tc in transaction_cost:
        ax[0].plot(long_short_result.index, long_short_result[f"cumulative_strategy_{int(tc * 10000)}bp"],
                   label=f"Strategy Return_{int(tc * 10000)}bp", linewidth=2)
    ax[0].plot(long_short_result.index, long_short_result["cumulative_benchmark"],
               label=f"Benchmark Return_{int(benchmark_bp * 10000)}bp", linestyle="--", linewidth=2)
    ax[0].plot(long_short_result.index, long_short_result["cumulative_buy_and_hold"], label="Buy & Hold Return",
               linestyle=":", linewidth=4)
    ax[0].plot(long_short_result.index, long_short_result['cumulative_risk_free'], label="Risk Free Rate Return",
               linestyle="-.", linewidth=4)
    ax[0].set_ylabel("Cumulative Return")
    ax[0].set_title("Trading Strategy vs Benchmark vs Buy & Hold (No Short Constraint)")
    ax[0].legend()
    ax[0].grid()

    # plot long-only strategy
    for tc in transaction_cost:
        ax[1].plot(long_result.index, long_result[f"cumulative_strategy_{int(tc * 10000)}bp"],
                   label=f"Strategy Return_{int(tc * 10000)}bp", linewidth=2)
    ax[1].plot(long_result.index, long_result["cumulative_benchmark"],
               label=f"Benchmark Return_{int(benchmark_bp * 10000)}bp", linestyle="--", linewidth=2)
    ax[1].plot(long_result.index, long_result["cumulative_buy_and_hold"], label="Buy & Hold Return", linestyle=":",
               linewidth=4)
    ax[1].plot(long_result.index, long_result['cumulative_risk_free'], label="Risk Free Rate Return", linestyle="-.",
               linewidth=4)
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Cumulative Return")
    ax[1].set_title("Trading Strategy vs Benchmark vs Buy & Hold (Long Only)")
    ax[1].legend()
    ax[1].grid()
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()
    return long_short_result, long_result

# This is to add varies performance matrices to the dataframe to display
def add_performance_matrices(df):
    #Calculate win rate for strategy and benchmark.
    # Ensure required columns exist
    required_cols = {"ret_pred", "ret_real", "historical_mean"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Calculate strategy win rate
    df["strategy_win_rate"] = (df["ret_pred"] * df["ret_real"]) > 0
    # Calculate benchmark win rate
    df["benchmark_win_rate"] = (df["historical_mean"] * df["ret_real"]) > 0
    df["strategy_correct_percent"] = 1-np.where(
    df["strategy_win_rate"],  # Condition: Only calculate when strategy_win_rate is True
    (1 - (abs(abs(df["ret_real"]) - abs(df["ret_pred"])) / (abs(df["ret_real"]) + abs(df["ret_pred"])))),  # Compute strategy correctness percentage
    2)    # Set to -1 when strategy_win_rate is False
    df["benchmark_correct_percent"] = 1-np.where(
    df["benchmark_win_rate"],  # Condition: Only calculate when strategy_win_rate is True
    (1 - (abs(abs(df["ret_real"]) - abs(df["historical_mean"])) / (abs(df["ret_real"]) + abs(df["historical_mean"])))),  # Compute strategy correctness percentage
    2)    # Set to -1 when strategy_win_rate is False
     # Calculate SSE (Sum of Squared Errors) for strategy and benchmark
    df["strategy_SSE"] = (df["ret_real"] - df["ret_pred"]) ** 2
    df["benchmark_SSE"] = (df["ret_real"] - df["historical_mean"]) ** 2
    df["SSE_difference"] = df["benchmark_SSE"] - df["strategy_SSE"]
    df["cumulative_strategy_SSE"] = df["strategy_SSE"].cumsum()
    df["cumulative_benchmark_SSE"] = df["benchmark_SSE"].cumsum()
    df["cumulative_SSE_difference"] = df["SSE_difference"].cumsum()

    return df

# Function to plot win rate performance
def plot_win_rate(df, column_name, title):
    plt.figure(figsize=(12, 6))

    # Extract x (index) and y (values)
    x = df.index
    y = df[column_name]

    # Clip values to ensure within [-1, 1] range
    y = np.clip(y, -1, 1)

    # Define bar width (small width to resemble thin cylinders)
    bar_width = 0.5

    # Plot bars: Green for positive values, Red for negative values
    plt.bar(x[y >= 0], y[y >= 0], color='g', width=bar_width, align='center', edgecolor='green')
    plt.bar(x[y < 0], y[y < 0], color='r', width=bar_width, align='center', edgecolor='red')

    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a horizontal zero line
    plt.xlabel("Index")
    plt.ylabel(column_name)
    plt.title(title)

    plt.show()


def performance_calculator(model_result,monthly_risk_free_rate,fwd_ret_period,transaction_cost,benchmark_bp=0):
    performance_metrics = []

    def calculate_sharpe_ratio(returns, risk_free_rate=monthly_risk_free_rate[f"US000{fwd_ret_period}M Index"]):
        """ Calculate the sharpe ratio """
        excess_returns = returns - risk_free_rate
        annualized_excess_return = (np.exp(excess_returns.sum())) ** (1 / (len(risk_free_rate) / 12)) - 1
        annualized_std = returns.std() * (np.sqrt(12))
        return annualized_excess_return / annualized_std if excess_returns.std() != 0 else np.nan

    def calculate_max_drawdown(cumulative_returns):
        """ Calculate the max drawdown """
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    def calculate_win_rate(model_win_col):
        model_win_rate = model_win_col.sum() / len(model_win_col)
        return model_win_rate

    def calculate_r_square(model, strategy_flag):
        if strategy_flag == True:
            r2 = r2_score(model["ret_real"], model["ret_pred"])
        else:
            r2 = r2_score(model["ret_real"], model["historical_mean"])
        return r2

    # Calculate performance matrices for the strategy
    for tc in transaction_cost:
        col_name_return = f"strategy_return_{int(tc * 10000)}bp"
        col_name_cum = f"cumulative_strategy_{int(tc * 10000)}bp"

        final_net_value = model_result[col_name_cum].iloc[-1]
        annualized_return = (np.exp(model_result[col_name_return].sum())) ** (
                    1 / (len(model_result[col_name_return]) / 12)) - 1
        sharpe_ratio = calculate_sharpe_ratio(model_result[col_name_return],
                                              monthly_risk_free_rate[f"US000{fwd_ret_period}M Index"])
        max_drawdown = calculate_max_drawdown(model_result[col_name_cum])
        strategy_win_rate = calculate_win_rate(model_result['strategy_win_rate'])
        strategy_r2 = calculate_r_square(model_result, True)
        performance_metrics.append(
            ["Strategy", f"{int(tc * 10000)}bp", final_net_value, annualized_return, sharpe_ratio, max_drawdown,
             strategy_win_rate, strategy_r2])

    # Calculate performance matrices for the benchmark
    final_net_value_benchmark = model_result["cumulative_benchmark"].iloc[-1]
    annualized_return_benchmark = np.exp(model_result["benchmark_return"].sum()) ** (
                1 / len(model_result["benchmark_return"]) / 12) - 1
    sharpe_ratio_benchmark = calculate_sharpe_ratio(model_result["benchmark_return"],
                                                    monthly_risk_free_rate[f"US000{fwd_ret_period}M Index"])
    max_drawdown_benchmark = calculate_max_drawdown(model_result["cumulative_benchmark"])
    benchmark_win_rate = calculate_win_rate(model_result['benchmark_win_rate'])
    benchmark_r2 = calculate_r_square(model_result, False)
    performance_metrics.append(
        ["Benchmark", f"{int(benchmark_bp * 10000)}bp", final_net_value_benchmark, annualized_return_benchmark,
         sharpe_ratio_benchmark, max_drawdown_benchmark, benchmark_win_rate, benchmark_r2])

    # Calculate performance matrices for buy & hold
    final_net_value_bh = model_result["cumulative_buy_and_hold"].iloc[-1]
    annualized_return_bh = np.exp(model_result["buy_and_hold_return"].sum()) ** (
                1 / len(model_result["benchmark_return"]) / 12) - 1
    sharpe_ratio_bh = calculate_sharpe_ratio(model_result["buy_and_hold_return"],
                                             monthly_risk_free_rate[f"US000{fwd_ret_period}M Index"])
    max_drawdown_bh = calculate_max_drawdown(model_result["cumulative_buy_and_hold"])
    performance_metrics.append(
        ["Buy & Hold", "-", final_net_value_bh, annualized_return_bh, sharpe_ratio_bh, max_drawdown_bh, np.nan])

    # create the performance DataFrame
    performance_df = pd.DataFrame(performance_metrics,
                                  columns=["Strategy", "Transaction Cost", "Final Net Value", "Annualized Return",
                                           "Sharpe Ratio", "Max Drawdown", "Win Rate", "R2"])
    return performance_df

# calculate the cumulative and max drawdown of each time step
def calculate_drawdowns(df,transaction_cost):
    # support function to calculate drawdown
    def compute_drawdown(series):
        """calculate accumulative drawn（Drawdown）and max drawdown（Max Drawdown）"""
        max_cum = series.cummax()  # Calculate the max net value
        cum_drawdown = (series - max_cum) / max_cum  # calculate the current
        max_drawdown = cum_drawdown.cummin()  # calculate the max drawdown
        return cum_drawdown, max_drawdown

    # calculate the drawdown for strategies with different transaction cost
    for tc in transaction_cost:
        df[f'drawdown_strategy_{int(tc * 10000)}bp'], df[
            f'max_drawdown_strategy_{int(tc * 10000)}bp'] = compute_drawdown(
            df[f'cumulative_strategy_{int(tc * 10000)}bp'])
    # calculate the drawdown for the benchmark
    df['cumulative_drawdown_benchmark'], df['max_drawdown_benchmark'] = compute_drawdown(df['cumulative_benchmark'])

    # alculate the drawdown for the buy & hold
    df['cumulative_drawdown_buy_and_hold'], df['max_drawdown_buy_and_hold'] = compute_drawdown(df['cumulative_buy_and_hold'])

    return df