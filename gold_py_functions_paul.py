import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Similar to trading_plot from gpf
#Here you can input a list of learning_results, and their associated names in a list the same order.
#You can still specify txs fees
def trading_plot_2(gc_data,GC_type,fwd_ret_period,monthly_risk_free_rate,excess_ret_thershold,_learning_result, transaction_cost=[0, 0.002, 0.004, 0.01], benchmark_bp=0.000, names = None):
    # create subplots
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    long_short_dict = {}
    long_only_dict = {}
    
    
    for i in range(len(_learning_result)):
        learning_result = _learning_result[i]
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
            """ Remarks: Strategy Return = Realized Return / Length of holding period  
            <-- We invest a portion of (1/length of holding period every month), achieving monthly averaging effect
            """
            
            if ret_pred > excess_ret_thershold:  # If the predicted excess return is higher than the threshold, go long
                strategy_action = "long"
                strategy_return = matching_ret[row.name] / fwd_ret_period * 1 - tc
                
            # If the predicted excess return is around 0
            # excess return is ret_real-rf, we would only short if the excess return is not only less than the -rf, but the real return also need to less than the -rf
            # hence abs(excess_return）< 2rf
            
            elif (0 < ret_pred < excess_ret_thershold) or (abs(ret_pred) < 2 * rf_rate and ret_pred < 0):
                strategy_action = "risk_free"
                strategy_return = rf_rate - tc
                
            # we short when the real return less than 0 and absolute value greater than rf
            elif ret_pred < 0 and abs(ret_pred) > 2 * rf_rate:
                if strategy == "long_short":
                    strategy_action = "short"
                    strategy_return = matching_ret[row.name] / fwd_ret_period * (-1) - tc
                else: #long-only strategy
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

        long_short_dict[names[i]] = long_short_result #newly added
        
        
        # calculate benchmark return
        long_short_result[["benchmark_return", "benchmark_action"]] = long_short_result.apply(calculate_strategy_return,
                                                                                            axis=1, args=(
            "historical_mean", benchmark_bp, "long_short"))

        # calculate buy & hold return
        long_short_result["buy_and_hold_return"] = gc_data[f"{GC_type}_Monthly_raw_Return"].loc[long_short_result.index]


        # calculate cumulative return for LONG-SHORT STRATEGY
        long_short_result["cumulative_benchmark"] = np.exp(long_short_result["benchmark_return"].cumsum())
        long_short_result["cumulative_buy_and_hold"] = np.exp(long_short_result["buy_and_hold_return"].cumsum())
        long_short_result['cumulative_risk_free'] = np.exp(monthly_risk_free_rate[f"US000{fwd_ret_period}M Index"].cumsum())
        long_short_result = long_short_result.dropna()


        # calculate cumulative return for LONG-ONLY STRATEGY
        long_result = pd.DataFrame(learning_result)
        if "Date" in long_result.columns:
            long_result.set_index("Date", inplace=True)
        for tc in transaction_cost:
            long_result[[f"strategy_return_{int(tc * 10000)}bp", "strategy_action"]] = long_result.apply(
                calculate_strategy_return, axis=1, args=("ret_pred", tc, "long"))
            long_result[f"cumulative_strategy_{int(tc * 10000)}bp"] = np.exp(
                long_result[f"strategy_return_{int(tc * 10000)}bp"].cumsum())
        
        long_only_dict[names[i]] = long_result #newly added

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


        
    
        #plot the strategy first
        # plot long-short strategy
        for tc in transaction_cost:
            if names is not None:
                name = names[i]+f'_{int(tc * 10000)}bp'
            else:
                name = f"Strategy Return_{int(tc * 10000)}bp"
            ax[0].plot(long_short_result.index, long_short_result[f"cumulative_strategy_{int(tc * 10000)}bp"],
                    label=name, linewidth=2)
        
        # plot long-only strategy
            ax[1].plot(long_result.index, long_result[f"cumulative_strategy_{int(tc * 10000)}bp"],
                    label=name, linewidth=2)
        
        
        
        #print(long_only_dict)
        
    
    # -----------------------------------------
    
    #plot the benchmarks for long-short
    # ax[0].plot(long_short_result.index, long_short_result["cumulative_benchmark"],
    #            label=f"Benchmark Return_{int(benchmark_bp * 10000)}bp", linestyle="--", linewidth=2) #no longer use benchmark return
    ax[0].plot(long_short_result.index, long_short_result["cumulative_buy_and_hold"], label="Buy & Hold Return",
               linestyle=":", linewidth=4)
    ax[0].plot(long_short_result.index, long_short_result['cumulative_risk_free'], label="Risk Free Rate Return",
               linestyle="-.", linewidth=4)
    ax[0].set_ylabel("Cumulative Return")
    ax[0].set_title("Trading Strategy vs Benchmark vs Buy & Hold (No Short Constraint)")
    ax[0].legend()
    ax[0].grid()

    #plot the benchmarks for long-only
    # ax[1].plot(long_result.index, long_result["cumulative_benchmark"],
    #            label=f"Benchmark Return_{int(benchmark_bp * 10000)}bp", linestyle="--", linewidth=2)   #no longer use benchmark return
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
    return long_short_dict, long_only_dict
    #return long_short_result, long_result
    