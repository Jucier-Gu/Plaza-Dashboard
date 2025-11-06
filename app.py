import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go  # Import Plotly


# --- Core Analysis Functions (Backend) --- 1
@st.cache_data
def calculate_metrics_from_prices(
    data_df, benchmark_ticker, risk_free_rate=0.02, cvar_alpha=0.05
):
    """
    Calculates all key risk metrics (for the top table)
    *** Assumes data_df contains daily prices ***
    *** Returns raw numbers (floats) ***
    """

    daily_returns = data_df.pct_change().dropna()

    if benchmark_ticker not in daily_returns.columns:
        st.error(f"Error: Benchmark '{benchmark_ticker}' not found in CSV columns.")
        return pd.DataFrame()

    benchmark_returns = daily_returns[benchmark_ticker]

    T = 252
    metrics_list = []

    for fund in daily_returns.columns:
        fund_returns = daily_returns[fund]
        annual_return = (1 + fund_returns.mean()) ** T - 1
        annual_volatility = fund_returns.std() * np.sqrt(T)

        covariance = fund_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance

        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

        downside_returns = fund_returns[fund_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(T)
        sortino_ratio = (
            (annual_return - risk_free_rate) / downside_std
            if downside_std != 0
            else np.nan
        )

        fund_prices = data_df[fund]
        peak = fund_prices.expanding(min_periods=1).max()
        drawdown = (fund_prices - peak) / peak
        max_drawdown = drawdown.min()

        var_95 = fund_returns.quantile(cvar_alpha)
        cvar_95 = fund_returns[fund_returns <= var_95].mean()

        delta = fund_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        metrics_list.append(
            {
                "Fund": fund,
                "Annual. Return": annual_return,
                "Annual. Volatility": annual_volatility,
                "Beta": beta,
                "Sharpe Ratio": sharpe_ratio,
                "Sortino Ratio": sortino_ratio,
                "Max Drawdown": max_drawdown,
                "CVaR (5%)": cvar_95,
                "RSI (14 Day)": current_rsi,
            }
        )

    return pd.DataFrame(metrics_list).set_index("Fund")


def generate_interpretation(metrics_df, benchmark_ticker):
    """Auto-generate interpretations (handles raw numbers)"""
    benchmark_metrics = metrics_df.loc[benchmark_ticker]
    st.markdown("---")
    st.subheader("🤖 Automated Analysis & Suggestions")

    for fund in metrics_df.index:
        if fund == benchmark_ticker:
            continue

        fund_metrics = metrics_df.loc[fund]
        st.markdown(f"**Analysis for {fund}:**")

        try:
            if fund_metrics["Sharpe Ratio"] > benchmark_metrics["Sharpe Ratio"]:
                st.success(
                    f"📈 **Risk-Adjusted Return (Sharpe):** {fund} ({fund_metrics['Sharpe Ratio']:.2f}) is outperforming the benchmark ({benchmark_metrics['Sharpe Ratio']:.2f})."
                )
            else:
                st.warning(
                    f"📉 **Risk-Adjusted Return (Sharpe):** {fund} ({fund_metrics['Sharpe Ratio']:.2f}) is underperforming the benchmark ({benchmark_metrics['Sharpe Ratio']:.2f})."
                )

            if fund_metrics["Max Drawdown"] > benchmark_metrics["Max Drawdown"]:
                st.success(
                    f"🛡️ **Risk Control (Max Drawdown):** {fund} ({fund_metrics['Max Drawdown']:,.2%}) has a smaller max drawdown than the benchmark ({benchmark_metrics['Max Drawdown']:,.2%}), showing better resilience."
                )
            else:
                st.warning(
                    f"🚩 **Risk Control (Max Drawdown):** {fund} ({fund_metrics['Max Drawdown']:,.2%}) has a larger max drawdown than the benchmark ({benchmark_metrics['Max Drawdown']:,.2%})."
                )

            rsi = fund_metrics["RSI (14 Day)"]
            if rsi > 70:
                st.warning(
                    f"🌡️ **Short-Term Signal (RSI):** {fund}'s current RSI is {rsi:.2f}, which is in the 'overbought' territory. A pullback may be possible."
                )
            elif rsi < 30:
                st.success(
                    f"🌡️ **Short-Term Signal (RSI):** {fund}'s current RSI is {rsi:.2f}, which is in the 'oversold' territory. A rebound may be possible."
                )
            else:
                st.info(
                    f"🌡️ **Short-Term Signal (RSI):** {fund}'s current RSI is {rsi:.2f}, which is in a neutral zone."
                )

        except Exception as e:
            if pd.isna(fund_metrics["Sharpe Ratio"]) or pd.isna(
                fund_metrics["RSI (14 Day)"]
            ):
                st.warning(
                    f"Could not generate some insights for {fund} (insufficient data, e.g., for RSI)."
                )
            else:
                st.error(f"Error generating insights for {fund}: {e}.")


def load_data_from_csv(uploaded_file, date_column, start_date, end_date):
    """Load, parse, and filter data from the uploaded CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        if date_column not in df.columns:
            st.error(f"Error: Date column '{date_column}' not found in the CSV.")
            return None
        df["Date_Parsed"] = pd.to_datetime(df[date_column])
        df = df.set_index("Date_Parsed")
        df = df.loc[start_date:end_date]
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            st.error("No numerical data found in the specified date range.")
            return None
        return df_numeric.dropna(axis=1, how="all")
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None


# --- Streamlit Application UI (Frontend) ---

st.set_page_config(layout="wide", page_title="Plaza Fund Risk Dashboard")
st.title("📈 Plaza Automated Fund Risk Analysis")
st.info("✅ CSV must contain **Daily Prices**. Charts are rendered with Plotly.")

# --- Fix 1: Initialize session state ---
# (This flag will be "remembered" across Streamlit reruns)
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

# --- 1. Inputs (Sidebar) ---
st.sidebar.header("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("1. Upload Your Fund Data (CSV)", type=["csv"])
st.sidebar.info("CSV must contain a 'Date' column and daily **Price** columns.")
date_column = st.sidebar.text_input(
    "2. Enter the 'Date' column name from your CSV", "Date"
)
benchmark_column = st.sidebar.text_input(
    "3. Enter the 'Benchmark' column name from your CSV", "SP500"
)

start_date = st.sidebar.date_input(
    "4. Select Analysis Start Date", pd.to_datetime("2020-10-13")
)
end_date = st.sidebar.date_input(
    "5. Select Analysis End Date", pd.to_datetime("2025-10-16")
)

risk_free_rate = (
    st.sidebar.slider("6. Annual Risk-Free Rate (%)", 0.0, 5.0, 2.0, 0.1) / 100
)
run_button = st.sidebar.button("🚀 Run Analysis")

# --- Fix 2: Update main logic ---

# When the button is clicked, set the flag
if run_button:
    st.session_state.analysis_run = True

# Check the "flag", not the "run_button"
if st.session_state.analysis_run:

    # Check if a file has been uploaded
    if uploaded_file is not None:
        with st.spinner("Loading data and calculating metrics..."):

            # Stage 2A: Load and prepare data
            raw_data = load_data_from_csv(
                uploaded_file, date_column, start_date, end_date
            )

            if raw_data is not None and not raw_data.empty:
                # Stage 2B: Calculate
                metrics_table = calculate_metrics_from_prices(
                    raw_data, benchmark_column, risk_free_rate
                )

                if not metrics_table.empty:
                    # Stage 3A: Display the core metrics table
                    st.subheader("📊 Core Risk Metrics")
                    st.dataframe(
                        metrics_table.style.format(
                            {
                                "Annual. Return": "{:.2%}",
                                "Annual. Volatility": "{:.2%}",
                                "Beta": "{:.2f}",
                                "Sharpe Ratio": "{:.2f}",
                                "Sortino Ratio": "{:.2f}",
                                "Max Drawdown": "{:.2%}",
                                "CVaR (5%)": "{:.2%}",
                                "RSI (14 Day)": "{:.2f}",
                            }
                        )
                    )

                    # --- Stage 3B: Prepare all time-series data for charts ---
                    T = 252
                    daily_returns = raw_data.pct_change().dropna()
                    normalized_returns = (1 + daily_returns).cumprod()
                    normalized_returns = normalized_returns / normalized_returns.iloc[0]
                    rolling_vol_20d = daily_returns.rolling(window=20).std() * np.sqrt(
                        T
                    )
                    prices = raw_data
                    peak = prices.expanding(min_periods=1).max()
                    drawdown_series = (prices - peak) / peak
                    rolling_window_20d = 20
                    rolling_annual_return_20d = (
                        1 + daily_returns.rolling(window=rolling_window_20d).mean()
                    ) ** T - 1
                    rolling_annual_vol_20d = daily_returns.rolling(
                        window=rolling_window_20d
                    ).std() * np.sqrt(T)
                    rolling_sharpe_20d = (
                        rolling_annual_return_20d - risk_free_rate
                    ) / rolling_annual_vol_20d
                    downside_returns = daily_returns.where(daily_returns < 0, np.nan)
                    rolling_downside_std_20d = downside_returns.rolling(
                        window=rolling_window_20d
                    ).std() * np.sqrt(T)
                    rolling_sortino_20d = (
                        rolling_annual_return_20d - risk_free_rate
                    ) / rolling_downside_std_20d
                    delta = raw_data.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi_series = 100 - (100 / (1 + rs))

                    # --- Stage 3C: Chart Filter (Checkbox Table) ---
                    st.divider()
                    st.subheader("🎨 Chart Filter")

                    fund_list = raw_data.columns.tolist()

                    # 1. Create a DataFrame to manage the selection state
                    selection_df = pd.DataFrame(
                        {
                            "Select": [True] * len(fund_list),  # Default all to True
                            "Fund": fund_list,
                        }
                    )

                    st.info(
                        "Please check the funds you want to display in the charts below:"
                    )

                    # 2. Display the editable checkbox table
                    edited_df = st.data_editor(
                        selection_df,
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "Select", default=True
                            ),
                            "Fund": "Fund",
                        },
                        hide_index=True,
                        width=300,
                    )

                    # 3. Get the list of selected funds from the edited table
                    selected_funds = edited_df[edited_df["Select"]]["Fund"].tolist()

                    if not selected_funds:
                        st.warning("Please select at least one fund to display charts.")
                    else:
                        # --- Stage 3D: Plot all charts ---

                        st.subheader("📉 Cumulative Returns (Normalized)")
                        fig_cum_returns = go.Figure()
                        for col in selected_funds:
                            fig_cum_returns.add_trace(
                                go.Scatter(
                                    x=normalized_returns.index,
                                    y=normalized_returns[col],
                                    mode="lines",
                                    name=col,
                                )
                            )
                        fig_cum_returns.update_layout(hovermode="x unified")
                        st.plotly_chart(fig_cum_returns, use_container_width=True)

                        st.subheader("🌊 20-Day Rolling Volatility")
                        fig_roll_vol = go.Figure()
                        rolling_vol_clean = rolling_vol_20d.dropna()
                        for col in selected_funds:
                            if col in rolling_vol_clean.columns:
                                fig_roll_vol.add_trace(
                                    go.Scatter(
                                        x=rolling_vol_clean.index,
                                        y=rolling_vol_clean[col],
                                        mode="lines",
                                        name=col,
                                    )
                                )
                        fig_roll_vol.update_layout(hovermode="x unified")
                        st.plotly_chart(fig_roll_vol, use_container_width=True)

                        st.subheader("📉 Drawdown (Time Series)")
                        st.info(
                            "This chart shows the percentage drop from the fund's historical peak. 0% means a new all-time high."
                        )
                        fig_drawdown = go.Figure()
                        drawdown_series_clean = drawdown_series.dropna()
                        for col in selected_funds:
                            if col in drawdown_series_clean.columns:
                                fig_drawdown.add_trace(
                                    go.Scatter(
                                        x=drawdown_series_clean.index,
                                        y=drawdown_series_clean[col],
                                        mode="lines",
                                        name=col,
                                        fill="tozeroy",
                                    )
                                )
                        fig_drawdown.update_layout(
                            hovermode="x unified", yaxis_tickformat=".2%"
                        )
                        st.plotly_chart(fig_drawdown, use_container_width=True)

                        st.subheader("📊 20-Day Rolling Sharpe Ratio")
                        fig_roll_sharpe = go.Figure()
                        rolling_sharpe_clean = rolling_sharpe_20d.dropna()
                        for col in selected_funds:
                            if col in rolling_sharpe_clean.columns:
                                fig_roll_sharpe.add_trace(
                                    go.Scatter(
                                        x=rolling_sharpe_clean.index,
                                        y=rolling_sharpe_clean[col],
                                        mode="lines",
                                        name=col,
                                    )
                                )
                        fig_roll_sharpe.add_hline(
                            y=0, line_dash="dot", line_color="grey"
                        )
                        fig_roll_sharpe.update_layout(hovermode="x unified")
                        st.plotly_chart(fig_roll_sharpe, use_container_width=True)

                        st.subheader("📊 20-Day Rolling Sortino Ratio")
                        fig_roll_sortino = go.Figure()
                        rolling_sortino_clean = rolling_sortino_20d.dropna()
                        for col in selected_funds:
                            if col in rolling_sortino_clean.columns:
                                fig_roll_sortino.add_trace(
                                    go.Scatter(
                                        x=rolling_sortino_clean.index,
                                        y=rolling_sortino_clean[col],
                                        mode="lines",
                                        name=col,
                                    )
                                )
                        fig_roll_sortino.add_hline(
                            y=0, line_dash="dot", line_color="grey"
                        )
                        fig_roll_sortino.update_layout(hovermode="x unified")
                        st.plotly_chart(fig_roll_sortino, use_container_width=True)

                        st.subheader("🌡️ Relative Strength Index (RSI, 14-Day)")
                        fig_rsi = go.Figure()
                        rsi_series_clean = rsi_series.dropna()
                        for col in selected_funds:
                            if col in rsi_series_clean.columns:
                                fig_rsi.add_trace(
                                    go.Scatter(
                                        x=rsi_series_clean.index,
                                        y=rsi_series_clean[col],
                                        mode="lines",
                                        name=col,
                                    )
                                )
                        fig_rsi.add_hline(
                            y=70,
                            line_dash="dot",
                            line_color="red",
                            annotation_text="Overbought (70)",
                        )
                        fig_rsi.add_hline(
                            y=30,
                            line_dash="dot",
                            line_color="green",
                            annotation_text="Oversold (30)",
                        )
                        fig_rsi.update_layout(
                            hovermode="x unified", yaxis_range=[0, 100]
                        )
                        st.plotly_chart(fig_rsi, use_container_width=True)

                    # --- Stage 3E: Automated Interpretation ---
                    generate_interpretation(metrics_table, benchmark_column)

                else:
                    st.error(
                        "Data loading failed or the data is empty for the selected date range. Please check your CSV and parameters."
                    )
                    st.session_state.analysis_run = False  # Reset "flag"

    else:
        # If "flag" is True, but the file was removed
        st.warning("⚠️ Please upload a CSV file in the sidebar.")
        st.session_state.analysis_run = False  # Reset "flag"

else:
    # Default state (analysis_run == False)
    st.info(
        "Please set parameters in the left sidebar and upload a CSV file, then click 'Run Analysis'."
    )
