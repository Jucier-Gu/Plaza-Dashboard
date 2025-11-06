# app.py - 第 7 版 (CSV 每日价格 + Plotly + 复选框 + 修复页面消失)

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go  # 导入 Plotly


# --- 核心分析函数 (后端) ---
# [calculate_metrics_from_prices, generate_interpretation, load_data_from_csv]
# [这些函数与上一版完全相同，保持不变]
@st.cache_data
def calculate_metrics_from_prices(
    data_df, benchmark_ticker, risk_free_rate=0.02, cvar_alpha=0.05
):
    """
    计算所有关键风险指标 (用于顶部的表格)
    *** 假定 data_df 包含每日价格 ***
    *** 返回原始数字 (floats) ***
    """

    daily_returns = data_df.pct_change().dropna()

    if benchmark_ticker not in daily_returns.columns:
        st.error(f"错误：基准 '{benchmark_ticker}' 在CSV列中未找到。")
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

        fund_prices = data_df[fund]  # <-- 修复了 NameError
        peak = fund_prices.expanding(min_periods=1).max()
        drawdown = (fund_prices - peak) / peak  # <-- 修复了 NameError
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
                "基金/列名 (Fund)": fund,
                "年化收益 (Return)": annual_return,
                "波动率 (Vol)": annual_volatility,
                "贝塔 (Beta)": beta,
                "夏普 (Sharpe)": sharpe_ratio,
                "索提诺 (Sortino)": sortino_ratio,
                "最大回撤 (Max DD)": max_drawdown,
                "CVaR (5%)": cvar_95,
                "RSI (14天)": current_rsi,
            }
        )

    return pd.DataFrame(metrics_list).set_index("基金/列名 (Fund)")


def generate_interpretation(metrics_df, benchmark_ticker):
    """自动生成解读文本 (处理原始数字)"""
    benchmark_metrics = metrics_df.loc[benchmark_ticker]
    st.markdown("---")
    st.subheader("🤖 自动解读与建议")

    for fund in metrics_df.index:
        if fund == benchmark_ticker:
            continue

        fund_metrics = metrics_df.loc[fund]
        st.markdown(f"**关于 {fund} 的分析:**")

        try:
            if fund_metrics["夏普 (Sharpe)"] > benchmark_metrics["夏普 (Sharpe)"]:
                st.success(
                    f"📈 **风险调整后收益 (夏普):** {fund} ({fund_metrics['夏普 (Sharpe)']:.2f}) 优于基准 ({benchmark_metrics['夏普 (Sharpe)']:.2f})。"
                )
            else:
                st.warning(
                    f"📉 **风险调整后收益 (夏普):** {fund} ({fund_metrics['夏普 (Sharpe)']:.2f}) 落后于基准 ({benchmark_metrics['夏普 (Sharpe)']:.2f})。"
                )

            if (
                fund_metrics["最大回撤 (Max DD)"]
                > benchmark_metrics["最大回撤 (Max DD)"]
            ):
                st.success(
                    f"🛡️ **风险控制 (最大回撤):** {fund} ({fund_metrics['最大回撤 (Max DD)']:,.2%}) 的历史最大回撤小于基准 ({benchmark_metrics['最大回撤 (Max DD)']:,.2%})，表现出更好的抗跌性。"
                )
            else:
                st.warning(
                    f"🚩 **风险控制 (最大回撤):** {fund} ({fund_metrics['最大回撤 (Max DD)']:,.2%}) 的历史最大回撤大于基准 ({benchmark_metrics['最大回撤 (Max DD)']:,.2%})。"
                )

            rsi = fund_metrics["RSI (14天)"]
            if rsi > 70:
                st.warning(
                    f"🌡️ **短期信号 (RSI):** {fund} 当前的 RSI 为 {rsi:.2f}，处于“超买”区域，可能存在短期回调风险。"
                )
            elif rsi < 30:
                st.success(
                    f"🌡️ **短期信号 (RSI):** {fund} 当前的 RSI 为 {rsi:.2f}，处于“超卖”区域，可能存在短期反弹机会。"
                )
            else:
                st.info(
                    f"🌡️ **短期信号 (RSI):** {fund} 当前的 RSI 为 {rsi:.2f}，处于中性区域。"
                )

        except Exception as e:
            if pd.isna(fund_metrics["夏普 (Sharpe)"]) or pd.isna(
                fund_metrics["RSI (14天)"]
            ):
                st.warning(
                    f"无法为 {fund} 生成部分解读（数据不足，例如 RSI 无法计算）。"
                )
            else:
                st.error(f"为 {fund} 生成解读时出错: {e}。")


def load_data_from_csv(uploaded_file, date_column, start_date, end_date):
    """从上传的 CSV 加载、解析和过滤数据"""
    try:
        df = pd.read_csv(uploaded_file)
        if date_column not in df.columns:
            st.error(f"错误: 在 CSV 中未找到指定的日期列 '{date_column}'。")
            return None
        df["Date_Parsed"] = pd.to_datetime(df[date_column])
        df = df.set_index("Date_Parsed")
        df = df.loc[start_date:end_date]
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            st.error("在指定日期范围内没有找到数值数据。")
            return None
        return df_numeric.dropna(axis=1, how="all")
    except Exception as e:
        st.error(f"处理 CSV 文件时出错: {e}")
        return None


# --- Streamlit 应用程序 UI (前端) ---

st.set_page_config(layout="wide", page_title="Plaza 基金风险仪表盘")
st.title("📈 Plaza 自动化基金风险分析")
st.info("✅ CSV 必须包含 **每日价格**。图表使用 Plotly 渲染。")

# --- (新增) 修复 1: 初始化会话状态 ---
# (这个 "flag" 会在 streamlit 刷新时被 "记住")
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

# --- 1. 输入 (侧边栏) ---
st.sidebar.header("⚙️ 控制面板")
uploaded_file = st.sidebar.file_uploader("1. 上传您的基金数据 (CSV)", type=["csv"])
st.sidebar.info("CSV 必须包含 '日期' 列和每日 **价格** 列。")
date_column = st.sidebar.text_input("2. 输入 CSV 中的'日期'列名", "Date")
benchmark_column = st.sidebar.text_input("3. 输入 CSV 中的'基准'列名", "SP500")

start_date = st.sidebar.date_input("4. 选择分析开始日期", pd.to_datetime("2020-10-13"))
end_date = st.sidebar.date_input("5. 选择分析结束日期", pd.to_datetime("2025-10-16"))

risk_free_rate = st.sidebar.slider("6. 年化无风险利率 (%)", 0.0, 5.0, 2.0, 0.1) / 100
run_button = st.sidebar.button("🚀 运行分析")

# --- (新增) 修复 2: 更新主逻辑 ---

# 当按钮被点击时，设置 "flag"
if run_button:
    st.session_state.analysis_run = True

# 检查 "flag"，而不是检查 "run_button"
if st.session_state.analysis_run:

    # 检查文件是否已上传
    if uploaded_file is not None:
        with st.spinner("正在加载数据并计算指标..."):

            # 阶段 2A: 加载和准备数据
            raw_data = load_data_from_csv(
                uploaded_file, date_column, start_date, end_date
            )

            if raw_data is not None and not raw_data.empty:
                # 阶段 2B: 计算
                metrics_table = calculate_metrics_from_prices(
                    raw_data, benchmark_column, risk_free_rate
                )

                if not metrics_table.empty:
                    # 阶段 3A: 显示核心指标表
                    st.subheader("📊 核心风险指标对比")
                    st.dataframe(
                        metrics_table.style.format(
                            {
                                "年化收益 (Return)": "{:.2%}",
                                "波动率 (Vol)": "{:.2%}",
                                "贝塔 (Beta)": "{:.2f}",
                                "夏普 (Sharpe)": "{:.2f}",
                                "索提诺 (Sortino)": "{:.2f}",
                                "最大回撤 (Max DD)": "{:.2%}",
                                "CVaR (5%)": "{:.2%}",
                                "RSI (14天)": "{:.2f}",
                            }
                        )
                    )

                    # --- 阶段 3B: 准备图表所需的所有时间序列数据 ---
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

                    # --- 阶段 3C: 图表过滤器 (复选框表格) ---
                    st.divider()
                    st.subheader("🎨 图表过滤器")

                    fund_list = raw_data.columns.tolist()

                    selection_df = pd.DataFrame(
                        {
                            "Select": [True] * len(fund_list),  # 默认全选
                            "Fund": fund_list,
                        }
                    )

                    st.info("请在下表中勾选您想在图表中查看的基金：")

                    edited_df = st.data_editor(
                        selection_df,
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "勾选", default=True
                            ),
                            "Fund": "基金",
                        },
                        hide_index=True,
                        width=300,
                    )

                    selected_funds = edited_df[edited_df["Select"]]["Fund"].tolist()

                    if not selected_funds:
                        st.warning("请至少选择一只基金来显示图表。")
                    else:
                        # --- 阶段 3D: 绘制所有图表 ---
                        # (这部分代码保持不变，它会正确地使用 selected_funds)

                        st.subheader("📉 累计收益走势 (归一化)")
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

                        st.subheader("🌊 滚动波动率 (20天)")
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

                        st.subheader("📉 最大回撤时序图")
                        st.info(
                            "这张图显示了基金从其历史高点回撤的百分比。0% 意味着处于历史新高。"
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

                        st.subheader("📊 滚动夏普比率 (20天)")
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

                        st.subheader("📊 滚动索提诺比率 (20天)")
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

                        st.subheader("🌡️ 相对强弱指数 (RSI, 14天)")
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
                            annotation_text="超买 (70)",
                        )
                        fig_rsi.add_hline(
                            y=30,
                            line_dash="dot",
                            line_color="green",
                            annotation_text="超卖 (30)",
                        )
                        fig_rsi.update_layout(
                            hovermode="x unified", yaxis_range=[0, 100]
                        )
                        st.plotly_chart(fig_rsi, use_container_width=True)

                    # --- 阶段 3E: 自动解读 ---
                    generate_interpretation(metrics_table, benchmark_column)

            else:
                st.error("数据加载失败或在指定日期范围内为空。请检查 CSV 和参数。")
                st.session_state.analysis_run = False  # 重置 "flag"

    else:
        # 如果 "flag" 是 True，但文件被移除了
        st.warning("⚠️ 请在侧边栏上传一个 CSV 文件。")
        st.session_state.analysis_run = False  # 重置 "flag"

else:
    # 默认状态 (analysis_run == False)
    st.info("请在左侧侧边栏设置参数并上传 CSV 文件，然后点击“运行分析”。")
