# -*- coding: utf-8 -*-
"""
因子调仓回测（V1.3 Step A4）。

基于因子综合得分定期调仓：
- 每个调仓日计算股票池的因子截面 → 标准化 → 合成综合得分；
- 选取得分最高的 top_n 只股票等权（或得分加权）持有到下一调仓日；
- 记录组合净值、换手率、持仓明细，并与等权全市场基准对比。

防前视约束：
- 因子值只使用调仓日 t 及之前行情；
- IC 加权时，权重只使用 t 之前的 IC 历史（滚动窗口 ic_lookback）；
- 持仓收益为 t 日收盘到下一调仓日收盘的收益。

仅研究用途，不构成投资建议。
"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.factor_engine import (
    FACTOR_REGISTRY, compute_factor, standardize_cross_section,
)
from core.factor_analysis import (
    load_all_daily, get_rebalance_dates, compute_forward_returns,
)
from core.factor_combine import (
    compute_ic_weights, combine_factors, infer_directions,
)

DISCLAIMER = "仅研究用途，不构成投资建议"

# 各调仓频率的默认 IC 滚动窗口（按交易日数等比缩放：周频 5 日 → 52 期）
IC_LOOKBACK_BY_FREQ = {"W": 52, "2W": 26, "M": 12}

# 各调仓频率的每年周期数：用于年化收益与夏普换算（周频约 52 期、双周 26 期、月频 12 期）
PERIODS_PER_YEAR = {"W": 52, "2W": 26, "M": 12}


def _finite(value):
    """把非有限数转为 None，便于 JSON 序列化。"""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


# ---------------------------------------------------------------------------
# 调仓回测主流程
# ---------------------------------------------------------------------------

def _rebalance_dates(codes, freq, fq):
    """
    生成调仓日序列，支持周频 W / 双周频 2W / 月频 M。

    2W 在周频调仓日基础上隔周取样，保证与周频共用同一套交易日历。
    """
    if freq == "2W":
        weekly = get_rebalance_dates(codes, freq="W", fq=fq)
        return weekly[::2]
    if freq not in ("W", "M"):
        raise ValueError("freq 只支持 'W'（周）/ '2W'（双周）/ 'M'（月）")
    return get_rebalance_dates(codes, freq=freq, fq=fq)


def run_factor_backtest(codes, factor_names, start_date=None, end_date=None,
                        freq="W", top_n=20, method="ic", weighting="equal",
                        fq="qfq", ic_lookback=None, directions=None, progress_cb=None):
    """
    运行因子调仓回测。

    参数：
        codes: 股票池代码列表
        factor_names: 因子名列表
        start_date / end_date: 回测区间（YYYY-MM-DD），None 表示不限制
        freq: 调仓频率 "W" 周频 / "2W" 双周频 / "M" 月频
        top_n: 每期持仓股票数
        method: 合成方法 "equal" 等权 / "ic" IC 加权
        weighting: 持仓加权 "equal" 等权 / "score" 按得分加权
        fq: 复权口径
        ic_lookback: IC 加权时的滚动窗口（调仓日数），None 表示按频率自动缩放
                     （周频 52 / 双周 26 / 月频 12）
        directions: 因子方向 dict {factor_name: 1/-1}，仅 method="equal" 生效；
                    用于按筛查结论固定因子方向（-1 表示因子越小越看多）
        progress_cb: 进度回调 callback(done, total, text)

    返回 dict:
        nav: Series（调仓日净值，初始 1.0）
        benchmark_nav: Series（等权全市场基准净值）
        holdings: DataFrame（index=调仓日，columns=持仓代码列表）
        turnover: Series（每期换手率）
        metrics: dict（总收益、年化、夏普、最大回撤、换手率均值、胜率）
        rebalance_dates: 调仓日列表
        ic_history: list（每个调仓日使用的 IC 权重，可追溯防前视）
    """
    for name in factor_names:
        if name not in FACTOR_REGISTRY:
            raise ValueError("未注册的因子：%s" % name)

    data_map = load_all_daily(codes, fq=fq)
    codes = list(data_map.keys())
    if not codes:
        raise ValueError("无可用日线数据")

    rebalance_dates = _rebalance_dates(codes, freq=freq, fq=fq)
    if start_date:
        rebalance_dates = [d for d in rebalance_dates if d >= start_date]
    if end_date:
        rebalance_dates = [d for d in rebalance_dates if d <= end_date]
    if len(rebalance_dates) < 3:
        raise ValueError("调仓日不足，无法回测")

    # IC 滚动窗口按调仓频率等比缩放（周频约 5 个交易日、双周 10 个、月频 21 个）
    if ic_lookback is None:
        ic_lookback = IC_LOOKBACK_BY_FREQ[freq]

    total = len(rebalance_dates)
    nav = [1.0]
    benchmark_nav = [1.0]
    holdings_records = []
    turnover_list = []
    ic_history = []
    prev_holdings = set()
    # IC 历史：每个调仓日计算的当期 IC，供后续调仓日使用
    ic_history_by_factor = {name: [] for name in factor_names}

    for i, trade_date in enumerate(rebalance_dates):
        if progress_cb:
            progress_cb(i, total, "回测 %s" % trade_date)

        # 1. 计算因子截面
        factor_values = {}
        for name in factor_names:
            vals = {}
            for code in codes:
                df = data_map[code]
                df_t = df[df["date"] <= trade_date]
                vals[code] = compute_factor(df_t, name) if not df_t.empty else np.nan
            factor_values[name] = pd.Series(vals)
        std_frame = pd.DataFrame(factor_values)
        std_frame = standardize_cross_section(std_frame)

        # 2. 合成综合得分
        if method == "equal":
            weights = None
            score = combine_factors(std_frame, method="equal", directions=directions)
        elif method == "ic":
            # 只用 trade_date 之前的 IC 计算权重（防前视）
            ic_series_dict = {}
            for name in factor_names:
                ic_vals = ic_history_by_factor[name]
                ic_series_dict[name] = pd.DataFrame({"ic": ic_vals})
            weights = compute_ic_weights(ic_series_dict, window=ic_lookback)
            # 若 IC 权重全为 0（如首期无历史 IC），回退到等权合成
            if all(abs(w) < 1e-12 for w in weights.values()):
                score = combine_factors(std_frame, method="equal", directions=directions)
            else:
                score = combine_factors(std_frame, method="ic", weights=weights)
        else:
            raise ValueError("未知合成方法：%s" % method)

        ic_history.append({"date": trade_date, "weights": weights})

        # 3. 选股：得分最高的 top_n 只
        score = score.dropna().sort_values(ascending=False)
        if len(score) == 0:
            holdings_records.append([])
            turnover_list.append(0.0)
            nav.append(nav[-1])
            benchmark_nav.append(benchmark_nav[-1])
            # 仍然更新 IC 历史（用当期实际 IC）
            _update_ic_history(ic_history_by_factor, factor_values, data_map,
                               trade_date, factor_names)
            continue

        selected = score.head(top_n).index.tolist()

        # 4. 计算持仓收益（trade_date 收盘到下一调仓日收盘）
        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
            holding_returns = {}
            for code in selected:
                df = data_map[code]
                dates = df["date"].values
                if trade_date not in dates or next_date not in dates:
                    holding_returns[code] = np.nan
                    continue
                idx_t = int(np.where(dates == trade_date)[0][0])
                idx_n = int(np.where(dates == next_date)[0][0])
                close_t = float(df.iloc[idx_t]["close"])
                close_n = float(df.iloc[idx_n]["close"])
                holding_returns[code] = (close_n / close_t - 1) if close_t > 0 else np.nan

            if weighting == "score":
                # 按得分加权（得分越高权重越大）
                sel_scores = score.loc[selected]
                weights_arr = sel_scores.values
                weights_arr = weights_arr - weights_arr.min() + 1e-8  # 平移为正
                weights_arr = weights_arr / weights_arr.sum()
                valid_mask = ~np.isnan(list(holding_returns.values()))
                if valid_mask.sum() > 0:
                    rets = np.array([holding_returns[c] for c in selected])
                    port_ret = float(np.nansum(rets * weights_arr) / np.nansum(weights_arr * ~np.isnan(rets).astype(float)))
                else:
                    port_ret = 0.0
            else:
                # 等权
                rets = [holding_returns[c] for c in selected]
                valid_rets = [r for r in rets if not np.isnan(r)]
                port_ret = float(np.mean(valid_rets)) if valid_rets else 0.0

            # 基准：全市场等权
            all_rets = []
            for code in codes:
                df = data_map[code]
                dates = df["date"].values
                if trade_date in dates and next_date in dates:
                    idx_t = int(np.where(dates == trade_date)[0][0])
                    idx_n = int(np.where(dates == next_date)[0][0])
                    close_t = float(df.iloc[idx_t]["close"])
                    close_n = float(df.iloc[idx_n]["close"])
                    if close_t > 0:
                        all_rets.append(close_n / close_t - 1)
            bench_ret = float(np.mean(all_rets)) if all_rets else 0.0
        else:
            port_ret = 0.0
            bench_ret = 0.0

        # 5. 换手率
        current_holdings = set(selected)
        if prev_holdings:
            changed = len(current_holdings.symmetric_difference(prev_holdings))
            turnover = changed / (2 * max(top_n, 1))
        else:
            turnover = 1.0  # 首次建仓换手率 100%
        prev_holdings = current_holdings

        holdings_records.append(selected)
        turnover_list.append(turnover)
        nav.append(nav[-1] * (1 + port_ret))
        benchmark_nav.append(benchmark_nav[-1] * (1 + bench_ret))

        # 6. 更新 IC 历史（用当期实际因子值与未来收益的 IC，供下一期使用）
        _update_ic_history(ic_history_by_factor, factor_values, data_map,
                           trade_date, factor_names)

    # 构建结果
    nav_series = pd.Series(nav[1:], index=rebalance_dates, name="nav")
    bench_series = pd.Series(benchmark_nav[1:], index=rebalance_dates, name="benchmark_nav")
    turnover_series = pd.Series(turnover_list, index=rebalance_dates, name="turnover")

    # 持仓 DataFrame
    max_hold = max(len(h) for h in holdings_records) if holdings_records else 0
    holdings_df = pd.DataFrame(
        [{i: h[i] if i < len(h) else None for i in range(max_hold)} for h in holdings_records],
        index=rebalance_dates,
    )

    # 绩效指标（年化与夏普按调仓频率换算周期数，避免低频调仓被高估）
    per_year = PERIODS_PER_YEAR.get(freq, 52)
    period_returns = nav_series.pct_change().dropna()
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    periods = len(nav_series)
    annual_return = (1 + total_return) ** (per_year / periods) - 1 if periods > 0 else 0.0
    sharpe = (period_returns.mean() / period_returns.std(ddof=1) * math.sqrt(per_year)
              if len(period_returns) > 1 and period_returns.std(ddof=1) > 0 else 0.0)
    running_max = nav_series.cummax()
    max_drawdown = ((nav_series / running_max) - 1).min()
    win_rate = float((period_returns > 0).mean()) if len(period_returns) > 0 else 0.0
    avg_turnover = float(np.mean(turnover_list)) if turnover_list else 0.0

    # 基准绩效
    bench_total = bench_series.iloc[-1] / bench_series.iloc[0] - 1
    excess_return = total_return - bench_total

    metrics = {
        "total_return": _finite(total_return),
        "annual_return": _finite(annual_return),
        "sharpe": _finite(sharpe),
        "max_drawdown": _finite(abs(max_drawdown)),
        "win_rate": _finite(win_rate),
        "avg_turnover": _finite(avg_turnover),
        "total_turnover": _finite(float(np.sum(turnover_list)) if turnover_list else 0.0),
        "periods": periods,
        "periods_per_year": per_year,
        "benchmark_total_return": _finite(bench_total),
        "excess_return": _finite(excess_return),
    }

    return {
        "nav": nav_series,
        "benchmark_nav": bench_series,
        "holdings": holdings_df,
        "turnover": turnover_series,
        "metrics": metrics,
        "rebalance_dates": rebalance_dates,
        "ic_history": ic_history,
    }


def _update_ic_history(ic_history_by_factor, factor_values, data_map, trade_date, factor_names):
    """用当期因子值与未来 5 日收益计算 IC，追加到历史，供后续调仓日使用。"""
    from core.factor_analysis import compute_forward_returns
    fwd_ret = compute_forward_returns(data_map, trade_date, n=5)
    for name in factor_names:
        aligned = pd.concat([factor_values[name], fwd_ret], axis=1).dropna()
        aligned.columns = ["factor", "return"]
        if len(aligned) >= 10:
            ic = aligned["factor"].corr(aligned["return"], method="spearman")
            ic_history_by_factor[name].append(float(ic) if pd.notna(ic) else np.nan)
        else:
            ic_history_by_factor[name].append(np.nan)


# ---------------------------------------------------------------------------
# 行业分布统计
# ---------------------------------------------------------------------------

def industry_distribution(holdings_df, data_map=None):
    """
    统计各调仓日的持仓行业分布。
    本地 stock_info.industry 可能为空，返回原始分布。
    返回 DataFrame：index=调仓日，columns=行业，值=该行业持仓数。
    """
    industries_map = {}
    try:
        rows = db.query_all("SELECT code, industry FROM stock_info")
        industries_map = {r["code"]: (r["industry"] or "未知") for r in rows}
    except Exception:
        pass

    records = []
    for date in holdings_df.index:
        row = holdings_df.loc[date]
        codes = [c for c in row.tolist() if c is not None and pd.notna(c)]
        ind_counts = {}
        for code in codes:
            ind = industries_map.get(code, "未知")
            ind_counts[ind] = ind_counts.get(ind, 0) + 1
        rec = {"date": date}
        rec.update(ind_counts)
        records.append(rec)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("date").fillna(0)
    return df


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()

    # 取前 100 只股票，最近 2 年周频
    rows = db.query_all("SELECT DISTINCT code FROM daily_bars WHERE fq='qfq' ORDER BY code LIMIT 100")
    codes = [r["code"] for r in rows]
    print("测试股票数:", len(codes))

    factors = ["return_5", "volatility_20", "rsi14", "volume_ratio_5"]

    # 用最近 3 年数据，收益数字更直观
    result = run_factor_backtest(
        codes=codes, factor_names=factors,
        start_date="2023-11-01", freq="W", top_n=20, method="ic",
        weighting="equal", fq="qfq", ic_lookback=26,
    )

    print("\n=== 调仓回测结果 ===")
    m = result["metrics"]
    print("调仓期数: %d" % m["periods"])
    print("总收益: %.2f%%" % (m["total_return"] * 100))
    print("年化收益: %.2f%%" % (m["annual_return"] * 100))
    print("夏普比率: %.4f" % m["sharpe"])
    print("最大回撤: %.2f%%" % (m["max_drawdown"] * 100))
    print("胜率: %.2f%%" % (m["win_rate"] * 100))
    print("平均换手率: %.2f%%" % (m["avg_turnover"] * 100))
    print("基准总收益: %.2f%%" % (m["benchmark_total_return"] * 100))
    print("超额收益: %.2f%%" % (m["excess_return"] * 100))

    print("\n净值序列首尾:")
    print("  起始: %.4f" % result["nav"].iloc[0])
    print("  结束: %.4f" % result["nav"].iloc[-1])
    print("  基准结束: %.4f" % result["benchmark_nav"].iloc[-1])

    print("\n首期持仓(前5):", result["holdings"].iloc[0].dropna().tolist()[:5])

    # 防前视验证：首期 IC 权重应该为 0（无历史 IC）
    first_weights = result["ic_history"][0]["weights"]
    print("\n首期 IC 权重(应为全0，无历史):", {k: "%.4f" % v for k, v in first_weights.items()})
    assert all(abs(v) < 1e-10 for v in first_weights.values()), "首期不应有非零 IC 权重"

    # 净值合理性：不能全为 1.0（应该有波动）
    assert result["nav"].std() > 0, "净值无波动，回测可能异常"

    # 换手率在 [0, 1] 范围内
    assert result["turnover"].between(0, 1).all(), "换手率超出 [0,1]"

    # 调仓频率扩展验证：周频 / 双周频 / 月频（月频期数最少，换手应更低）
    print("\n=== 调仓频率对比（周 / 双周 / 月）===")
    summary = {}
    for freq in ("W", "2W", "M"):
        res = run_factor_backtest(
            codes=codes, factor_names=factors, start_date="2023-11-01",
            freq=freq, top_n=20, method="ic", weighting="equal", fq="qfq",
        )
        fm = res["metrics"]
        summary[freq] = fm
        print("%-3s 期数=%-4d 年化=%6.2f%% 夏普=%6.3f 换手=%6.2f%% 超额=%7.2f%%" % (
            freq, fm["periods"], fm["annual_return"] * 100, fm["sharpe"],
            fm["avg_turnover"] * 100, fm["excess_return"] * 100))
        assert fm["periods"] >= 3, "%s 调仓期数不足" % freq
        assert 0 <= fm["avg_turnover"] <= 1.01, "%s 换手率异常" % freq

    # 期数应满足 周 > 双周 > 月；换手用「累计换手」比较（单期换手在低频下反而更高）
    assert summary["W"]["periods"] > summary["2W"]["periods"] > summary["M"]["periods"], \
        "调仓期数不符合 周 > 双周 > 月"
    assert summary["W"]["total_turnover"] > summary["2W"]["total_turnover"] > summary["M"]["total_turnover"], \
        "累计换手应满足 周 > 双周 > 月"
    print("IC 窗口自动缩放: %s" % IC_LOOKBACK_BY_FREQ)

    print("\nfactor_backtest.py 自测通过；%s" % DISCLAIMER)
